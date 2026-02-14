from typing import Any, Annotated, Sequence
import uuid
from abc import abstractmethod, ABC
import textwrap
from enum import Enum, auto
from pydantic import ConfigDict, Field, PlainSerializer, model_validator, field_validator
from serapeum.core.utils.base import truncate_text
from serapeum.core.types import SerializableModel

TRUNCATE_LENGTH = 350
WRAP_WIDTH = 70
DEFAULT_METADATA_TMPL = "{key}: {value}"


class NodeContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    INDEX = "index"
    DOCUMENT = "document"
    MULTIMODAL = "multimodal"


EnumNameSerializer = PlainSerializer(
    lambda e: e.value, return_type="str", when_used="always"
)


class NodeInfo(SerializableModel):
    id: str
    type: Annotated[NodeContentType, EnumNameSerializer] | str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    hash: str | None = None

    @classmethod
    def class_name(cls) -> str:
        return "NodeInfo"


NodeInfoType = NodeInfo | list[NodeInfo]


class MetadataMode(str, Enum):
    ALL = "all"
    EMBED = "embed"
    LLM = "llm"
    NONE = "none"


class NodeType(str, Enum):
    """
    Node links used in `BaseNode` class.

    Attributes:
        SOURCE: The node is the source document.
        PREVIOUS: The node is the previous node in the document.
        NEXT: The node is the next node in the document.
        PARENT: The node is the parent node in the document.
        CHILD: The node is a child node in the document.

    """

    SOURCE = "source"
    PREVIOUS = "previous"
    NEXT = "next"
    PARENT = "parent"
    CHILD = "child"


class LinkedNodes(SerializableModel):
    model_config = ConfigDict(frozen=True)

    source: NodeInfo | None = None
    previous: NodeInfo | None = None
    next: NodeInfo | None = None
    parent: NodeInfo | None = None
    children: list[NodeInfo] | None = None

    @field_validator('source', 'previous', 'next', 'parent')
    @classmethod
    def validate_single_node(cls, v: Any) -> NodeInfo | None:
        """Validate that single-node fields contain NodeInfo objects.

        Uses Pydantic's field_validator instead of manual validation.
        Applies to: source, previous, next, parent fields.
        """
        if v is not None and not isinstance(v, NodeInfo):
            raise ValueError("Must be a NodeInfo object, not a list")
        return v

    @field_validator('children')
    @classmethod
    def validate_children_list(cls, v: Any) -> list[NodeInfo] | None:
        """Validate that children field contains a list of NodeInfo objects.

        Uses Pydantic's field_validator instead of manual validation.
        """
        if v is not None and not isinstance(v, list):
            raise ValueError("Children must be a list of NodeInfo objects")
        return v

    @classmethod
    def create(
        cls, linked_nodes_info: dict[NodeType, NodeInfoType]
    ) -> "LinkedNodes":
        """Create LinkedNodes from a dict mapping NodeType to NodeInfo/list.

        Pydantic validators automatically handle type checking for each field.
        """
        return cls(
            source=linked_nodes_info.get(NodeType.SOURCE),
            previous=linked_nodes_info.get(NodeType.PREVIOUS),
            next=linked_nodes_info.get(NodeType.NEXT),
            parent=linked_nodes_info.get(NodeType.PARENT),
            children=linked_nodes_info.get(NodeType.CHILD),
        )

    def as_dict(self) -> dict[NodeType, NodeInfoType | None]:
        linked_nodes = {
            NodeType.SOURCE: self.source,
            NodeType.PREVIOUS: self.previous,
            NodeType.NEXT: self.next,
            NodeType.PARENT: self.parent,
            NodeType.CHILD: self.children,
        }

        linked_nodes = {
            key: value for key, value in linked_nodes.items() if value is not None
        }
        return linked_nodes

    @property
    def source_id(self) -> str | None:
        source_id = None
        if self.source is not None:
            source_id = self.source.id
        return source_id


class BaseNode(SerializableModel, ABC):
    """Base node Object.

    Attributes:
        metadata fields
            - injected as part of the text shown to LLMs as context
            - injected as part of the text for generating embeddings
            - used by vector DBs for metadata filtering

    """
    # hash is computed on a local field, during the validation process
    model_config = ConfigDict(populate_by_name=True, validate_assignment=True)

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique ID of the node."
    )
    embedding: list[float] | None = Field(
        default=None, description="Embedding of the node."
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="A flat dictionary of metadata fields",
    )
    excluded_embed_metadata_keys: list[str] = Field(
        default_factory=list,
        description="Metadata keys that are excluded from text for the embed model.",
    )
    excluded_llm_metadata_keys: list[str] = Field(
        default_factory=list,
        description="Metadata keys that are excluded from text for the LLM.",
    )
    links: dict[
        Annotated[NodeType, EnumNameSerializer],
        NodeInfoType,
    ] = Field(
        default_factory=dict,
        description="A mapping of links to other nodes.",
    )
    metadata_template: str = Field(
        default=DEFAULT_METADATA_TMPL,
        description=(
            "Template for how metadata is formatted, with {key} and "
            "{value} placeholders."
        ),
    )
    metadata_separator: str = Field(
        default="\n",
        description="Separator between metadata fields when converting to string.",
    )

    linked_nodes_cache: LinkedNodes | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="Cached LinkedNodes object, invalidated when links change.",
    )

    # Track the links dict id to detect changes
    links_dict_id: int | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="ID of the links dict to detect when it's reassigned.",
    )

    @model_validator(mode='after')
    def _invalidate_linked_nodes_cache_on_links_change(self) -> 'BaseNode':
        """Invalidate the linked_nodes cache when links dict is reassigned.

        This validator tracks the id of the links dict. When it changes
        (i.e., links is reassigned), the cache is cleared.

        Uses Pydantic v2's @model_validator with object.__setattr__ to avoid recursion.
        """
        current_links_id = id(self.links)

        # Check if links dict was reassigned (different id)
        if self.links_dict_id is None or self.links_dict_id != current_links_id:
            # Links changed, clear cache and update tracked id
            object.__setattr__(self, 'linked_nodes_cache', None)
            object.__setattr__(self, 'links_dict_id', current_links_id)

        return self

    @classmethod
    @abstractmethod
    def get_type(cls) -> str:
        """Get Object type."""

    @abstractmethod
    def get_content(self, metadata_mode: MetadataMode = MetadataMode.ALL) -> str:
        """Get object content."""

    def get_metadata_str(self, mode: MetadataMode = MetadataMode.ALL) -> str:
        """Metadata info string."""
        if mode == MetadataMode.NONE:
            return ""

        excluded = set()
        if mode == MetadataMode.LLM:
            excluded = set(self.excluded_llm_metadata_keys)
        elif mode == MetadataMode.EMBED:
            excluded = set(self.excluded_embed_metadata_keys)

        filtered = (
            self.metadata.items()
            if not excluded
            else ((key, value) for key, value in self.metadata.items() if key not in excluded)
        )
        return self.metadata_separator.join(
            self.metadata_template.format(key=key, value=str(value))
            for key, value in filtered
        )

    @abstractmethod
    def set_content(self, value: Any) -> None:
        """Set the content of the node."""

    @property
    @abstractmethod
    def hash(self) -> str:
        """Get hash of node."""

    @property
    def source_id(self) -> str | None:
        return self.linked_nodes.source_id

    def _clear_linked_nodes_cache(self) -> None:
        """Manually clear the linked_nodes cache.

        Call this method if you mutate the links dict in-place.
        This is necessary because Pydantic's field validators only trigger
        on field assignment, not on in-place mutations.

        Example:
            >>> node.links[NodeType.SOURCE] = NodeInfo(id="new")
            >>> node._clear_linked_nodes_cache()  # Clear cache after in-place mutation
            >>> linked = node.linked_nodes  # Will recompute from updated links
        """
        self.linked_nodes_cache = None

    @property
    def linked_nodes(self) -> LinkedNodes:
        """Get linked nodes from the links dictionary.

        This property validates and converts the links dictionary into a
        LinkedNodes object. The result is cached and automatically invalidated
        when the links field is reassigned through Pydantic's field validation.

        Returns:
            LinkedNodes: A validated and cached LinkedNodes object.

        Note:
            - Cache is automatically cleared when `links` is reassigned
            - For in-place mutations (e.g., node.links[key] = value), you must
              either reassign the entire dict OR call _clear_linked_nodes_cache()
            - Uses Pydantic's @field_validator to manage cache invalidation

        Example:
            >>> # Automatic cache invalidation (reassignment)
            >>> node.links = {NodeType.SOURCE: ref}  # Cache auto-cleared

            >>> # Manual cache clear needed (in-place mutation)
            >>> node.links[NodeType.SOURCE] = ref
            >>> node._clear_linked_nodes_cache()

            >>> # Or reassign to trigger validation
            >>> node.links = node.links  # Cache auto-cleared
        """
        if self.linked_nodes_cache is None:
            # Compute and cache the LinkedNodes
            self.linked_nodes_cache = LinkedNodes.create(self.links)
        return self.linked_nodes_cache


    def __str__(self) -> str:
        source_text_truncated = truncate_text(
            self.get_content().strip(), TRUNCATE_LENGTH
        )
        source_text_wrapped = textwrap.fill(
            f"Text: {source_text_truncated}\n", width=WRAP_WIDTH
        )
        return f"Node ID: {self.id}\n{source_text_wrapped}"

    def get_embedding(self) -> list[float]:
        """
        Get embedding.

        Errors if embedding is None.

        """
        if self.embedding is None:
            raise ValueError("embedding not set.")
        return self.embedding

    def get_node_info(self) -> NodeInfo:
        """Get node info."""
        return NodeInfo(
            id=self.id,
            type=self.get_type(),
            metadata=self.metadata,
            hash=self.hash,
        )


class CallMixin(ABC):
    """Base class for transform components."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        """Transform nodes."""

    async def acall(
        self, nodes: Sequence[BaseNode], **kwargs: Any
    ) -> Sequence[BaseNode]:
        """Async transform nodes."""
        return self.__call__(nodes, **kwargs)
