from typing import Any, Annotated, Sequence, ClassVar
import uuid
from abc import abstractmethod, ABC
import textwrap
from enum import Enum, auto
from pydantic import ConfigDict, Field, PlainSerializer
from serapeum.core.utils.base import truncate_text
from serapeum.core.types import SerializableModel

TRUNCATE_LENGTH = 350
WRAP_WIDTH = 70
DEFAULT_METADATA_TMPL = "{key}: {value}"


class NodeType(str, Enum):
    TEXT = auto()
    IMAGE = auto()
    INDEX = auto()
    DOCUMENT = auto()
    MULTIMODAL = auto()


EnumNameSerializer = PlainSerializer(
    lambda e: e.value, return_type="str", when_used="always"
)


class NodeReference(SerializableModel):
    id: str
    type: Annotated[NodeType, EnumNameSerializer] | str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    hash: str | None = None

    @classmethod
    def class_name(cls) -> str:
        return "NodeReference"


RelatedNodeType = NodeReference | list[NodeReference]


class MetadataMode(str, Enum):
    ALL = "all"
    EMBED = "embed"
    LLM = "llm"
    NONE = "none"


class NodeRelationship(str, Enum):
    """
    Node relationships used in `BaseNode` class.

    Attributes:
        SOURCE: The node is the source document.
        PREVIOUS: The node is the previous node in the document.
        NEXT: The node is the next node in the document.
        PARENT: The node is the parent node in the document.
        CHILD: The node is a child node in the document.

    """

    SOURCE = auto()
    PREVIOUS = auto()
    NEXT = auto()
    PARENT = auto()
    CHILD = auto()


class LinkedNodes(SerializableModel):
    model_config = ConfigDict(frozen=True)

    SOURCE_ERROR: ClassVar[str] = (
        "The Source Node must be a single NodeReference object"
    )
    PREVIOUS_ERROR: ClassVar[str] = (
        "The Previous Node must be a single NodeReference object"
    )
    NEXT_ERROR: ClassVar[str] = "The Next Node must be a single NodeReference object"
    PARENT_ERROR: ClassVar[str] = (
        "The Parent Node must be a single NodeReference object"
    )
    CHILDREN_ERROR: ClassVar[str] = (
        "Child Nodes must be a list of NodeReference objects."
    )

    source: NodeReference | None = None
    previous: NodeReference | None = None
    next: NodeReference | None = None
    parent: NodeReference | None = None
    children: list[NodeReference] | None = None

    @classmethod
    def from_relationships(
        cls, relationships: dict[NodeRelationship, RelatedNodeType]
    ) -> "LinkedNodes":
        linked = cls(
            source=cls._get_single(
                relationships, NodeRelationship.SOURCE, cls.SOURCE_ERROR
            ),
            previous=cls._get_single(
                relationships, NodeRelationship.PREVIOUS, cls.PREVIOUS_ERROR
            ),
            next=cls._get_single(
                relationships, NodeRelationship.NEXT, cls.NEXT_ERROR
            ),
            parent=cls._get_single(
                relationships, NodeRelationship.PARENT, cls.PARENT_ERROR
            ),
            children=cls._get_list(
                relationships, NodeRelationship.CHILD, cls.CHILDREN_ERROR
            ),
        )
        return linked

    @staticmethod
    def _get_single(
        relationships: dict[NodeRelationship, RelatedNodeType],
        relationship: NodeRelationship,
        error_message: str,
    ) -> NodeReference | None:
        value = relationships.get(relationship)
        if value is not None and not isinstance(value, NodeReference):
            raise ValueError(error_message)
        return value  # type: ignore[return-value]

    @staticmethod
    def _get_list(
        relationships: dict[NodeRelationship, RelatedNodeType],
        relationship: NodeRelationship,
        error_message: str,
    ) -> list[NodeReference] | None:
        value = relationships.get(relationship)
        if value is not None and not isinstance(value, list):
            raise ValueError(error_message)
        return value  # type: ignore[return-value]

    def as_dict(self) -> dict[NodeRelationship, RelatedNodeType | None]:
        return {
            NodeRelationship.SOURCE: self.source,
            NodeRelationship.PREVIOUS: self.previous,
            NodeRelationship.NEXT: self.next,
            NodeRelationship.PARENT: self.parent,
            NodeRelationship.CHILD: self.children,
        }

    def to_relationships(self) -> dict[NodeRelationship, RelatedNodeType]:
        filtered = {key: value for key, value in self.as_dict().items() if value is not None}
        return filtered

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
        alias="extra_info",
    )
    excluded_embed_metadata_keys: list[str] = Field(
        default_factory=list,
        description="Metadata keys that are excluded from text for the embed model.",
    )
    excluded_llm_metadata_keys: list[str] = Field(
        default_factory=list,
        description="Metadata keys that are excluded from text for the LLM.",
    )
    relationships: dict[
        Annotated[NodeRelationship, EnumNameSerializer],
        RelatedNodeType,
    ] = Field(
        default_factory=dict,
        description="A mapping of relationships to other node information.",
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
        alias="metadata_seperator",
    )

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
    def linked_nodes(self) -> LinkedNodes:
        return LinkedNodes.from_relationships(self.relationships)

    @property
    def ref_doc_id(self) -> str | None:  # pragma: no cover
        """Deprecated: Get ref doc id."""
        return self.linked_nodes.source_id

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

    def as_related_node_info(self) -> NodeReference:
        """Get node as NodeReference."""
        return NodeReference(
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
