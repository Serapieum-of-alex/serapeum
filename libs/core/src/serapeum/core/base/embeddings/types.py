from typing import Any, Annotated, Sequence
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


class ObjectType(str, Enum):
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
    type: Annotated[ObjectType, EnumNameSerializer] | str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    hash: str | None = None

    @classmethod
    def class_name(cls) -> str:
        return "NodeReference"


RelatedNodeType = NodeReference | list[NodeReference]


class LinkedNodes(SerializableModel):
    model_config = ConfigDict(frozen=True)

    source: NodeReference | None = None
    previous: NodeReference | None = None
    next: NodeReference | None = None
    parent: NodeReference | None = None
    children: list[NodeReference] | None = None

    def as_dict(self) -> dict["NodeRelationship", RelatedNodeType | None]:
        return {
            NodeRelationship.SOURCE: self.source,
            NodeRelationship.PREVIOUS: self.previous,
            NodeRelationship.NEXT: self.next,
            NodeRelationship.PARENT: self.parent,
            NodeRelationship.CHILD: self.children,
        }


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


class BaseNode(SerializableModel, ABC):
    """Base node Object.

    Attributes:
        metadata fields
            - injected as part of the text shown to LLMs as context
            - injected as part of the text for generating embeddings
            - used by vector DBs for metadata filtering

    """
    # hash is computed on local field, during the validation process
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

        usable_metadata_keys = set(self.metadata.keys())
        if mode == MetadataMode.LLM:
            for key in self.excluded_llm_metadata_keys:
                if key in usable_metadata_keys:
                    usable_metadata_keys.remove(key)
        elif mode == MetadataMode.EMBED:
            for key in self.excluded_embed_metadata_keys:
                if key in usable_metadata_keys:
                    usable_metadata_keys.remove(key)

        return self.metadata_separator.join(
            [
                self.metadata_template.format(key=key, value=str(value))
                for key, value in self.metadata.items()
                if key in usable_metadata_keys
            ]
        )

    @abstractmethod
    def set_content(self, value: Any) -> None:
        """Set the content of the node."""

    @property
    @abstractmethod
    def hash(self) -> str:
        """Get hash of node."""

    def _get_relationship(
        self,
        relationship: NodeRelationship,
        expected_type: type,
        *,
        error_message: str,
    ) -> RelatedNodeType | None:
        relation = self.relationships.get(relationship)
        if relation is not None:
            if not isinstance(relation, expected_type):
                raise ValueError(error_message)
        return relation

    def _get_linked(self) -> LinkedNodes:
        return LinkedNodes(
            source=self._get_relationship(
                NodeRelationship.SOURCE,
                NodeReference,
                error_message="Source object must be a single NodeReference object",
            ),
            previous=self._get_relationship(
                NodeRelationship.PREVIOUS,
                NodeReference,
                error_message="Previous object must be a single NodeReference object",
            ),
            next=self._get_relationship(
                NodeRelationship.NEXT,
                NodeReference,
                error_message="Next object must be a single NodeReference object",
            ),
            parent=self._get_relationship(
                NodeRelationship.PARENT,
                NodeReference,
                error_message="Parent object must be a single NodeReference object",
            ),
            children=self._get_relationship(
                NodeRelationship.CHILD,
                list,
                error_message="Child objects must be a list of NodeReference objects.",
            ),
        )

    @property
    def linked_nodes(self) -> LinkedNodes:
        return self._get_linked()

    @property
    def ref_doc_id(self) -> str | None:  # pragma: no cover
        """Deprecated: Get ref doc id."""
        source_node = self.linked_nodes.source
        if source_node is None:
            return None
        return source_node.id

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
