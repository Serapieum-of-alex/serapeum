"""Base types and data models for document nodes and embeddings.

This module provides the foundational types for representing documents and their
relationships in the Serapeum framework. It includes enumerations for node types,
data models for node metadata and relationships, and abstract base classes for
document/node handling.

The node system supports:
- Document chunking and hierarchical relationships
- Metadata management with selective inclusion for LLM vs embedding contexts
- Automatic hash computation and change detection
- Efficient caching of linked node relationships
- Serialization and deserialization via Pydantic

Key Components:
    - NodeContentType: Enum for classifying node content (text, image, etc.)
    - NodeType: Enum for relationship types (source, parent, child, etc.)
    - MetadataMode: Enum for controlling metadata visibility
    - NodeInfo: Lightweight reference to a node with metadata
    - LinkedNodes: Immutable container for node relationships
    - BaseNode: Abstract base class for all document nodes
    - CallMixin: Interface for node transformation components

Examples:
    - Creating a simple node reference
        ```python
        >>> from serapeum.core.base.embeddings.types import NodeInfo, NodeContentType
        >>> node_ref = NodeInfo(
        ...     id="doc-123",
        ...     type=NodeContentType.TEXT,
        ...     metadata={"author": "Alice"}
        ... )
        >>> node_ref.id
        'doc-123'

        ```
    - Building node relationships
        ```python
        >>> from serapeum.core.base.embeddings.types import LinkedNodes, NodeType
        >>> parent = NodeInfo(id="parent-1", type=NodeContentType.DOCUMENT)
        >>> child1 = NodeInfo(id="child-1", type=NodeContentType.TEXT)
        >>> links = LinkedNodes.create({
        ...     NodeType.PARENT: parent,
        ...     NodeType.CHILD: [child1]
        ... })
        >>> links.parent.id
        'parent-1'

        ```

See Also:
    serapeum.core.types.base.SerializableModel: Base class for serialization
    serapeum.core.utils.base.truncate_text: Text truncation utility
"""

from __future__ import annotations
from typing import Any, Annotated, Sequence
import uuid
from abc import abstractmethod, ABC
import textwrap
from enum import Enum
from pydantic import (
    ConfigDict,
    Field,
    PlainSerializer,
    model_validator,
    field_validator,
)
from serapeum.core.utils.base import truncate_text
from serapeum.core.types import SerializableModel

TRUNCATE_LENGTH = 350
"""Maximum length for truncating node text in string representations."""

WRAP_WIDTH = 70
"""Character width for wrapping node text in string representations."""

DEFAULT_METADATA_TMPL = "{key}: {value}"
"""Default template for formatting metadata key-value pairs."""


class NodeContentType(str, Enum):
    """Enumeration of content types that can be stored in a node.

    This enum classifies the type of content a node contains, which helps
    downstream components (LLMs, embeddings, parsers) handle the content
    appropriately. String-based enum values enable direct serialization.

    Attributes:
        TEXT: Plain text content, the most common node type.
        IMAGE: Image data or references to images.
        INDEX: Index structures or metadata about other nodes.
        DOCUMENT: Complete document content before chunking.
        MULTIMODAL: Content combining multiple modalities (text + images).

    Examples:
        - Checking content type
            ```python
            >>> from serapeum.core.base.embeddings.types import NodeContentType
            >>> content_type = NodeContentType.TEXT
            >>> content_type.value
            'text'

            ```
        - Using in node metadata
            ```python
            >>> from serapeum.core.base.embeddings.types import NodeInfo
            >>> node = NodeInfo(id="node-1", type=NodeContentType.IMAGE)
            >>> node.type
            <NodeContentType.IMAGE: 'image'>

            ```
        - String comparison
            ```python
            >>> NodeContentType.TEXT == "text"
            True

            ```

    See Also:
        NodeInfo: Uses this enum to specify node content type.
        BaseNode.get_type: Abstract method returning content type string.
    """

    TEXT = "text"
    IMAGE = "image"
    INDEX = "index"
    DOCUMENT = "document"
    MULTIMODAL = "multimodal"


# Pydantic serializer that converts enum instances to their string values
EnumNameSerializer = PlainSerializer(
    lambda e: e.value, return_type="str", when_used="always"
)


class NodeInfo(SerializableModel):
    """Lightweight reference to a node with essential identification metadata.

    NodeInfo provides a compact representation of a node without its full content,
    useful for creating references and relationships between nodes. It includes
    the node's ID, content type, metadata, and optional hash for change detection.

    Attributes:
        id: Unique identifier for the node.
        type: Content type classification (NodeContentType enum or string).
        metadata: Arbitrary metadata dictionary for the node.
        hash: Optional hash value for detecting content changes.

    Examples:
        - Creating a basic node reference
            ```python
            >>> from serapeum.core.base.embeddings.types import NodeInfo, NodeContentType
            >>> ref = NodeInfo(
            ...     id="doc-456",
            ...     type=NodeContentType.TEXT,
            ...     metadata={"page": 1}
            ... )
            >>> ref.id
            'doc-456'

            ```
        - Serialization and deserialization
            ```python
            >>> ref = NodeInfo(id="node-1", type=NodeContentType.DOCUMENT)
            >>> json_str = ref.to_json()
            >>> restored = NodeInfo.from_json(json_str)
            >>> restored.id
            'node-1'

            ```
        - Using with hash for change detection
            ```python
            >>> import hashlib
            >>> content = "Sample text"
            >>> content_hash = hashlib.sha256(content.encode()).hexdigest()
            >>> ref = NodeInfo(id="node-2", hash=content_hash)
            >>> ref.hash[:8]  # First 8 chars of hash  # doctest: +SKIP
            'e3b0c442'

            ```

    See Also:
        BaseNode: Full node implementation that generates NodeInfo.
        LinkedNodes: Container for node relationships using NodeInfo.
        SerializableModel: Base class providing serialization methods.
    """

    id: str
    type: Annotated[NodeContentType, EnumNameSerializer] | str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    hash: str | None = None

    @classmethod
    def class_name(cls) -> str:
        """Return the class name identifier for serialization.

        Returns:
            Always returns "NodeInfo" as the stable class identifier.

        Examples:
            - Getting class name
                ```python
                >>> from serapeum.core.base.embeddings.types import NodeInfo
                >>> NodeInfo.class_name()
                'NodeInfo'

                ```
        """
        return "NodeInfo"


NodeInfoType = NodeInfo | list[NodeInfo]


class MetadataMode(str, Enum):
    """Enumeration for controlling which metadata is included in different contexts.

    Different use cases require different metadata visibility. For example, you
    might exclude certain metadata from embeddings (to avoid semantic pollution)
    while including it for LLM context (to provide additional information).

    Attributes:
        ALL: Include all metadata fields.
        EMBED: Include only metadata for embedding generation (excludes fields
            in excluded_embed_metadata_keys).
        LLM: Include only metadata for LLM context (excludes fields in
            excluded_llm_metadata_keys).
        NONE: Exclude all metadata.

    Examples:
        - Filtering metadata for embeddings
            ```python
            >>> from serapeum.core.base.embeddings.types import MetadataMode
            >>> mode = MetadataMode.EMBED
            >>> mode.value
            'embed'

            ```
        - Using with node content retrieval (conceptual)
            ```python
            >>> MetadataMode.LLM == "llm"
            True
            >>> MetadataMode.NONE == "none"
            True

            ```
        - Checking mode type
            ```python
            >>> isinstance(MetadataMode.ALL, str)
            True

            ```

    See Also:
        BaseNode.get_content: Uses this mode to control metadata inclusion.
        BaseNode.get_metadata_str: Filters metadata based on this mode.
        BaseNode.excluded_embed_metadata_keys: Metadata excluded for EMBED mode.
        BaseNode.excluded_llm_metadata_keys: Metadata excluded for LLM mode.
    """

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
    """Immutable container for node relationships in a document hierarchy.

    LinkedNodes manages references between nodes in a document structure, supporting
    linear sequences (previous/next), hierarchical relationships (parent/children),
    and source document tracking. The model is frozen to prevent accidental mutation
    of relationship structures.

    Attributes:
        source: Reference to the original source document node.
        previous: Reference to the previous node in a sequence.
        next: Reference to the next node in a sequence.
        parent: Reference to the parent node in a hierarchy.
        children: List of child node references in a hierarchy.

    Examples:
        - Creating a linear sequence of nodes
            ```python
            >>> from serapeum.core.base.embeddings.types import LinkedNodes, NodeInfo, NodeType
            >>> prev_node = NodeInfo(id="chunk-1")
            >>> next_node = NodeInfo(id="chunk-3")
            >>> links = LinkedNodes(previous=prev_node, next=next_node)
            >>> links.previous.id
            'chunk-1'

            ```
        - Building hierarchical relationships
            ```python
            >>> parent = NodeInfo(id="section-1")
            >>> child1 = NodeInfo(id="para-1")
            >>> child2 = NodeInfo(id="para-2")
            >>> links = LinkedNodes(parent=parent, children=[child1, child2])
            >>> len(links.children)
            2

            ```
        - Using factory method with NodeType enum
            ```python
            >>> from serapeum.core.base.embeddings.types import NodeType
            >>> source = NodeInfo(id="doc-main")
            >>> links_dict = {NodeType.SOURCE: source}
            >>> links = LinkedNodes.create(links_dict)
            >>> links.source.id
            'doc-main'

            ```
        - Accessing source ID property
            ```python
            >>> source = NodeInfo(id="original-doc")
            >>> links = LinkedNodes(source=source)
            >>> links.source_id
            'original-doc'

            ```

    See Also:
        NodeType: Enum defining relationship types.
        NodeInfo: References stored in relationship fields.
        BaseNode.linked_nodes: Property that creates LinkedNodes from links dict.
    """

    model_config = ConfigDict(frozen=True)

    source: NodeInfo | None = None
    previous: NodeInfo | None = None
    next: NodeInfo | None = None
    parent: NodeInfo | None = None
    children: list[NodeInfo] | None = None

    @field_validator("source", "previous", "next", "parent")
    @classmethod
    def validate_single_node(cls, v: Any) -> NodeInfo | None:
        """Validate that single-node fields contain NodeInfo objects.

        Ensures that source, previous, next, and parent fields contain exactly
        one NodeInfo instance (not a list). Called automatically by Pydantic
        during model instantiation and validation.

        Args:
            v: Value to validate, expected to be NodeInfo or None.

        Returns:
            The validated NodeInfo instance or None.

        Raises:
            ValueError: If v is not None and not a NodeInfo instance.

        Examples:
            - Valid single node assignment
                ```python
                >>> from serapeum.core.base.embeddings.types import LinkedNodes, NodeInfo
                >>> node = NodeInfo(id="valid")
                >>> links = LinkedNodes(source=node)
                >>> links.source.id
                'valid'

                ```
            - Invalid list assignment to single-node field
                ```python
                >>> LinkedNodes(source=[NodeInfo(id="bad")])  # doctest: +SKIP
                Traceback (most recent call last):
                    ...
                ValueError: Must be a NodeInfo object, not a list

                ```

        Note:
            This validator applies to: source, previous, next, parent fields.
            The children field has a separate validator for list validation.
        """
        if v is not None and not isinstance(v, NodeInfo):
            raise ValueError("Must be a NodeInfo object, not a list")
        return v

    @field_validator("children")
    @classmethod
    def validate_children_list(cls, v: Any) -> list[NodeInfo] | None:
        """Validate that children field contains a list of NodeInfo objects.

        Ensures the children field is a list (not a single NodeInfo instance).
        Called automatically by Pydantic during model instantiation and validation.

        Args:
            v: Value to validate, expected to be list[NodeInfo] or None.

        Returns:
            The validated list of NodeInfo instances or None.

        Raises:
            ValueError: If v is not None and not a list.

        Examples:
            - Valid children list
                ```python
                >>> from serapeum.core.base.embeddings.types import LinkedNodes, NodeInfo
                >>> child1 = NodeInfo(id="child-1")
                >>> child2 = NodeInfo(id="child-2")
                >>> links = LinkedNodes(children=[child1, child2])
                >>> len(links.children)
                2

                ```
            - Invalid single NodeInfo for children
                ```python
                >>> LinkedNodes(children=NodeInfo(id="bad"))  # doctest: +SKIP
                Traceback (most recent call last):
                    ...
                ValueError: Children must be a list of NodeInfo objects

                ```
            - Empty children list is valid
                ```python
                >>> links = LinkedNodes(children=[])
                >>> links.children
                []

                ```

        Note:
            This validator is specific to the children field, which represents
            one-to-many relationships.
        """
        if v is not None and not isinstance(v, list):
            raise ValueError("Children must be a list of NodeInfo objects")
        return v

    @classmethod
    def create(cls, linked_nodes_info: dict[NodeType, NodeInfoType]) -> "LinkedNodes":
        """Create LinkedNodes from a dict mapping NodeType to NodeInfo/list.

        Factory method that converts a dictionary with NodeType keys into a
        validated LinkedNodes instance. Pydantic validators automatically check
        that single-node fields contain NodeInfo and children contains a list.

        Args:
            linked_nodes_info: Dictionary mapping NodeType enum values to either
                NodeInfo (for single relationships) or list[NodeInfo] (for
                children). Missing keys are treated as None.

        Returns:
            A new LinkedNodes instance with validated relationships.

        Raises:
            ValueError: If a single-node field (SOURCE, PREVIOUS, NEXT, PARENT)
                receives a list, or if children receives a non-list value.

        Examples:
            - Creating from a dict with mixed relationships
                ```python
                >>> from serapeum.core.base.embeddings.types import LinkedNodes, NodeInfo, NodeType
                >>> source = NodeInfo(id="doc-1")
                >>> parent = NodeInfo(id="section-1")
                >>> children = [NodeInfo(id="para-1"), NodeInfo(id="para-2")]
                >>> links_dict = {
                ...     NodeType.SOURCE: source,
                ...     NodeType.PARENT: parent,
                ...     NodeType.CHILD: children
                ... }
                >>> links = LinkedNodes.create(links_dict)
                >>> links.source.id
                'doc-1'

                ```
            - Creating with only some relationships
                ```python
                >>> prev = NodeInfo(id="chunk-1")
                >>> next_node = NodeInfo(id="chunk-3")
                >>> links = LinkedNodes.create({
                ...     NodeType.PREVIOUS: prev,
                ...     NodeType.NEXT: next_node
                ... })
                >>> links.previous.id
                'chunk-1'

                ```
            - Empty dict creates all-None instance
                ```python
                >>> links = LinkedNodes.create({})
                >>> links.source is None
                True

                ```

        See Also:
            LinkedNodes.as_dict: Inverse operation converting LinkedNodes to dict.
            NodeType: Enum defining valid relationship types.
        """
        return cls(
            source=linked_nodes_info.get(NodeType.SOURCE),
            previous=linked_nodes_info.get(NodeType.PREVIOUS),
            next=linked_nodes_info.get(NodeType.NEXT),
            parent=linked_nodes_info.get(NodeType.PARENT),
            children=linked_nodes_info.get(NodeType.CHILD),
        )

    def as_dict(self) -> dict[NodeType, NodeInfoType | None]:
        """Convert LinkedNodes to a dictionary mapping NodeType to NodeInfo.

        Creates a dictionary representation with NodeType enum keys and NodeInfo
        values. None values are excluded from the result to create a compact
        representation containing only active relationships.

        Returns:
            Dictionary with NodeType keys and NodeInfo/list[NodeInfo] values.
            Only non-None relationships are included.

        Examples:
            - Converting to dict with multiple relationships
                ```python
                >>> from serapeum.core.base.embeddings.types import LinkedNodes, NodeInfo, NodeType
                >>> source = NodeInfo(id="doc-1")
                >>> parent = NodeInfo(id="section-1")
                >>> links = LinkedNodes(source=source, parent=parent)
                >>> result = links.as_dict()
                >>> result[NodeType.SOURCE].id
                'doc-1'

                ```
            - None values are excluded
                ```python
                >>> links = LinkedNodes(source=NodeInfo(id="doc-1"))
                >>> result = links.as_dict()
                >>> NodeType.PREVIOUS in result
                False

                ```
            - Round-trip with create method
                ```python
                >>> original = LinkedNodes(
                ...     source=NodeInfo(id="src"),
                ...     children=[NodeInfo(id="child-1")]
                ... )
                >>> as_dict = original.as_dict()
                >>> restored = LinkedNodes.create(as_dict)
                >>> restored.source.id
                'src'

                ```

        See Also:
            LinkedNodes.create: Factory method for creating from dict.
            BaseNode.links: Uses this format for storing relationships.
        """
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
        """Get the ID of the source node if it exists.

        Convenience property for accessing the source node's ID without
        checking if source is None first.

        Returns:
            The source node's ID string, or None if no source is set.

        Examples:
            - Accessing source ID when source exists
                ```python
                >>> from serapeum.core.base.embeddings.types import LinkedNodes, NodeInfo
                >>> source = NodeInfo(id="document-123")
                >>> links = LinkedNodes(source=source)
                >>> links.source_id
                'document-123'

                ```
            - Accessing when source is None
                ```python
                >>> links = LinkedNodes()
                >>> links.source_id is None
                True

                ```

        See Also:
            BaseNode.source_id: Uses this property for node source tracking.
        """
        source_id = None
        if self.source is not None:
            source_id = self.source.id
        return source_id


class BaseNode(SerializableModel, ABC):
    r"""Abstract base class for document nodes with metadata and relationship management.

    BaseNode provides the foundational functionality for representing chunks of
    documents with rich metadata, embeddings, and hierarchical relationships. It
    supports selective metadata inclusion for different contexts (LLM vs embeddings),
    automatic change detection via hashing, and efficient relationship caching.

    Key features:
    - Automatic UUID generation for node identification
    - Metadata management with selective inclusion/exclusion for LLM and embedding contexts
    - Relationship tracking (source, parent, children, previous, next)
    - Embedding storage and retrieval
    - Cached LinkedNodes computation with automatic invalidation
    - Customizable metadata formatting and serialization

    Attributes:
        id: Unique identifier for the node (auto-generated UUID if not provided).
        embedding: Optional vector embedding for the node's content.
        metadata: Flat dictionary of metadata fields used for context and filtering.
        excluded_embed_metadata_keys: Metadata keys excluded from embedding context.
        excluded_llm_metadata_keys: Metadata keys excluded from LLM context.
        links: Dictionary mapping NodeType to NodeInfo for relationships.
        metadata_template: Template string for formatting metadata (default: "{key}: {value}").
        metadata_separator: Separator between metadata fields (default: newline).

    Note:
        This is an abstract base class. Subclasses must implement:
        - get_type(): Return the node's content type identifier
        - get_content(): Return the node's content with optional metadata
        - set_content(): Update the node's content
        - hash: Property returning the content hash for change detection

    Examples:
        - Creating a concrete node subclass
            ```python
            >>> from serapeum.core.base.embeddings.types import BaseNode, MetadataMode, NodeType, NodeInfo
            >>> import hashlib
            >>> from pydantic import Field
            >>>
            >>> class TextNode(BaseNode):
            ...     text: str = Field(default="", description="Text content of the node")
            ...
            ...     @classmethod
            ...     def get_type(cls) -> str:
            ...         return "text"
            ...
            ...     def get_content(self, metadata_mode: MetadataMode = MetadataMode.ALL) -> str:
            ...         metadata_str = self.get_metadata_str(mode=metadata_mode)
            ...         return f"{metadata_str}\\n{self.text}" if metadata_str else self.text
            ...
            ...     def set_content(self, value: str) -> None:
            ...         self.text = value
            ...
            ...     @property
            ...     def hash(self) -> str:
            ...         return hashlib.sha256(self.text.encode()).hexdigest()

            ```
            - Create a node with metadata
            ```python
            >>> node = TextNode(
            ...     text="Hello world",
            ...     metadata={"page": 1, "author": "Alice"}
            ... )
            >>> node.get_type()
            'text'
            >>> node.get_content(metadata_mode=MetadataMode.NONE)
            'Hello world'

            ```
        - Using metadata exclusion for different contexts
            ```python
            >>> node = TextNode(
            ...     text="Sensitive content",
            ...     metadata={"public": "yes", "internal_id": "secret123"},
            ...     excluded_llm_metadata_keys=["internal_id"]
            ... )

            ```
            - For LLM context (excludes internal_id)
            ```python
            >>> content_for_llm = node.get_content(metadata_mode=MetadataMode.LLM)
            >>> "internal_id" in content_for_llm
            False
            >>> "public" in content_for_llm
            True

            ```
        - Setting up node relationships
            ```python
            >>> parent = NodeInfo(id="parent-doc", type="document")
            >>> child = NodeInfo(id="child-chunk", type="text")
            >>>
            >>> node = TextNode(
            ...     text="Child content",
            ...     links={NodeType.PARENT: parent, NodeType.SOURCE: parent}
            ... )
            >>> node.linked_nodes.parent.id
            'parent-doc'
            >>> node.source_id
            'parent-doc'

            ```
        - Working with embeddings
            ```python
            >>> node = TextNode(text="Sample text")
            >>> node.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
            >>> embedding_vec = node.get_embedding()
            >>> len(embedding_vec)
            5
            >>> embedding_vec[0]
            0.1

            ```

    See Also:
        NodeInfo: Lightweight reference to a node.
        LinkedNodes: Container for node relationships.
        MetadataMode: Controls metadata inclusion in different contexts.
        SerializableModel: Base class providing serialization capabilities.
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

    @model_validator(mode="after")
    def _invalidate_linked_nodes_cache_on_links_change(self) -> "BaseNode":
        """Invalidate the linked_nodes cache when links dict is reassigned.

        This validator tracks the id of the links dict. When it changes
        (i.e., links is reassigned), the cache is cleared.

        Uses Pydantic v2's @model_validator with object.__setattr__ to avoid recursion.
        """
        current_links_id = id(self.links)

        # Check if links dict was reassigned (different id)
        if self.links_dict_id is None or self.links_dict_id != current_links_id:
            # Links changed, clear cache and update tracked id
            object.__setattr__(self, "linked_nodes_cache", None)
            object.__setattr__(self, "links_dict_id", current_links_id)

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
            else (
                (key, value)
                for key, value in self.metadata.items()
                if key not in excluded
            )
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

        Examples:
            >>> from serapeum.core.base.embeddings.types import BaseNode, NodeInfo, NodeType, MetadataMode
            >>> import hashlib
            >>> from pydantic import Field
            >>> class TextNode(BaseNode):
            ...     text: str = Field(default="")
            ...     @classmethod
            ...     def get_type(cls) -> str:
            ...         return "text"
            ...     def get_content(self, metadata_mode=MetadataMode.ALL) -> str:
            ...         return self.text
            ...     def set_content(self, value: str) -> None:
            ...         self.text = value
            ...     @property
            ...     def hash(self) -> str:
            ...         return hashlib.sha256(self.text.encode()).hexdigest()
            >>> node = TextNode(text="Sample", links={})
            >>> new_source = NodeInfo(id="updated-source", type="document")
            >>> node.links[NodeType.SOURCE] = new_source
            >>> node._clear_linked_nodes_cache()
            >>> node.linked_nodes.source.id
            'updated-source'
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

        Examples:
            >>> from serapeum.core.base.embeddings.types import BaseNode, NodeInfo, NodeType, MetadataMode
            >>> import hashlib
            >>> from pydantic import Field
            >>> class TextNode(BaseNode):
            ...     text: str = Field(default="")
            ...     @classmethod
            ...     def get_type(cls) -> str:
            ...         return "text"
            ...     def get_content(self, metadata_mode=MetadataMode.ALL) -> str:
            ...         return self.text
            ...     def set_content(self, value: str) -> None:
            ...         self.text = value
            ...     @property
            ...     def hash(self) -> str:
            ...         return hashlib.sha256(self.text.encode()).hexdigest()
            >>> node = TextNode(text="Sample")
            >>> source_ref = NodeInfo(id="doc-123", type="document")
            >>> node.links = {NodeType.SOURCE: source_ref}
            >>> node.linked_nodes.source.id
            'doc-123'

            >>> node = TextNode(text="Sample", links={})
            >>> prev_ref = NodeInfo(id="prev-chunk", type="text")
            >>> node.links[NodeType.PREVIOUS] = prev_ref
            >>> node._clear_linked_nodes_cache()
            >>> node.linked_nodes.previous.id
            'prev-chunk'

            >>> node = TextNode(text="Sample")
            >>> parent = NodeInfo(id="parent-1", type="document")
            >>> child1 = NodeInfo(id="child-1", type="text")
            >>> child2 = NodeInfo(id="child-2", type="text")
            >>> node.links = {NodeType.PARENT: parent, NodeType.CHILD: [child1, child2]}
            >>> node.linked_nodes.parent.id
            'parent-1'
            >>> len(node.linked_nodes.children)
            2
        """
        if self.linked_nodes_cache is None:
            # Compute and cache the LinkedNodes
            self.linked_nodes_cache = LinkedNodes.create(self.links)
        return self.linked_nodes_cache

    def __str__(self) -> str:
        """STR."""
        source_text_truncated = truncate_text(
            self.get_content().strip(), TRUNCATE_LENGTH
        )
        source_text_wrapped = textwrap.fill(
            f"Text: {source_text_truncated}\n", width=WRAP_WIDTH
        )
        return f"Node ID: {self.id}\n{source_text_wrapped}"

    def get_embedding(self) -> list[float]:
        """Get embedding.

        Raises:
            ValueErrors if embedding is None.
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
    """Base class for node transformation components.

    CallMixin defines the interface for components that transform sequences of nodes,
    such as embedders, parsers, or metadata enrichers. It provides both synchronous
    and asynchronous calling interfaces.

    The mixin uses callable syntax (`obj(nodes)`) for synchronous transforms and
    `obj.acall(nodes)` for asynchronous transforms, enabling composable pipelines.

    Attributes:
        model_config: Pydantic configuration allowing arbitrary types in subclasses.

    Examples:
        >>> from serapeum.core.base.embeddings.types import CallMixin, BaseNode, MetadataMode
        >>> from typing import Sequence, Any
        >>> import hashlib
        >>> from pydantic import Field
        >>> class TextNode(BaseNode):
        ...     text: str = Field(default="")
        ...     @classmethod
        ...     def get_type(cls) -> str:
        ...         return "text"
        ...     def get_content(self, metadata_mode=MetadataMode.ALL) -> str:
        ...         return self.text
        ...     def set_content(self, value: str) -> None:
        ...         self.text = value
        ...     @property
        ...     def hash(self) -> str:
        ...         return hashlib.sha256(self.text.encode()).hexdigest()
        >>> class UppercaseTransform(CallMixin):
        ...     def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        ...         result = []
        ...         for node in nodes:
        ...             node.set_content(node.get_content().upper())
        ...             result.append(node)
        ...         return result
        >>> transformer = UppercaseTransform()
        >>> nodes = [TextNode(text="hello"), TextNode(text="world")]
        >>> transformed = transformer(nodes)
        >>> transformed[0].get_content()
        'HELLO'
        >>> transformed[1].get_content()
        'WORLD'

    See Also:
        BaseEmbedding: Uses CallMixin to enable embedding nodes.
        BaseNode: The node type that this mixin transforms.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        """Transform a sequence of nodes synchronously.

        Subclasses must implement this method to define their transformation logic.
        This method is called when the object is invoked directly: `obj(nodes)`.

        Args:
            nodes: Sequence of BaseNode instances to transform.
            **kwargs: Additional keyword arguments specific to the transformation.

        Returns:
            Transformed sequence of BaseNode instances.

        Examples:
            >>> from serapeum.core.base.embeddings.types import CallMixin, BaseNode, MetadataMode
            >>> import hashlib
            >>> from pydantic import Field
            >>> class TextNode(BaseNode):
            ...     text: str = Field(default="")
            ...     @classmethod
            ...     def get_type(cls) -> str:
            ...         return "text"
            ...     def get_content(self, metadata_mode=MetadataMode.ALL) -> str:
            ...         return self.text
            ...     def set_content(self, value: str) -> None:
            ...         self.text = value
            ...     @property
            ...     def hash(self) -> str:
            ...         return hashlib.sha256(self.text.encode()).hexdigest()
            >>> class MetadataAdder(CallMixin):
            ...     def __call__(self, nodes, **kwargs):
            ...         result = []
            ...         for i, node in enumerate(nodes):
            ...             node.metadata["index"] = i
            ...             result.append(node)
            ...         return result
            >>> adder = MetadataAdder()
            >>> nodes = [TextNode(text="first"), TextNode(text="second")]
            >>> processed = adder(nodes)
            >>> processed[0].metadata["index"]
            0
            >>> processed[1].metadata["index"]
            1

        Note:
            Implementations should preserve node identity where possible and
            avoid mutating input nodes unless explicitly documented.
        """

    async def acall(
        self, nodes: Sequence[BaseNode], **kwargs: Any
    ) -> Sequence[BaseNode]:
        """Transform a sequence of nodes asynchronously.

        Default implementation delegates to synchronous `__call__`. Subclasses
        can override this for true async implementations (e.g., async API calls).

        Args:
            nodes: Sequence of BaseNode instances to transform.
            **kwargs: Additional keyword arguments specific to the transformation.

        Returns:
            Transformed sequence of BaseNode instances.

        Examples:
            >>> import asyncio
            >>> from serapeum.core.base.embeddings.types import CallMixin, BaseNode, MetadataMode
            >>> import hashlib
            >>> from pydantic import Field
            >>> class TextNode(BaseNode):
            ...     text: str = Field(default="")
            ...     @classmethod
            ...     def get_type(cls) -> str:
            ...         return "text"
            ...     def get_content(self, metadata_mode=MetadataMode.ALL) -> str:
            ...         return self.text
            ...     def set_content(self, value: str) -> None:
            ...         self.text = value
            ...     @property
            ...     def hash(self) -> str:
            ...         return hashlib.sha256(self.text.encode()).hexdigest()
            >>> class AsyncTransform(CallMixin):
            ...     def __call__(self, nodes, **kwargs):
            ...         return nodes
            ...     async def acall(self, nodes, **kwargs):
            ...         await asyncio.sleep(0)
            ...         for node in nodes:
            ...             node.metadata["async_processed"] = True
            ...         return nodes
            >>> transform = AsyncTransform()
            >>> nodes = [TextNode(text="test")]
            >>> result = asyncio.run(transform.acall(nodes))
            >>> result[0].metadata["async_processed"]
            True

            >>> class SyncOnlyTransform(CallMixin):
            ...     def __call__(self, nodes, **kwargs):
            ...         for node in nodes:
            ...             node.metadata["processed"] = True
            ...         return nodes
            >>> sync_transform = SyncOnlyTransform()
            >>> nodes = [TextNode(text="test")]
            >>> result = asyncio.run(sync_transform.acall(nodes))
            >>> result[0].metadata["processed"]
            True

        Note:
            If no true async implementation is needed, the default delegation
            to `__call__` is sufficient. Override only if the transformation
            benefits from async/await (e.g., I/O operations).
        """
        return self.__call__(nodes, **kwargs)
