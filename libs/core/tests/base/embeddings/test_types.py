import asyncio

import pytest
from pydantic import ValidationError

from serapeum.core.base.embeddings.types import (
    BaseNode,
    CallMixin,
    LinkedNodes,
    MetadataMode,
    NodeInfo,
    NodeType,
    NodeContentType,
)


@pytest.fixture()
def my_node_class():
    class MyNode(BaseNode):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)

        @classmethod
        def get_type(cls) -> str:
            return NodeContentType.TEXT

        def get_content(self, metadata_mode: MetadataMode = MetadataMode.ALL) -> str:
            return "Test content"

        def set_content(self, value) -> None:
            return super().set_content(value)

        @property
        def hash(self) -> str:
            return super().hash

    return MyNode


class TestNodeReferenceClassName:
    def test_returns_class_name(self):
        """
        Inputs: none.
        Expected result: class_name returns the literal "NodeInfo".
        Checks: exact string value.
        """
        value = NodeInfo.class_name()
        assert value == "NodeInfo"


class TestNodeTypeEnum:
    @pytest.mark.parametrize("member", list(NodeContentType))
    def test_values_are_strings(self, member):
        """
        Inputs: each NodeContentType enum member.
        Expected result: enum values are strings.
        Checks: value type is str for all members.
        """
        value = member.value
        assert isinstance(value, str)


class TestMetadataModeEnum:
    @pytest.mark.parametrize("member", list(MetadataMode))
    def test_values_are_strings(self, member):
        """
        Inputs: each MetadataMode enum member.
        Expected result: enum values are strings.
        Checks: value type is str for all members.
        """
        value = member.value
        assert isinstance(value, str)


class TestNodeRelationshipEnum:
    @pytest.mark.parametrize("member", list(NodeType))
    def test_values_are_strings(self, member):
        """
        Inputs: each NodeType enum member.
        Expected result: enum values are strings.
        Checks: value type is str for all members.
        """
        value = member.value
        assert isinstance(value, str)


class TestLinkedNodesCreate:
    def test_builds_linked_nodes_from_valid_relationships(self):
        """
        Inputs: links dict with valid NodeInfo values.
        Expected result: LinkedNodes fields are populated as provided.
        Checks: mapping to source/previous/next/parent/children.
        """
        ref = NodeInfo(id="a")
        linked_nodes = {
            NodeType.SOURCE: ref,
            NodeType.PREVIOUS: ref,
            NodeType.NEXT: ref,
            NodeType.PARENT: ref,
            NodeType.CHILD: [ref],
        }
        linked = LinkedNodes.create(linked_nodes)
        assert linked.source is ref
        assert linked.previous is ref
        assert linked.next is ref
        assert linked.parent is ref
        assert linked.children == [ref]

    @pytest.mark.parametrize(
        "node_type, invalid_value, expected_field",
        [
            (NodeType.SOURCE, [NodeInfo(id="a")], "source"),
            (NodeType.PREVIOUS, [NodeInfo(id="a")], "previous"),
            (NodeType.NEXT, [NodeInfo(id="a")], "next"),
            (NodeType.PARENT, [NodeInfo(id="a")], "parent"),
            (NodeType.CHILD, NodeInfo(id="a"), "children"),
        ],
    )
    def test_invalid_types_raise_validation_error(
        self, node_type, invalid_value, expected_field
    ):
        """
        Inputs: links with invalid value types for each key.
        Expected result: Pydantic ValidationError.
        Checks: exception type and field name.

        Note: Uses Pydantic's field_validator which raises ValidationError
        instead of ValueError for better error messages and type safety.
        """
        linked_nodes = {node_type: invalid_value}
        with pytest.raises(ValidationError) as exc_info:
            LinkedNodes.create(linked_nodes)

        # Verify the error is for the correct field
        errors = exc_info.value.errors()
        assert any(expected_field in str(err["loc"]) for err in errors)


class TestLinkedNodesAsDict:
    def test_returns_full_mapping(self):
        """
        Inputs: LinkedNodes with mixed values.
        Expected result: dict contains all relationship keys with values.
        Checks: key presence and value mapping.
        """
        ref = NodeInfo(id="a")
        linked = LinkedNodes(source=ref, children=[ref])
        mapping = linked.as_dict()
        assert mapping[NodeType.SOURCE] is ref
        assert mapping[NodeType.CHILD] == [ref]

    def test_exclude_none_filters(self):
        """
        Inputs: LinkedNodes with only source populated and exclude_none True.
        Expected result: mapping contains only non-None values.
        Checks: only SOURCE key remains.
        """
        ref = NodeInfo(id="a")
        linked = LinkedNodes(source=ref)
        mapping = linked.as_dict()
        assert mapping == {NodeType.SOURCE: ref}


class TestLinkedNodesSourceId:
    def test_returns_source_id_or_none(self):
        """
        Inputs: LinkedNodes with source, and without source.
        Expected result: source_id matches id or is None.
        Checks: correct id string and None handling.
        """
        ref = NodeInfo(id="a")
        linked_with = LinkedNodes(source=ref)
        linked_without = LinkedNodes()
        value_with = linked_with.source_id
        value_without = linked_without.source_id
        assert value_with == "a"
        assert value_without is None


class TestBaseNodeGetMetadataStr:
    def test_none_mode_returns_empty(self, my_node_class):
        """
        Inputs: metadata with values, mode=NONE.
        Expected result: empty string.
        Checks: exact empty output.
        """
        node = my_node_class(metadata={"k": "v"})
        value = node.get_metadata_str(MetadataMode.NONE)
        assert value == ""

    def test_all_mode_includes_all(self, my_node_class):
        """
        Inputs: metadata with two keys, mode=ALL.
        Expected result: both entries present in output.
        Checks: substring inclusion for both keys.
        """
        node = my_node_class(metadata={"a": 1, "b": 2})
        value = node.get_metadata_str(MetadataMode.ALL)
        assert "a: 1" in value
        assert "b: 2" in value

    def test_llm_mode_excludes_keys(self, my_node_class):
        """
        Inputs: metadata with excluded key for LLM.
        Expected result: excluded key absent from output.
        Checks: included key present, excluded key missing.
        """
        node = my_node_class(
            metadata={"allowed": "yes", "forbidden": "no"},
            excluded_llm_metadata_keys=["forbidden"],
        )
        value = node.get_metadata_str(MetadataMode.LLM)
        assert "allowed: yes" in value
        assert "forbidden" not in value

    def test_embed_mode_excludes_keys(self, my_node_class):
        """
        Inputs: metadata with excluded key for EMBED.
        Expected result: excluded key absent from output.
        Checks: included key present, excluded key missing.
        """
        node = my_node_class(
            metadata={"allowed": "yes", "forbidden": "no"},
            excluded_embed_metadata_keys=["forbidden"],
        )
        value = node.get_metadata_str(MetadataMode.EMBED)
        assert "allowed: yes" in value
        assert "forbidden" not in value

    def test_empty_metadata_returns_empty(self, my_node_class):
        """
        Inputs: empty metadata dict.
        Expected result: empty string.
        Checks: exact empty output.
        """
        node = my_node_class(metadata={})
        value = node.get_metadata_str(MetadataMode.ALL)
        assert value == ""


class TestBaseNodeLinkedNodes:
    def test_linked_nodes_consistent_when_relationships_unchanged(self, my_node_class):
        """
        Inputs: node with links unchanged between calls.
        Expected result: linked_nodes returns equivalent objects.
        Checks: object equality is stable across calls (not identity).
        Note: Caching removed to avoid stale cache bugs from in-place mutations.
        """
        ref = NodeInfo(id="a")
        node = my_node_class(links={NodeType.SOURCE: ref})
        first = node.linked_nodes
        second = node.linked_nodes
        # Objects are equal but not identical (no caching)
        assert first == second
        assert first.source is ref
        assert second.source is ref

    def test_linked_nodes_updates_when_relationships_replaced(self, my_node_class):
        """
        Inputs: node with links replaced.
        Expected result: linked_nodes reflects new links immediately.
        Checks: updated object includes new source id.
        Note: No caching means changes are always reflected immediately.
        """
        node = my_node_class(links={})
        first = node.linked_nodes
        assert first.source_id is None

        node.links = {NodeType.SOURCE: NodeInfo(id="new")}
        updated = node.linked_nodes
        assert updated.source_id == "new"

    def test_linked_nodes_with_manual_cache_clear(self, my_node_class):
        """
        Test that demonstrates proper cache management for in-place mutations.

        Inputs: node with links mutated in-place.
        Expected result: Manual cache clear required for in-place mutations.
        Checks: Cache is properly cleared and recomputed after manual clear.

        Note: With Pydantic v2 caching implementation, in-place dict mutations
        require manual cache clearing. Use _clear_linked_nodes_cache() or
        reassign the entire dict to trigger automatic cache invalidation.
        """
        ref = NodeInfo(id="original")
        node = my_node_class(links={NodeType.SOURCE: ref})

        # Get linked nodes - should have original source
        first = node.linked_nodes
        assert first.source_id == "original"

        # Mutate the links dict in-place
        node.links[NodeType.PREVIOUS] = NodeInfo(id="prev")

        # Option 1: Manually clear cache after in-place mutation
        node._clear_linked_nodes_cache()

        # Now linked_nodes reflects the mutation
        second = node.linked_nodes
        assert second.source_id == "original"
        assert second.previous is not None
        assert second.previous.id == "prev"

    def test_linked_nodes_auto_invalidation_on_reassignment(self, my_node_class):
        """
        Test automatic cache invalidation on dict reassignment.

        Inputs: node with links dict reassigned.
        Expected result: Cache automatically cleared, no manual action needed.
        Checks: Changes reflected without calling _clear_linked_nodes_cache().

        Note: This is the preferred pattern - reassign the dict rather than
        mutating in-place to get automatic cache invalidation.
        """
        ref = NodeInfo(id="original")
        node = my_node_class(links={NodeType.SOURCE: ref})

        first = node.linked_nodes
        assert first.source_id == "original"

        # Reassign the entire dict (triggers validation and cache clear)
        new_links = dict(node.links)  # Copy existing
        new_links[NodeType.PREVIOUS] = NodeInfo(id="prev")
        node.links = new_links  # Automatic cache invalidation!

        # Cache is auto-cleared, changes reflected immediately
        second = node.linked_nodes
        assert second.source_id == "original"
        assert second.previous is not None
        assert second.previous.id == "prev"


class TestBaseNodeRefDocId:
    def test_uses_source_id(self, my_node_class):
        """
        Inputs: node with and without source relationship.
        Expected result: ref_doc_id equals source id or None.
        Checks: correct id and None handling.
        """
        with_source = my_node_class(links={NodeType.SOURCE: NodeInfo(id="a")})
        without_source = my_node_class()
        value_with = with_source.source_id
        value_without = without_source.source_id
        assert value_with == "a"
        assert value_without is None


class TestBaseNodeStr:
    def test_includes_id_and_text(self, my_node_class):
        """
        Inputs: node with explicit id and short content.
        Expected result: formatted string contains node id and text line.
        Checks: exact output for short content.
        """
        node = my_node_class(id="node-1")
        value = str(node)
        assert value == "Node ID: node-1\nText: Test content"


class TestBaseNodeGetEmbedding:
    def test_raises_when_embedding_missing(self, my_node_class):
        """
        Inputs: node with embedding unset.
        Expected result: ValueError raised.
        Checks: exception type and message.
        """
        node = my_node_class()
        with pytest.raises(ValueError, match="embedding not set."):
            node.get_embedding()

    def test_returns_embedding(self, my_node_class):
        """
        Inputs: node with embedding set to list of floats.
        Expected result: list returned unchanged.
        Checks: exact list equality.
        """
        node = my_node_class()
        node.embedding = [0.1, 0.2]
        value = node.get_embedding()
        assert value == [0.1, 0.2]


class TestBaseNodeAsRelatedNodeInfo:
    def test_builds_node_reference(self, my_node_class):
        """
        Inputs: node with id, metadata, and hash.
        Expected result: NodeInfo reflects node properties.
        Checks: id, type, metadata, and hash fields.
        """
        node = my_node_class(id="node-1", metadata={"k": "v"})
        ref = node.get_node_info()
        assert ref.id == "node-1"
        assert ref.type == NodeContentType.TEXT
        assert ref.metadata == {"k": "v"}
        assert ref.hash == node.hash


class TestCallMixinAcall:
    def test_delegates_to_call(self, my_node_class):
        """
        Inputs: nodes sequence and simple call mixin implementation.
        Expected result: acall returns the same value as __call__.
        Checks: identity of returned sequence.
        """

        class Dummy(CallMixin):
            def __call__(self, nodes, **kwargs):
                return nodes

        dummy = Dummy()
        nodes = [my_node_class(), my_node_class()]
        result = asyncio.run(dummy.acall(nodes))
        assert result is nodes
