from typing import Any

import pytest
from serapeum.core.base.embeddings.types import (
    BaseNode,
    MetadataMode,
    NodeType,
    NodeContentType,
    NodeInfo,
)


@pytest.fixture()
def my_node():
    class MyNode(BaseNode):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)

        @classmethod
        def get_type(cls):
            return NodeContentType.TEXT

        def get_content(self, metadata_mode: MetadataMode = MetadataMode.ALL) -> str:
            return "Test content"

        def set_content(self, value: Any) -> None:
            return super().set_content(value)

        @property
        def hash(self) -> str:
            return super().hash

    return MyNode


def test_get_metadata_str(my_node):
    metadata = {
        "key": "value",
        "forbidden": "true",
    }
    excluded = ["forbidden"]
    node = my_node(
        metadata=metadata,
        excluded_llm_metadata_keys=excluded,
        excluded_embed_metadata_keys=excluded,
    )

    assert node.get_metadata_str(MetadataMode.NONE) == ""
    assert node.get_metadata_str(MetadataMode.LLM) == "key: value"
    assert node.get_metadata_str(MetadataMode.EMBED) == "key: value"


def test_node_id(my_node):
    n = my_node()
    n.id = "this"
    assert n.id == "this"


def test_linked_source_node(my_node):
    n1 = my_node()
    n2 = my_node(links={NodeType.SOURCE: NodeInfo(id=n1.id)})
    assert n2.linked_nodes.source.hash == n1.hash
    assert n1.linked_nodes.source is None

    with pytest.raises(
        ValueError, match="The Source Node must be a single NodeInfo object"
    ):
        my_node(
            links={NodeType.SOURCE: [NodeInfo(id=n1.id)]}
        ).linked_nodes.source


def test_linked_prev_node(my_node):
    n1 = my_node()
    n2 = my_node(
        links={NodeType.PREVIOUS: NodeInfo(id=n1.id)}
    )
    assert n2.linked_nodes.previous.hash == n1.hash
    assert n1.linked_nodes.previous is None

    with pytest.raises(
        ValueError, match="The Previous Node must be a single NodeInfo object"
    ):
        my_node(
            links={NodeType.PREVIOUS: [NodeInfo(id=n1.id)]}
        ).linked_nodes.previous


def test_linked_next_node(my_node):
    n1 = my_node()
    n2 = my_node(links={NodeType.NEXT: NodeInfo(id=n1.id)})
    assert n2.linked_nodes.next.hash == n1.hash
    assert n1.linked_nodes.next is None

    with pytest.raises(
        ValueError, match="The Next Node must be a single NodeInfo object"
    ):
        my_node(
            links={NodeType.NEXT: [NodeInfo(id=n1.id)]}
        ).linked_nodes.next


def test_linked_parent_node(my_node):
    n1 = my_node()
    n2 = my_node(links={NodeType.PARENT: NodeInfo(id=n1.id)})
    assert n2.linked_nodes.parent.hash == n1.hash
    assert n1.linked_nodes.parent is None

    with pytest.raises(
        ValueError, match="The Parent Node must be a single NodeInfo object"
    ):
        my_node(
            links={NodeType.PARENT: [NodeInfo(id=n1.id)]}
        ).linked_nodes.parent


def test_linked_child_node(my_node):
    n1 = my_node()
    n2 = my_node(
        links={NodeType.CHILD: [NodeInfo(id=n1.id)]}
    )
    assert n2.linked_nodes.children[0].hash == n1.hash
    assert n1.linked_nodes.children is None

    with pytest.raises(
        ValueError, match="Child Nodes must be a list of NodeInfo objects"
    ):
        my_node(
            links={NodeType.CHILD: NodeInfo(id=n1.id)}
        ).linked_nodes.children


def test___str__(my_node):
    n = my_node()
    n.id = "test_node"
    assert str(n) == "Node ID: test_node\nText: Test content"


def test_get_embedding(my_node):
    n = my_node()
    with pytest.raises(ValueError, match="embedding not set."):
        n.get_embedding()
    n.embedding = [0.0, 0.0]
    assert n.get_embedding() == [0.0, 0.0]


def test_as_related_node_info(my_node):
    n = my_node(id="test_node")
    assert n.get_node_info().id == "test_node"
