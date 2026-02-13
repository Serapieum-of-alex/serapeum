from typing import Any

import pytest
from serapeum.core.base.embeddings.types import (
    BaseNode,
    MetadataMode,
    NodeRelationship,
    ObjectType,
    NodeReference,
)


@pytest.fixture()
def my_node():
    class MyNode(BaseNode):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)

        @classmethod
        def get_type(cls):
            return ObjectType.TEXT

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
    n2 = my_node(relationships={NodeRelationship.SOURCE: NodeReference(id=n1.id)})
    assert n2.linked.source.hash == n1.hash
    assert n1.linked.source is None

    with pytest.raises(
        ValueError, match="Source object must be a single NodeReference object"
    ):
        my_node(
            relationships={NodeRelationship.SOURCE: [NodeReference(id=n1.id)]}
        ).linked.source


def test_linked_prev_node(my_node):
    n1 = my_node()
    n2 = my_node(
        relationships={NodeRelationship.PREVIOUS: NodeReference(id=n1.id)}
    )
    assert n2.linked.previous.hash == n1.hash
    assert n1.linked.previous is None

    with pytest.raises(
        ValueError, match="Previous object must be a single NodeReference object"
    ):
        my_node(
            relationships={NodeRelationship.PREVIOUS: [NodeReference(id=n1.id)]}
        ).linked.previous


def test_linked_next_node(my_node):
    n1 = my_node()
    n2 = my_node(relationships={NodeRelationship.NEXT: NodeReference(id=n1.id)})
    assert n2.linked.next.hash == n1.hash
    assert n1.linked.next is None

    with pytest.raises(
        ValueError, match="Next object must be a single NodeReference object"
    ):
        my_node(
            relationships={NodeRelationship.NEXT: [NodeReference(id=n1.id)]}
        ).linked.next


def test_linked_parent_node(my_node):
    n1 = my_node()
    n2 = my_node(relationships={NodeRelationship.PARENT: NodeReference(id=n1.id)})
    assert n2.linked.parent.hash == n1.hash
    assert n1.linked.parent is None

    with pytest.raises(
        ValueError, match="Parent object must be a single NodeReference object"
    ):
        my_node(
            relationships={NodeRelationship.PARENT: [NodeReference(id=n1.id)]}
        ).linked.parent


def test_linked_child_node(my_node):
    n1 = my_node()
    n2 = my_node(
        relationships={NodeRelationship.CHILD: [NodeReference(id=n1.id)]}
    )
    assert n2.linked.children[0].hash == n1.hash
    assert n1.linked.children is None

    with pytest.raises(
        ValueError, match="Child objects must be a list of NodeReference objects"
    ):
        my_node(
            relationships={NodeRelationship.CHILD: NodeReference(id=n1.id)}
        ).linked.children


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
    assert n.as_related_node_info().id == "test_node"
