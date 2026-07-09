from __future__ import annotations
import pytest
from compute_capacity_orchestrator.schemas.resources import ClusterNode


def test_cluster_node_validation() -> None:
    node = ClusterNode(
        node_id="node-001",
        total_gpus=8,
        available_gpus=6,
        topology_group="rack-a",
    )
    assert node.node_id == "node-001"
    assert node.total_gpus == 8
    assert node.available_gpus == 6
    assert node.topology_group == "rack-a"

    with pytest.raises(ValueError, match="node-002.*available_gpus"):
        ClusterNode(
            node_id="node-002",
            total_gpus=4,
            available_gpus=5,
        )
