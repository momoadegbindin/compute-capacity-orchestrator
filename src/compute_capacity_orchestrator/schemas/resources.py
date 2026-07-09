"""Schemas for physical compute resources."""

from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ClusterNode:
    """A physical compute node with GPU capacity available to the scheduler."""

    node_id: str
    total_gpus: int
    available_gpus: int
    topology_group: str | None = None

    def __post_init__(self) -> None:
        node_ref = self.node_id or "<missing-node-id>"

        if not self.node_id:
            raise ValueError("ClusterNode node_id must be non-empty")

        if self.total_gpus < 0:
            raise ValueError(f"ClusterNode {node_ref}: total_gpus cannot be negative")

        if self.available_gpus < 0:
            raise ValueError(f"ClusterNode {node_ref}: available_gpus cannot be negative")

        if self.available_gpus > self.total_gpus:
            raise ValueError(
                f"ClusterNode {node_ref}: available_gpus cannot exceed total_gpus"
            )
