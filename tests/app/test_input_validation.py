import math

from app.views.input_validation import validate_number, validate_order
from app.views import control_config as cfg
from app.views.simulation_view import validate_simulation_controls
from app.views.snapshot_view import validate_snapshot_controls

def test_validate_number_accepts_valid_integer() -> None:
    assert (
        validate_number(
            name="Horizon",
            value=25,
            min_value=1,
            max_value=200,
            integer=True,
        )
        == []
    )


def test_validate_number_accepts_valid_float() -> None:
    assert (
        validate_number(
            name="Arrival rate",
            value=3.5,
            min_value=0.0,
            max_value=20.0,
        )
        == []
    )


def test_validate_number_rejects_below_minimum() -> None:
    assert validate_number(
        name="Horizon",
        value=0,
        min_value=1,
        max_value=200,
        integer=True,
    ) == ["Horizon must be between 1 and 200."]


def test_validate_number_rejects_above_maximum() -> None:
    assert validate_number(
        name="Horizon",
        value=201,
        min_value=1,
        max_value=200,
        integer=True,
    ) == ["Horizon must be between 1 and 200."]


def test_validate_number_rejects_non_integer_when_required() -> None:
    assert validate_number(
        name="Number of nodes",
        value=3.5,
        min_value=1,
        max_value=100,
        integer=True,
    ) == ["Number of nodes must be an integer."]


def test_validate_number_rejects_non_finite_value() -> None:
    assert validate_number(
        name="Arrival rate",
        value=math.inf,
        min_value=0.0,
        max_value=20.0,
    ) == ["Arrival rate must be finite."]


def test_validate_order_accepts_valid_range() -> None:
    assert (
        validate_order(
            low_name="Minimum GPU demand",
            low_value=1,
            high_name="maximum GPU demand",
            high_value=16,
        )
        == []
    )


def test_validate_order_rejects_inverted_range() -> None:
    assert validate_order(
        low_name="Minimum GPU demand",
        low_value=32,
        high_name="maximum GPU demand",
        high_value=16,
    ) == ["Minimum GPU demand must be less than or equal to maximum GPU demand."]


def test_simulation_validation_accepts_greedy_without_mip_controls() -> None:
    assert (
        validate_simulation_controls(
            scheduler_name=cfg.GREEDY_SCHEDULER,
            horizon=25,
            num_nodes=8,
            arrival_rate=3.0,
            seed=42,
            deadline_penalty_weight=1.0,
            time_limit_seconds=None,
            relative_gap=None,
        )
        == []
    )


def test_simulation_validation_rejects_exact_mip_without_mip_controls() -> None:
    errors = validate_simulation_controls(
        scheduler_name=cfg.EXACT_MIP_SCHEDULER,
        horizon=10,
        num_nodes=8,
        arrival_rate=3.0,
        seed=42,
        deadline_penalty_weight=1.0,
        time_limit_seconds=None,
        relative_gap=None,
    )

    assert "Exact MIP time limit is required." in errors
    assert "Relative gap is required for Exact MIP." in errors


def test_simulation_validation_rejects_exact_mip_large_horizon() -> None:
    errors = validate_simulation_controls(
        scheduler_name=cfg.EXACT_MIP_SCHEDULER,
        horizon=cfg.PUBLIC_EXACT_SIMULATION_HORIZON_MAX + 1,
        num_nodes=8,
        arrival_rate=1.0,
        seed=42,
        deadline_penalty_weight=1.0,
        time_limit_seconds=1,
        relative_gap=0.10,
    )

    assert any("Exact MIP simulation solves one optimization model per step" in error for error in errors)


def test_snapshot_validation_accepts_small_greedy_without_large_controls() -> None:
    assert (
        validate_snapshot_controls(
            scheduler_name=cfg.GREEDY_SCHEDULER,
            experiment_size=cfg.SMALL_EXPERIMENT,
            deadline_penalty_weight=0.0,
            time_limit_seconds=None,
            relative_gap=None,
            node_a_available=5,
            node_b_available=3,
        )
        == []
    )


def test_snapshot_validation_rejects_large_greedy_missing_large_controls() -> None:
    errors = validate_snapshot_controls(
        scheduler_name=cfg.GREEDY_SCHEDULER,
        experiment_size=cfg.LARGE_EXPERIMENT,
        deadline_penalty_weight=0.0,
        time_limit_seconds=None,
        relative_gap=None,
    )

    assert "Random seed is required." in errors
    assert "Number of nodes is required." in errors
    assert "Number of jobs is required." in errors
    assert "Minimum GPU demand is required." in errors
    assert "Maximum GPU demand is required." in errors


def test_snapshot_validation_rejects_large_inverted_gpu_range() -> None:
    errors = validate_snapshot_controls(
        scheduler_name=cfg.GREEDY_SCHEDULER,
        experiment_size=cfg.LARGE_EXPERIMENT,
        deadline_penalty_weight=0.0,
        time_limit_seconds=None,
        relative_gap=None,
        seed=27,
        num_nodes=20,
        num_jobs=100,
        gpu_demand_min=16,
        gpu_demand_max=1,
    )

    assert "Minimum GPU demand must be less than or equal to maximum GPU demand." in errors
