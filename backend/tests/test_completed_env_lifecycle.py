"""A finished run must unlock the report while its environment stays alive.

In wait mode the runner process outlives the round loop to serve interviews
(issue #779). Completion is published once every platform has logged
simulation_end and the environment is alive, after the Zep drain.
"""

import json

import pytest

from app.services import simulation_runner as runner_module
from app.services.simulation_runner import (
    RunnerStatus,
    SimulationRunState,
    SimulationRunner,
)


class FakeProcess:
    """Alive for `alive_polls` polls, then exits with `exit_code`."""

    pid = 4242

    def __init__(self, alive_polls, exit_code=0):
        self.alive_polls = alive_polls
        self.exit_code = exit_code
        self.returncode = None

    def poll(self):
        if self.alive_polls > 0:
            self.alive_polls -= 1
            return None
        self.returncode = self.exit_code
        return self.exit_code


def _write_finished_run(sim_dir, env_alive):
    twitter_dir = sim_dir / "twitter"
    twitter_dir.mkdir(parents=True)
    (twitter_dir / "actions.jsonl").write_text(
        '{"event_type":"simulation_end","total_rounds":1,"total_actions":0}\n',
        encoding="utf-8",
    )
    if env_alive:
        (sim_dir / "env_status.json").write_text(
            json.dumps({"status": "alive"}), encoding="utf-8"
        )


@pytest.fixture
def runner_env(monkeypatch, tmp_path):
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(runner_module.time, "sleep", lambda _seconds: None)
    published = []
    monkeypatch.setattr(
        SimulationRunner,
        "_sync_simulation_status",
        classmethod(
            lambda _cls, _sim_id, status, *_args, **_kwargs: published.append(status)
        ),
    )
    ids = []
    yield tmp_path, published, ids
    for simulation_id in ids:
        for registry in (
            SimulationRunner._run_states,
            SimulationRunner._processes,
            SimulationRunner._monitor_threads,
            SimulationRunner._graph_memory_enabled,
        ):
            registry.pop(simulation_id, None)
        SimulationRunner._manual_stop_requests.discard(simulation_id)


def _start_monitor_state(simulation_id, process):
    state = SimulationRunState(
        simulation_id=simulation_id,
        runner_status=RunnerStatus.RUNNING,
        twitter_running=True,
    )
    SimulationRunner._run_states[simulation_id] = state
    SimulationRunner._processes[simulation_id] = process
    return state


def test_completion_is_published_while_env_alive_after_zep_drain(
    monkeypatch, runner_env
):
    tmp_path, published, ids = runner_env
    simulation_id = "sim-alive"
    ids.append(simulation_id)
    _write_finished_run(tmp_path / simulation_id, env_alive=True)
    # Exits later with SIGTERM, as after close-env or stop.
    process = FakeProcess(alive_polls=3, exit_code=-15)
    state = _start_monitor_state(simulation_id, process)
    SimulationRunner._graph_memory_enabled[simulation_id] = True

    drained_while_alive = []
    monkeypatch.setattr(
        runner_module.ZepGraphMemoryManager,
        "stop_updater",
        classmethod(
            lambda _cls, _sim_id: drained_while_alive.append(process.alive_polls > 0)
        ),
    )

    SimulationRunner._monitor_simulation(simulation_id)

    assert drained_while_alive == [True]
    assert published == [RunnerStatus.STOPPING, RunnerStatus.COMPLETED]
    # The later non-zero exit must not overwrite the published result.
    assert state.runner_status == RunnerStatus.COMPLETED
    assert state.error is None


def test_without_live_env_completion_waits_for_process_exit(runner_env):
    tmp_path, published, ids = runner_env
    simulation_id = "sim-no-wait"
    ids.append(simulation_id)
    _write_finished_run(tmp_path / simulation_id, env_alive=False)
    state = _start_monitor_state(simulation_id, FakeProcess(alive_polls=3))

    SimulationRunner._monitor_simulation(simulation_id)

    assert published == [RunnerStatus.COMPLETED]
    assert state.runner_status == RunnerStatus.COMPLETED


def test_manual_stop_before_completion_still_ends_stopped(runner_env):
    tmp_path, published, ids = runner_env
    simulation_id = "sim-manual"
    ids.append(simulation_id)
    _write_finished_run(tmp_path / simulation_id, env_alive=True)
    state = _start_monitor_state(simulation_id, FakeProcess(alive_polls=3))
    SimulationRunner._manual_stop_requests.add(simulation_id)

    SimulationRunner._monitor_simulation(simulation_id)

    assert published == [RunnerStatus.STOPPED]
    assert state.runner_status == RunnerStatus.STOPPED


def test_stopping_a_completed_run_closes_env_and_keeps_completed(
    monkeypatch, runner_env
):
    _tmp_path, published, ids = runner_env
    simulation_id = "sim-close"
    ids.append(simulation_id)
    state = SimulationRunState(
        simulation_id=simulation_id,
        runner_status=RunnerStatus.COMPLETED,
    )
    SimulationRunner._run_states[simulation_id] = state
    SimulationRunner._processes[simulation_id] = FakeProcess(alive_polls=10)
    terminated = []
    monkeypatch.setattr(
        SimulationRunner,
        "_terminate_process",
        classmethod(lambda _cls, _process, sim_id: terminated.append(sim_id)),
    )

    result = SimulationRunner.stop_simulation(simulation_id)

    assert terminated == [simulation_id]
    assert result.runner_status == RunnerStatus.COMPLETED
    assert simulation_id not in SimulationRunner._manual_stop_requests
    assert published == []


def test_start_rejects_while_completed_env_process_is_alive(runner_env):
    tmp_path, _published, ids = runner_env
    simulation_id = "sim-restart"
    ids.append(simulation_id)
    sim_dir = tmp_path / simulation_id
    sim_dir.mkdir()
    (sim_dir / "simulation_config.json").write_text(
        json.dumps({"time_config": {}}), encoding="utf-8"
    )
    SimulationRunner._run_states[simulation_id] = SimulationRunState(
        simulation_id=simulation_id,
        runner_status=RunnerStatus.COMPLETED,
    )
    SimulationRunner._processes[simulation_id] = FakeProcess(alive_polls=10)

    with pytest.raises(ValueError):
        SimulationRunner.start_simulation(simulation_id, platform="twitter")
