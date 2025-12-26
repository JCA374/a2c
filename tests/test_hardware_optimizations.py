import pytest


def _import_train_a2c():
    pytest.importorskip("torch")
    import train_a2c
    return train_a2c


def test_cpu_optimization_auto_threads(monkeypatch):
    train_a2c = _import_train_a2c()
    calls = {}

    def fake_set_num_threads(value):
        calls["threads"] = value

    monkeypatch.setattr(train_a2c.torch, "set_num_threads", fake_set_num_threads)

    num_envs = train_a2c.resolve_num_envs(None, cpu_count=6)
    cpu_threads = train_a2c.resolve_cpu_threads(None, cpu_count=6)
    applied_threads = train_a2c.configure_torch_threads(cpu_threads)

    assert num_envs == 6
    assert applied_threads == 6
    assert calls["threads"] == 6


def test_ram_optimization_caps_rollout_length():
    train_a2c = _import_train_a2c()
    rollout_length = 128
    num_envs = 8
    obs_dim = 40
    action_dim = 3
    max_ram_mb = 1
    per_step_bytes = num_envs * (obs_dim + action_dim + 6) * 4
    expected = min(rollout_length, max(1, int((max_ram_mb * 1024 * 1024) // per_step_bytes)))

    adjusted = train_a2c.adjust_rollout_length_for_ram(
        rollout_length, num_envs, obs_dim, action_dim, max_ram_mb
    )

    assert adjusted == expected
