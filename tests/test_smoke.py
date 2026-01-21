import spd_transport
from spd_transport.train import TrainConfig, run_training


def test_smoke_run_training():
    config = TrainConfig(steps=5, log_every=5)
    loss = run_training(config)
    assert loss >= 0.0


def test_imports():
    assert spd_transport.cycle_edges(3) == [(0, 1), (1, 2), (2, 0)]
