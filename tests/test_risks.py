from pathlib import Path

from dl_acceptance.config import AcceptanceConfig
from dl_acceptance.risks import RiskEngine


def test_risk_level_aggregation():
    engine = RiskEngine(AcceptanceConfig({}))
    engine.add("WARN", "GPU", "warm", "details")
    engine.add("HIGH", "GPU", "bad", "details")
    assert engine.highest() == "HIGH"
    assert engine.has_high_or_critical() is True
    assert engine.counts()["WARN"] == 1
    assert engine.counts()["HIGH"] == 1


def test_inventory_gpu_count_risk():
    cfg = AcceptanceConfig({"expected": {"gpu_count": 4, "gpu_name_regex": "RTX PRO 6000", "gpu_memory_gb_min": 90}})
    engine = RiskEngine(cfg)
    engine.evaluate_inventory({"tools": {"nvidia-smi": {"found": True}}, "gpus": []})
    assert engine.has_critical() is True
    assert any("GPU count" in r.title for r in engine.risks)


def test_inventory_gpu_name_and_memory_risk():
    cfg = AcceptanceConfig({"expected": {"gpu_count": 1, "gpu_name_regex": "RTX PRO 6000", "gpu_memory_gb_min": 90}})
    engine = RiskEngine(cfg)
    engine.evaluate_inventory(
        {
            "tools": {"nvidia-smi": {"found": True}},
            "gpus": [{"index": 0, "name": "NVIDIA A10", "memory.total": 24576}],
        }
    )
    assert engine.highest() == "HIGH"
    assert any("model" in r.title.lower() for r in engine.risks)
    assert any("memory" in r.title.lower() for r in engine.risks)


def test_dcgm_unsupported_cuda_is_software_high_not_critical():
    engine = RiskEngine(AcceptanceConfig({}))
    engine.evaluate_command_output("dcgm_r1", "dcgm", "Detected unsupported Cuda version", 226)
    assert engine.highest() == "HIGH"
    assert not engine.has_critical()
    assert engine.risks[0].category == "SOFTWARE"
