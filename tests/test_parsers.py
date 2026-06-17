from dl_acceptance.parsers import (
    parse_dcgm_output,
    parse_dmesg_lines,
    parse_fio_json_or_text,
    parse_nccl_tests_output,
    parse_nvidia_smi_csv,
)


def test_parse_nvidia_smi_csv_handles_na_values():
    text = "2026/06/17 10:00:00, 0, NVIDIA RTX PRO 6000, GPU-abc, N/A, 00000000:81:00.0, 70, 300.5, 600, 2100, 12000, P0, 99, 20, 98304, 1024, Enabled\n"
    rows = parse_nvidia_smi_csv(text)
    assert rows[0]["index"] == 0
    assert rows[0]["serial"] is None
    assert rows[0]["memory.total"] == 98304
    assert rows[0]["power.draw"] == 300.5


def test_dmesg_xid_detection():
    risks = parse_dmesg_lines("[123] NVRM: Xid (PCI:0000:81:00): 79, GPU has fallen off the bus")
    assert risks
    assert risks[0]["severity"] == "CRITICAL"


def test_dmesg_edac_init_is_not_a_risk():
    risks = parse_dmesg_lines("[123] EDAC MC0: Giving out device to module i10nm_edac controller")
    assert risks == []


def test_dmesg_edac_error_detection():
    risks = parse_dmesg_lines("[123] EDAC MC0: 1 CE memory read error on CPU_SrcID#0_MC#0")
    assert risks
    assert risks[0]["severity"] == "HIGH"


def test_dcgm_fail_and_skip_detection():
    parsed = parse_dcgm_output("GPU 0: Pass\nGPU 1: Failed\nPlugin X: Skip - Not Supported")
    assert parsed["failed"] is True
    assert parsed["skipped"] is True
    assert any("Failed" in line for line in parsed["failures"])


def test_nccl_wrong_fail_detection():
    parsed = parse_nccl_tests_output("some line\nwrong result detected\nncclSystemError: unhandled system error")
    assert parsed["failed"] is True
    assert len(parsed["bad_lines"]) >= 2


def test_fio_err_detection_text():
    parsed = parse_fio_json_or_text("fio output\n  write: IOPS=1, err= 5\n")
    assert parsed["failed"] is True


def test_fio_err_detection_json():
    parsed = parse_fio_json_or_text('{"jobs":[{"jobname":"randrw","error":28}]}')
    assert parsed["failed"] is True
    assert "error=28" in parsed["errors"][0]
