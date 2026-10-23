import pytest

from triton_ops_lab.analysis.safety import VENDOR_WINOGRAD, vendor_race_attribution
from triton_ops_lab.analysis.safety import producer_tracking_control


def warning(kernel):
    return f"========= Warning: Race reported between Write access at {kernel}+0x10\n=========     and Read access at {kernel}+0x20 [10 hazards]\n"


def test_vendor_attribution_preserves_global_failure_and_rejects_project_races():
    log = (
        "..."
        + warning(VENDOR_WINOGRAD)
        + "========= RACECHECK SUMMARY: 1 hazard displayed (0 errors, 1 warning)\n"
    )
    repro = log + "VENDOR_REPRO_COMPLETE\n"
    result = vendor_race_attribution(log, repro)
    assert result["global_racecheck_status"] == "failed"
    with pytest.raises(ValueError, match="unverified kernel"):
        vendor_race_attribution(log + warning("_conv"), repro)
    with pytest.raises(ValueError, match="counts"):
        vendor_race_attribution(log.replace("1 hazard displayed", "2 hazards displayed"), repro)


def test_producer_tracking_requires_physical_result_and_a_real_filter_effect():
    full = "PRODUCER_PROBE_COMPLETE 1.0\n========= ERROR SUMMARY: 0 errors\n"
    filtered = "Uninitialized __global__ memory read\n at _reader+0x10\nPRODUCER_PROBE_COMPLETE 1.0\n========= ERROR SUMMARY: 1 error\n"
    assert producer_tracking_control(full, filtered)["physical_result"] == 1.0
    with pytest.raises(ValueError, match="filter effect"):
        producer_tracking_control(full, full)
    with pytest.raises(ValueError, match="filter effect"):
        producer_tracking_control(full, filtered.replace("1.0", "0.0"))
