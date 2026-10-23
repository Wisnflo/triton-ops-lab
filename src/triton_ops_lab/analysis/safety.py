"""Attribute an observed vendor warning without converting the global run to a pass."""

import re


VENDOR_WINOGRAD = "_5x_cudnn_ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148n_nt_v1"


def producer_tracking_control(unfiltered, filtered):
    first = [int(x) for x in re.findall(r"ERROR SUMMARY:\s*(\d+) errors?", unfiltered)]
    second = [int(x) for x in re.findall(r"ERROR SUMMARY:\s*(\d+) errors?", filtered)]
    if (
        not first
        or max(first) != 0
        or not second
        or min(second) <= 0
        or "at _reader+" not in filtered
        or any("PRODUCER_PROBE_COMPLETE 1.0" not in x for x in (unfiltered, filtered))
        or any("Internal Sanitizer Error" in x for x in (unfiltered, filtered))
    ):
        raise ValueError("Producer tracking control did not establish the filter effect")
    return dict(
        unfiltered_errors=first,
        filtered_errors=second,
        physical_result=1.0,
        conclusion="Excluding the input-producing kernel loses global initialization tracking; filtered initialization evidence cannot certify global inputs.",
    )


def vendor_race_attribution(log, reproduction):
    counts = []
    for text in (log, reproduction):
        summaries = re.findall(
            r"RACECHECK SUMMARY:\s*(\d+) hazards? displayed \((\d+) errors?,\s*(\d+) warnings?\)",
            text,
        )
        kernels = set(re.findall(r"access at (\S+)\+0x[0-9a-f]+", text))
        # pytest progress dots can precede a sanitizer header on the same line.
        headers = re.findall(r"=+\s*Warning: Race reported between[^\n]*", text)
        warnings = re.findall(r"=+\s*(?:Warning|WARNING)[: ][^\n]*", text)
        if (
            len(summaries) != 1
            or kernels != {VENDOR_WINOGRAD}
            or not headers
            or headers != warnings
            or any(
                x in text
                for x in (
                    "Internal Sanitizer Error",
                    "ERROR:",
                    "Exception ignored in atexit callback",
                )
            )
        ):
            raise ValueError("Race attribution includes an unverified kernel or tool failure")
        hazards, errors, warnings_count = map(int, summaries[0])
        if errors or hazards != warnings_count or warnings_count != len(headers):
            raise ValueError("Race warning counts do not match the raw summary")
        counts.append(warnings_count)
    if "VENDOR_REPRO_COMPLETE" not in reproduction:
        raise ValueError("Isolated vendor reproduction did not finish")
    return dict(
        global_racecheck_status="failed",
        vendor_kernel=VENDOR_WINOGRAD,
        original_warning_count=counts[0],
        isolated_vendor_warning_count=counts[1],
        note="Attribution only. This is not a claim that the vendor warnings are false positives or resolved; an exact vendor exclusion must be disclosed in follow-up coverage.",
    )
