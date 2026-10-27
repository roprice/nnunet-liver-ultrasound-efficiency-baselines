#!/usr/bin/env python3
"""Generate a brief, speculative carbon estimate from the observed energy summary."""

import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
FINGRID_2025_G_CO2_PER_KWH = 26.0
FINGRID_2024_G_CO2_PER_KWH = 33.0
HYDRO_G_CO2E_PER_KWH = 24.0
ONSHORE_WIND_G_CO2E_PER_KWH = 11.0
ASSUMED_MIX = "50% hydroelectric, 50% wind power"


def render_report(summary):
    locations = summary["datacenter_metadata"]["locations"]
    experiments = summary["experiments"]
    if not experiments:
        raise ValueError("No experiments in energy summary")
    names = set()
    gpu_energies, adjusted_energies = [], []
    for experiment in experiments:
        name = experiment["experiment"]
        if name in names:
            raise ValueError(f"Duplicate experiment: {name}")
        names.add(name)
        code = experiment["location_code"]
        location = locations[code]
        if location["country"] != "Finland":
            raise ValueError(f"Finnish grid factor unsupported for {code}")
        if code not in ("FIN-01", "FIN-02") or location["published_electricity_mix"] != ASSUMED_MIX:
            raise ValueError(f"Electricity mix assumption unsupported for {code}")
        energy = experiment["gpu_kwh"]
        pue = location["published_pue"]
        if not math.isfinite(energy) or energy < 0 or not math.isfinite(pue) or pue < 1:
            raise ValueError(f"Invalid energy or PUE for {name}")
        if experiment["published_pue"] != pue:
            raise ValueError(f"Inconsistent PUE for {name}")
        gpu_energies.append(energy)
        adjusted_energies.append(energy * pue)
    intensity = (HYDRO_G_CO2E_PER_KWH + ONSHORE_WIND_G_CO2E_PER_KWH) / 2
    gpu_kwh = math.fsum(gpu_energies)
    adjusted_kwh = math.fsum(adjusted_energies)
    pue_values = ", ".join(f"{code}: {locations[code]['published_pue']:g}" for code in
                           sorted({row["location_code"] for row in experiments}))
    return f"""# Speculative carbon footprint

For the {len(experiments)} experiments in `summary.json`, the **2025 Finnish consumption-average proxy** gives **{gpu_kwh * FINGRID_2025_G_CO2_PER_KWH / 1000:.2f} kg CO₂** for observed GPU electricity, or **{adjusted_kwh * FINGRID_2025_G_CO2_PER_KWH / 1000:.2f} kg CO₂** with PUE-allocated facility overhead.

| Accounting scenario | Factor | GPU only ({gpu_kwh:.3f} kWh) | GPU × PUE ({adjusted_kwh:.3f} kWh) |
|---|---|---:|---:|
| Finnish consumption, 2025 proxy | {FINGRID_2025_G_CO2_PER_KWH:g} g CO₂/kWh | {gpu_kwh * FINGRID_2025_G_CO2_PER_KWH / 1000:.3f} kg CO₂ | {adjusted_kwh * FINGRID_2025_G_CO2_PER_KWH / 1000:.3f} kg CO₂ |
| Finnish consumption, 2024 historical comparison | {FINGRID_2024_G_CO2_PER_KWH:g} g CO₂/kWh | {gpu_kwh * FINGRID_2024_G_CO2_PER_KWH / 1000:.3f} kg CO₂ | {adjusted_kwh * FINGRID_2024_G_CO2_PER_KWH / 1000:.3f} kg CO₂ |
| Assumed renewable lifecycle mix | {intensity:g} g CO₂e/kWh | {gpu_kwh * intensity / 1000:.3f} kg CO₂e | {adjusted_kwh * intensity / 1000:.3f} kg CO₂e |

Fingrid's consumption-weighted annual factors include imports and exports but exclude lifecycle impacts and guarantees of origin. The national 2025 average is a geographic and temporal proxy for the Helsinki facilities' October 2026 runs, not time-matched data. **These are alternative accounting scenarios, not an uncertainty interval.**

The lifecycle scenario assumes Verda's 50% hydro / 50% wind mix and IPCC global medians of {HYDRO_G_CO2E_PER_KWH:g}/{ONSHORE_WIND_G_CO2E_PER_KWH:g} g CO₂e/kWh, with wind assumed onshore; these are not verified supplier-specific factors.

Calculation: Σ GPU kWh × factor / 1,000. The overhead scenario first applies published PUE ({pue_values}), assumed applicable to these runs.

This is not a complete study footprint or verified supplier-specific emissions estimate. It excludes missing telemetry and unmonitored periods, non-GPU IT energy, local laptop use, LLM services, and computing-hardware manufacture. PUE allocates facility overhead; it does not account for the omitted IT energy.

Sources: [Fingrid, Table 2](https://www.fingrid.fi/en/electricity-market-information/real-time-co2-emissions-estimate/) (annual factors verified 2026-10-27); [Verda locations and PUE](https://docs.verda.com/welcome-to-verda/locations-and-sustainability/); [IPCC AR5, Table A.III.2](https://www.ipcc.ch/site/assets/uploads/2018/02/ipcc_wg3_ar5_annex-iii.pdf#page=7).

Further refinement: match accepted telemetry intervals to [Fingrid dataset 396](https://data.fingrid.fi/en/datasets/396) (15-minute consumption intensity), preserving telemetry gaps. Not calculated here.

Generated from `summary.json` by `python3 analysis/compute_efficiency/energy_consumption/speculative_carbon_footprint.py` (run from the project root). Observed-energy reports are unchanged.
"""


def main():
    summary = json.loads((HERE / "summary.json").read_text())
    report = render_report(summary)
    output = HERE / "speculative_carbon_footprint.md"
    if output.is_symlink():
        raise ValueError(f"Refusing report symlink: {output}")
    output.write_text(report)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
