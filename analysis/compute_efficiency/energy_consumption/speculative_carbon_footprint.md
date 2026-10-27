# Speculative carbon footprint

For the 4 experiments in `summary.json`, the **2025 Finnish consumption-average proxy** gives **0.74 kg CO₂** for observed GPU electricity, or **0.89 kg CO₂** with PUE-allocated facility overhead.

| Accounting scenario | Factor | GPU only (28.278 kWh) | GPU × PUE (34.383 kWh) |
|---|---|---:|---:|
| Finnish consumption, 2025 proxy | 26 g CO₂/kWh | 0.735 kg CO₂ | 0.894 kg CO₂ |
| Finnish consumption, 2024 historical comparison | 33 g CO₂/kWh | 0.933 kg CO₂ | 1.135 kg CO₂ |
| Assumed renewable lifecycle mix | 17.5 g CO₂e/kWh | 0.495 kg CO₂e | 0.602 kg CO₂e |

Fingrid's consumption-weighted annual factors include imports and exports but exclude lifecycle impacts and guarantees of origin. The national 2025 average is a geographic and temporal proxy for the Helsinki facilities' October 2026 runs, not time-matched data. **These are alternative accounting scenarios, not an uncertainty interval.**

The lifecycle scenario assumes Verda's 50% hydro / 50% wind mix and IPCC global medians of 24/11 g CO₂e/kWh, with wind assumed onshore; these are not verified supplier-specific factors.

Calculation: Σ GPU kWh × factor / 1,000. The overhead scenario first applies published PUE (FIN-01: 1.2, FIN-02: 1.3), assumed applicable to these runs.

This is not a complete study footprint or verified supplier-specific emissions estimate. It excludes missing telemetry and unmonitored periods, non-GPU IT energy, local laptop use, LLM services, and computing-hardware manufacture. PUE allocates facility overhead; it does not account for the omitted IT energy.

Sources: [Fingrid, Table 2](https://www.fingrid.fi/en/electricity-market-information/real-time-co2-emissions-estimate/) (annual factors verified 2026-10-27); [Verda locations and PUE](https://docs.verda.com/welcome-to-verda/locations-and-sustainability/); [IPCC AR5, Table A.III.2](https://www.ipcc.ch/site/assets/uploads/2018/02/ipcc_wg3_ar5_annex-iii.pdf#page=7).

Further refinement: match accepted telemetry intervals to [Fingrid dataset 396](https://data.fingrid.fi/en/datasets/396) (15-minute consumption intensity), preserving telemetry gaps. Not calculated here.

Generated from `summary.json` by `python3 analysis/compute_efficiency/energy_consumption/speculative_carbon_footprint.py` (run from the project root). Observed-energy reports are unchanged.
