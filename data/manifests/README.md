# Evaluation manifests

Each JSON file is one immutable labelled case and conforms to `EvaluationCase`.

- `evaluation` split: formal, versioned metrics only. Phase 4 requires at least 24 cases (12 optical and 12 SAR) before reporting aggregate results.
- `smoke` split: 2--4 small development cases, excluded by default from formal reports.
- `image_uri` and `label_uri` may be relative to the manifest. Labels use 1=water, 0=non-water; -1/nodata and other classes are excluded from scoring, never cast directly to Boolean.
- Record source URL, license, sensor type, bands, CRS, resolution, bounds and hashes. Do not add unlicensed data or per-case tuned parameters.

The water corpus contains 24 evaluation manifests (12 optical + 12 SAR) and 4 disjoint smoke manifests. These correspond to 12 evaluation scene pairs and 2 smoke scene pairs. Resolution tiers are derived UTM 10/20/30m grids.

The building corpus adds 12 fixed optical evaluation manifests from the WHU Building Dataset test split. Run `.venv/Scripts/python.exe scripts/run_building_phase4.py --prepare` to acquire and evaluate it. Its source PNGs do not carry CRS/geolocation; conversion uses a clearly marked local metric grid only to satisfy the GeoTIFF workflow contract, and must not be presented as original geographic metadata.

Run `.venv/Scripts/python.exe scripts/run_phase4.py --prepare` from the project root. It reads the project `.env`, downloads missing public research inputs, and writes separate rule/real-LLM reports. Missing credentials are an error, not a silently substituted rule run. `--rule-only` makes no model calls; `--smoke` excludes all evaluation cases.

Each `baseline_metrics.reference` points to a first-run fixed-rule metric file under `data/baselines`. Source/derived hashes, original split, acquisition date, full-chip crop and resampling provenance are recorded in each manifest. Selection is frozen in `data/phase4-selection.json`.

Publisher STAC declares **proprietary**, not CC/open licensing. Do not redistribute the downloaded imagery or labels with the repository or imply commercial permission. See [full methodology and licensing caveats](../../docs/PHASE4_EVALUATION.md). Generated runtime evidence is local and should not be committed.
