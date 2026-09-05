# Smoke data

The four smoke cases are Ghana_1033830 and India_1018317, each with optical and SAR imagery, selected from the publisher's validation split. Their manifests are in `data/manifests/` with `"split": "smoke"`; TIFFs are under `data/samples/sen1floods11-v1.1/`. They are not stored in this documentation folder and are excluded from formal Phase 4 metrics.

Run `.venv/Scripts/python.exe scripts/run_phase4.py --smoke --rule-only` to verify them without API charges.
