# Clinical Radiomics Platform

This repository now follows the layered backend architecture described in
`docs/clinical-radiomics-platform-architecture.md`, while keeping the current
desktop-oriented FastAPI UI and the existing radiomics workflow behavior.

## Current Backend Layers

- `pyrad_workflow/api/`: HTTP routes and page delivery.
- `pyrad_workflow/application/`: platform orchestration, preview, cache, and job submission.
- `pyrad_workflow/domain/`: validation, extraction, selection, training, and full-pipeline workflows.
- `pyrad_workflow/infrastructure/`: workspace detection, filesystem helpers, and in-memory job storage.
- `pyrad_workflow/static/`: local presentation layer assets.
- `app.py`: top-level runtime entry.

The lower-level algorithm modules are still intentionally simple:

- `pyrad_workflow/validation.py`
- `pyrad_workflow/extraction.py`
- `pyrad_workflow/modeling.py`
- `pyrad_workflow/io_utils.py`

Compatibility wrappers remain in place for older imports such as
`pyrad_workflow.webapp`, `pyrad_workflow.services`, `pyrad_workflow.app_config`,
and `pyrad_workflow.workspace`.

## Install

```powershell
conda env create -f environment.yml
conda activate pyrad-nifti
python -m pip install -e . --no-deps
python scripts/check_install.py
```

## Run The Web App

```powershell
uvicorn app:app --host 127.0.0.1 --port 8181 --reload
```

Then open [http://127.0.0.1:8181](http://127.0.0.1:8181).

Available pages:

- `/validate`
- `/extract`
- `/select`
- `/train`
- `/full`

## Default Files

- Manifest example: `outputs/examples/test_data_manifest.csv`
- Radiomics params: `configs/ct_radiomics.yaml`
- Labels example: `outputs/examples/test_data_labels.csv`
- Feature example: `outputs/examples/test_data_features.csv`

## Outputs

- `validation_report.csv`
- `features.csv`
- `feature_failures.csv`
- `prepared_features.csv`
- `selected_features.csv`
- `selection_summary.csv`
- `metrics.csv`

## Architecture Note

The broader platform note is available at
`docs/clinical-radiomics-platform-architecture.md`. This repository implements
the backend layering and local workflow execution model from that document, but
does not yet include the full React, Celery, Redis, DICOM/RT Structure, or SHAP
production stack described as the long-term target.
