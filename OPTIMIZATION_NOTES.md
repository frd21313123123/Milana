# Project optimization baseline

This branch adds a minimal quality gate without changing Milana's runtime behavior.

## Local check

Install the normal and development dependencies, then run:

```powershell
python -m pip install -r requirements.txt -r requirements-dev.txt
python scripts/check_project.py
```

The command compiles every Python module, runs correctness-focused Ruff checks,
and executes the complete `unittest` suite.

## CI

GitHub Actions runs the same checks on both Windows and Ubuntu. Windows remains
the primary runtime for the standalone control panel, while Linux catches
platform assumptions in shared code and tests.

## Known debt

`milana_service.py` currently references `GEMINI_LLM_CHOICE` without importing it.
Ruff keeps undefined-name checking enabled for the rest of the repository and
temporarily excludes this large legacy service module from F821 until the service
is split into smaller components and the import is fixed safely.
