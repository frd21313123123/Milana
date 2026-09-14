# CI

CI выполняет одинаковый базовый набор проверок на Windows и Ubuntu:

```text
compileall -> Ruff correctness -> unittest discover
```

Windows проверяет основной runtime проекта и Windows-specific ветви. Ubuntu
помогает замечать случайные платформенные зависимости в общем Python-коде.

Локально тот же набор запускается командой:

```powershell
python scripts/check_project.py
```
