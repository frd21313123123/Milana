# Testing

Полный набор тестов запускается стандартным `unittest` без отдельного test runner:

```powershell
python -m unittest discover -v
```

Для обычной разработки предпочтительнее единая команда:

```powershell
python scripts/check_project.py
```

Она сначала компилирует Python-файлы и запускает Ruff, поэтому очевидные ошибки
ловятся до более дорогого полного тестового прогона.
