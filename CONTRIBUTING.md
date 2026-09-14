# Contributing

Перед изменениями установите основные и dev-зависимости:

```powershell
python -m pip install -r requirements.txt -r requirements-dev.txt
```

Перед коммитом запустите единый локальный чек:

```powershell
python scripts/check_project.py
```

Он проверяет компиляцию Python-файлов, критические ошибки через Ruff и полный
набор `unittest`. Pull request также должен пройти CI на Windows и Ubuntu.

Не добавляйте в Git `.env`, Telegram session-файлы, локальные базы, runtime-data
и логи. Эти пути уже исключены в `.gitignore`.
