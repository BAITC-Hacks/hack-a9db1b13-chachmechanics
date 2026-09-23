# WindOracle: распределение работы

Проект разделён между тремя участниками. Участник 1 реализовал импорт, NOAA GFS,
as-of, хранение, replay, CLI, экспорт и service. Модели и интерфейс интегрируются
по docs/CONTRACTS.md. Фактический статус и проверенные команды участника 1 — docs/DECISIONS.md.

Имя Python-пакета — `windoracle`. Физическая папка исходников переименована
командой в `src/TwinTurbo.ai/`; все пути `src/windoracle/` в карте владения ниже
обозначают соответствующие файлы в этой папке. Выполнять `pip install -e .`
перед тестами и запуском. Скрытые .gitkeep сохраняют пустые директории в Git.

## Участник 1 — данные, погода, время и интеграция

Участник 1 публикует в `main` по указанию пользователя. Перед каждым push —
fetch, проверка изменений команды, слияние при необходимости и тесты. Force-push запрещён.

Сначала: подтвердить время/координаты и получить один реальный погодный run. Затем реализовать схемы, импорт, as-of, service, сохранение и replay. Этот участник объединяет изменения команды и владеет общими контрактами.

Критерий: по origin_time подготовлен допустимый снимок, вызвана модель, сохранён выпуск, новый run создаёт новую версию.

## Участник 2 — модели и качество

Ветка: `feat/forecast-models`.

Сначала: кривая мощности и baseline. Затем метрики, bias, интервалы и ML по оставшемуся времени. Модели принимают готовый снимок и не скачивают данные.

Критерий: PredictionBatch по общей схеме и воспроизводимый EvaluationReport. Нет утечки будущих целей при обучении и калибровке.

## Участник 3 — интерфейс и демонстрация

Ветка: `feat/dashboard`.

Сначала: интерфейс на согласованном fixture ForecastResult. Затем подключение к service.py, происхождение данных, журнал, сравнение выпусков и скачивание готового экспорта.

Критерий: жюри проходит сквозной сценарий через UI; ошибки и отсутствие данных показаны явно. README и DEMO.md соответствуют фактическому состоянию.

## Общие правила

- Перед стартом 15–20 минут согласовать docs/CONTRACTS.md. schemas.py меняет только участник 1.
- Каждый работает в отдельном клоне или worktree. Три сессии Codex не запускаются в одной рабочей папке.
- Общие зависимости, service.py, конфигурацию и структуру проекта меняет участник 1. Остальные передают ему необходимые изменения.
- UI не выполняет расчёты, ML не скачивает погоду, weather не обучает модели.
- Первый сквозной запуск нужен рано, до добавления сложных функций.
- Добавляя файл, указать владельца. Чужие файлы менять после согласования.
- Папки data/raw, data/weather, data/processed, artifacts, reports и outputs предназначены для локальных данных; они исключены из Git. Предоставленные CSV вручную разместить как data/raw/turbine_1.csv и data/raw/turbine_2.csv.
- configs/site.yaml включён в ZIP как пустой локальный файл, но исключён из Git. В клонах создавать его из site.example.yaml после заполнения шаблона.
- pyproject.toml и requirements.lock заполнены; эталонная установка проверена на Windows/Python 3.12.
- demo/manifest.json описывает синтетическую проверку интеграции; реальные погодные данные загружаются отдельно.
- docs/CONTRACTS.md описывает реализованный API. README и DEMO.md обновляет участник 3 по фактическим результатам.

## Карта владения файлами

Дополнительные владельцы: README.md — участник 3; TEAM.md, AGENTS.md, .gitignore, docs/CONTRACTS.md — участник 1 после согласования с командой.

### Файлы участника 1

- `.github/workflows/ci.yml` — установка, все тесты команды и offline-интеграция на Windows/Linux.
- `pyproject.toml`
- `requirements.lock`
- `.env.example`
- `configs/site.example.yaml`
- `configs/site.yaml`
- `src/windoracle/__init__.py`
- `src/windoracle/__main__.py`
- `src/windoracle/cli.py`
- `src/windoracle/config.py`
- `src/windoracle/schemas.py`
- `src/windoracle/clock.py`
- `src/windoracle/store.py`
- `src/windoracle/ingest.py`
- `src/windoracle/service.py`
- `src/windoracle/replay.py`
- `src/windoracle/export.py`
- `src/windoracle/weather/__init__.py`
- `src/windoracle/weather/base.py`
- `src/windoracle/weather/archive.py`
- `src/windoracle/weather/cache.py`
- `src/windoracle/weather/audit.py`
- `src/windoracle/agents/__init__.py`
- `src/windoracle/agents/orchestrator.py`
- `src/windoracle/agents/weather_archivist.py`
- `src/windoracle/agents/memory.py`
- `tests/__init__.py`
- `tests/conftest.py`
- `tests/test_ingest.py`
- `tests/test_clock.py`
- `tests/test_weather.py`
- `tests/test_temporal_contract.py`
- `tests/test_replay.py`
- `tests/test_service.py`
- `tests/test_export.py`
- `scripts/prepare_demo.py`
- `demo/manifest.json`
- `docs/DECISIONS.md`

### Файлы участника 2

- `src/windoracle/features.py`
- `src/windoracle/evaluate.py`
- `src/windoracle/models/__init__.py`
- `src/windoracle/models/baseline.py`
- `src/windoracle/models/power_curve.py`
- `src/windoracle/models/ml.py`
- `src/windoracle/models/ensemble.py`
- `src/windoracle/models/bias.py`
- `src/windoracle/models/intervals.py`
- `src/windoracle/models/registry.py`
- `src/windoracle/agents/twin_builder.py`
- `src/windoracle/agents/forecaster.py`
- `src/windoracle/agents/critic.py`
- `tests/test_models.py`
- `tests/test_bias.py`
- `tests/test_intervals.py`
- `tests/test_metrics.py`

### Файлы участника 3

- `app.py`
- `ui/__init__.py`
- `ui/dashboard.py`
- `ui/charts.py`
- `ui/controls.py`
- `ui/provenance.py`
- `ui/events.py`
- `src/windoracle/agents/advisor.py`
- `tests/test_ui_contract.py`
- `docs/DEMO.md`

## Служебные директории

| Папка | Назначение | Владелец |
| --- | --- | --- |
| data/raw | Неизменяемые CSV | 1 |
| data/weather | Архив и метаданные погоды | 1 |
| data/processed | Подготовленные наблюдения | 1 |
| artifacts/models | Версии обученных моделей | 2; хранение согласовано с 1 |
| artifacts/bias | Версии коррекции | 2; запись через слой 1 |
| reports | Отчёты проверки качества | 2 |
| outputs | Выпуски и экспорт | 1 |
| tests/fixtures | Согласованные тестовые данные | 1; расширения согласовывать |
| demo/fixtures | Примеры результатов для UI | 3; структура согласована с 1 |
| ui/assets | Ресурсы интерфейса | 3 |

Первые шаги: назначить роли → согласовать контракты → добавить каркас в репозиторий команды → создать отдельные ветки/worktree → начать реализацию.

## Дополнение участника 3 — TwinTurbo.ai UI

Добавлены и принадлежат участнику 3: `frontend/` целиком (ресурсы, контроллер, адаптер, проверки представления, preview, зависимости UI и синтетические fixtures), `docs/UI_INTEGRATION.md`, `docs/UI_VERIFICATION.md`, `docs/PROJECT_SPEC.md`. Существующие `app.py`, `ui/`, `tests/test_ui_contract.py`, `README.md`, `docs/DEMO.md` также остаются у участника 3.

Advisor расположен в `src/TwinTurbo.ai/agents/advisor.py` после переименования исходников. Имя Python-пакета `windoracle` сохранено в общей упаковке. Сервис, схемы, модели и общий lock не изменялись интерфейсной реализацией. Синтетические данные для демо разрешены пользователем; они маркируются и не допускаются в конкурсный экспорт.
