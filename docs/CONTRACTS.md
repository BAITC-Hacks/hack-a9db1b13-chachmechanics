# Контракты между участниками

Статус: интерфейсы участника 1 реализованы и проверены. Исходники находятся в
`src/TwinTurbo.ai/`, Python-импорты и CLI сохраняют имя `windoracle` через настройку setuptools.
Владелец схем — участник 1. Участники 2 и 3 согласуют необходимые поля до начала интеграции.

## Путь данных

Участник 1 готовит AsOfSnapshot → участник 2 возвращает PredictionBatch → участник 1 сохраняет ForecastResult → участник 3 отображает его через service.py.

## Минимальные структуры

| Контракт | Поля и смысл |
| --- | --- |
| ForecastRequest | origin_time, turbine_ids, horizon_hours, mode; момент решения, список турбин, 24/48 часов, режим |
| AsOfSnapshot | origin_time, observations, weather_values, weather_run_metadata, target_intervals, quality_flags; только входы, допустимые к origin_time |
| ModelState | model_id, training_cutoff, activated_at, artifact_ref, feature_schema_version; версия и допустимость модели |
| BiasState | bias_id, model_id, created_as_of, last_actual_available_at, parameters; состояние коррекции |
| PredictionBatch | строки turbine_id, target_start/end, prediction_norm, nullable q10/q50/q90, status; результаты модели |
| ForecastResult | forecast_id, origin_time, PredictionBatch, run_id, model_id, bias_id, parent_forecast_id, warnings, provenance |
| EvaluationReport | period, metrics_by_turbine_and_lead, sample_count, forecast_coverage, interval_metrics; реальные измеренные показатели |

Времена внутри системы — timezone-aware UTC. Все мощности по умолчанию нормализованы. Отсутствующее значение — null, не 0. Сырые измерения, прогнозы и тестовые fixtures различаются явно.

## Границы функций

- `service.py`: единая точка для CLI и UI; создать выпуск, получить сохранённый выпуск, сравнить выпуски, прочитать журнал, экспортировать.
- `weather/base.py`: интерфейс провайдера; `archive.py`: один выбранный рабочий адаптер; `cache.py`: сохранение; `audit.py`: проверка происхождения и покрытия.
- `features.py`: преобразование уже разрешённого снимка в признаки. Не читает исходные CSV и не обращается к погодным API.
- `models/`: обучение, прогноз и математические обновления. Не читает интерфейс и не управляет виртуальными часами.
- `models/registry.py`: описание/сериализация моделей. Проверку разрешённого времени активации выполняет слой участника 1.
- `agents/critic.py`: вычисляет оценку и предложение изменения состояния. Факт сохранения версии контролирует общий store.
- `ui/`: отображает ForecastResult и вызывает service. Не рассчитывает собственную мощность, метрики и bias.

Точные сигнатуры и типы зафиксированы в `src/TwinTurbo.ai/schemas.py`.
Следующие примеры являются текущим API, а не командами из первоначальной спецификации README.

### Подключение модели участника 2

Объект predictor содержит `state: ModelState` и метод
`predict(snapshot: AsOfSnapshot, bias: BiasState | None = None) -> PredictionBatch`.
`PredictionBatch(rows=(PredictionRow(...), ...))` содержит по одной строке на
каждую пару турбина/целевой час. `ModelState` обязательно включает
`max_label_available_at <= training_cutoff <= activated_at`, а также `artifact_ref`.
В рабочей модели `provenance="trained"`, в тестовой — `"synthetic"`.
`ModelState` описывает метаданные; сохранение/загрузка весов остаётся у участника 2.

`snapshot.observations` — tuple из Observation; `snapshot.weather_values` — tuple
из WeatherValue, уже отобранных на целевые часы. Доступ к полям через атрибуты,
таблицы pandas при необходимости создаются внутри features из `model_dump()`.
Неполные часы имеют `power_norm=None` и quality_flag, их нельзя обучать как нули.

Для CLI участник 2 предоставляет фабрику без аргументов, например
`windoracle.models.registry:load_predictor`. Она возвращает predictor с уже
загруженным состоянием. Фабрика реализована и читает путь JSON из `TWINTURBO_MODEL_ARTIFACT` (также поддерживается `TWINTURBO_AI_MODEL_PATH`).
Передать её явно: `--predictor windoracle.models.registry:load_predictor`.
Ни CLI, ни интегратор не подменяют отсутствие модели случайными числами.

### Подключение интерфейса участника 3

```python
from windoracle.config import load_config
from windoracle.store import Store
from windoracle.weather.archive import GFSArchive
from windoracle.service import ForecastService
from windoracle.schemas import ForecastRequest

config = load_config("configs/site.example.yaml")
store = Store(config.storage.database)
service = ForecastService(config, store, GFSArchive(config))
# Без predictor доступны чтение, сравнение, экспорт, журнал и описание данных.
results = service.list_forecasts()  # list[ForecastResult]
events = service.events()          # list[dict]
summary = service.data_summary()
```

Для расчёта передать predictor четвёртым аргументом ForecastService. Вызов:
`service.create_forecast(ForecastRequest(origin_time=..., turbine_ids=(...), mode="replay"))`.
UI передаёт aware datetime или ISO 8601 с offset, не строку без пояса.
`result.predictions.rows` — tuple PredictionRow; `result.manifest` содержит
происхождение данных, модель, конфигурацию и предупреждения.
`service.compare_forecasts(id1, id2)` сравнивает только пересекающиеся целевые часы.
`service.export([id1, id2])` возвращает текст CSV. Для fixture необходимо явно
`strict=False`; такой файл нельзя выдавать за конкурсный результат.

### Хранение и время

`store.observations_as_of(origin, turbine_ids)` выбирает только доступные записи.
`store.save_model(state)` / `models_as_of(origin)` сохраняют и выбирают версии модели.
`store.save_bias(state)` / `bias_as_of(model_id, origin)` делают то же для коррекции.
Существующая версия не может менять содержимое. Повтор идентичной записи безопасен.
`service.create_forecast(..., bias=bias)` проверяет версию и время bias.
Расчёт новых коэффициентов и обработка факта остаются у участника 2; автоматического
обучения без его компонента нет.

`replay(service, origins, mode="replay", include_updates=True)` возвращает
`{"forecast_ids": [...], "failures": [...]}`. `origins` — реальные aware datetime,
не индексы строк. VirtualClock не движется назад. При новых runs сохраняются новые
выпуски со ссылкой parent_forecast_id; старые прогнозы остаются неизменными.
Сохранённые JSON + index.json с checksum можно проверять и экспортировать на другой
машине без исходной SQLite через `verify --input` и `export --input`.

## Минимальные инварианты

1. `available_at <= origin_time` для каждого использованного входа.
2. У модели допустимы training_cutoff и activated_at; цели обучения уже поступили к cutoff.
3. Один штатный выпуск на 48 часов для двух турбин содержит 96 прогнозных строк; ошибка/неполное покрытие сообщается явно.
4. Новый погодный run создаёт новый выпуск; старый не изменяется.
5. Повтор одного запроса не удваивает прогноз и обновление bias.
6. Нет факта — нет новой коррекции по факту.
7. Нет интервала — nullable-квантили, а не три копии точечного прогноза.
8. Fixture не допускается в конкурсный экспорт.

## Как начинать независимо

`tests/conftest.py` содержит синтетические входы и тестовую модель.
`python scripts/prepare_demo.py --fixture --origin 2025-06-01T18:00:00Z` создаёт
три выпуска по 96 строк, index.json, CSV и журнал в outputs/integration-smoke.
Они подходят для подключения интерфейса, явно маркированы fixture и запрещены в strict-экспорте.
`--real-weather` использует уже скачанную настоящую погоду, но модель по-прежнему
тестовая: это проверка интеграции, а не оценка качества прогноза.

## Дополнения финальной интеграции

`train --origin ... --output ...` обучает кривую или baseline через TwinBuilder на допустимом снимке. `get_display_context(id, as_of=...)` отдаёт замороженную почасовую погоду из manifest и доступные факты. Время просмотра может продвигаться независимо от origin; старый прогноз не меняется.

Новые выпуски сохраняют `manifest.base_predictions` до bias/clipping. `Critic.review` и `residuals_from_forecasts` читают его автоматически; для старого скорректированного выпуска требуется явно переданный base_batches. `Forecaster.predict_with_trace` сохраняет API модельной ветки.

`windoracle.backtest.walk_forward_snapshots` выполняет сравнение подготовленных временных folds. Общий `evaluate.walk_forward` остаётся функцией для произвольных последовательных проверок. CLI подготовленного эксперимента: `python -m windoracle.evaluate --input prepared.json --output report.json`.
