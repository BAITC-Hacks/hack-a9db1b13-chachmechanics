# TwinTurbo.ai — модуль участника №2

Владелец: участник 2. Прогноз, признаки, bias, интервалы, метрики и Critic.
Импорты и команда проекта теперь `TwinTurbo.ai`, исходники — `src/TwinTurbo.ai/`.
Переименование сделано по указанию пользователя отдельным коммитом. После получения
ветки повторить `python -m pip install --no-deps -e .` в своей рабочей среде.

## Что реализовано

- `PowerCurvePredictor`: отдельная медианная кривая каждой турбины, бины 0.5 м/с,
  минимум 6 полных часов в бине; линейная интерполяция между опорными точками.
  Снаружи освоенного диапазона — постоянное продолжение ближайшего конца и
  `OUT_OF_DOMAIN`. Не выдумывает cut-out, монотонность, номинал и плотность воздуха.
- `ConstantBaseline`: средняя мощность train отдельно по турбинам.
  `PersistenceBaseline` дополнительно требует полный факт не старше 1 часа.
- Признаки: прогнозные ветер/температура, компоненты направления, lead до начала
  целевого часа, погодный lead, возраст run, циклический час/день года.
  По умолчанию календарь UTC, у ML можно явно задать `calendar_timezone`.
  Историческая фактическая погода целевого часа в ML-признаки не попадает.
- Bias: `actual - p_base`, окно 21 день, группы 1–6/7–12/13–24/25–48 часов,
  стягивание к средней ошибке турбины с lambda=48. После смены модели прогрев
  начинается отдельно. Повторные входы идемпотентны; новый факт новой версии
  заменяет предыдущий только в новом состоянии. Просроченное окно сбрасывается.
- Q10/Q50/Q90: эмпирические квантили ошибок уже выданных прогнозов, после bias.
  Минимум 30 зрелых ошибок в группе. Более общий пул отключён по умолчанию;
  включать `interval_fallback=True` только после отдельной временной проверки.
  Нет истории — три null. Точечный прогноз и Q50 могут различаться из-за
  медианы остаточной ошибки; Q50 не математическое ожидание.
- Необязательный HistGradientBoostingRegressor: loss=absolute_error, seed=42,
  `early_stopping=False` (нет случайного validation split). Обучение отдельно
  по турбинам, только на архивных прогнозных snapshots с уже поступившими целями.
- Ансамбль: веса {0, .25, .5, .75, 1} выбираются по MAE на более позднем,
  уже доступном validation, затем замораживаются по группам lead. Нет validation
  для группы — используется кривая. Новая версия активируется после validation.
- JSON-сериализация P0 с checksum и запретом перезаписи другой версии.
  Для собственных ML-артефактов — явный `trusted=True`, версия sklearn проверяется.
- Critic возвращает численные метрики, состояние и причины обновления/пропуска;
  сигнал дрейфа — основание проверить модель, без автоматического переобучения.

Параметры выше — фиксированные стартовые настройки из спецификации, а не результат
оптимизации на контрольном периоде. Не утверждаем, что кривая лучше baseline на
реальных данных. Сравнение и происхождение входов находятся в отчётах.

## Подключение к сервису участника №1

Все входы получает интегратор. Модель не открывает CSV, не скачивает погоду,
не выбирает run и не управляет виртуальными часами. Проверки временных инвариантов
на границе модели защищают от ошибочного вызова, но не заменяют as-of слой.

```python
from TwinTurbo.ai.models.power_curve import PowerCurvePredictor
from TwinTurbo.ai.models.registry import save_predictor, load_predictor

# training_snapshot уже подготовлен participant 1 на конкретный cutoff.
predictor = PowerCurvePredictor.fit(training_snapshot, activated_at=activation_time)
save_predictor(predictor, "artifacts/models/model.json")
# ForecastService(config, store, weather, predictor)
batch = predictor.predict(future_snapshot, bias=None)
```

`predictor.state` — общая ModelState; `predict` возвращает общую PredictionBatch.
96 строк для двух турбин и 48 часов. Без допустимой истории — явная ошибка,
без случайной модели или нулевых заполнителей. Режим synthetic наследуется от
fixture; общий сервис запрещает такую модель для replay/submission.
Обучающие снимки должны быть правильно маркированы интегратором: в Observation
нет собственного поля происхождения, поэтому метаданные snapshot задают этот режим.

Фабрика CLI без аргументов читает `TWINTURBO_AI_MODEL_PATH`, по умолчанию
`artifacts/models/model.json`. Сохранённая версия должна быть допустима в origin.

```sh
python -m TwinTurbo.ai predict --origin 2026-01-29T18:00:00Z --predictor TwinTurbo.ai.models.registry:load_predictor
```

`artifact_ref=model://<model_id>` — логический адрес неизменяемой версии; фактический
путь выбирает вызывающий код и указывает через переменную окружения. Сохранение
метаданных в SQLite остаётся `store.save_model` у участника 1.

## Critic: сохранить базовый прогноз до clipping

```python
from TwinTurbo.ai.agents.forecaster import Forecaster
from TwinTurbo.ai.agents.critic import Critic

trace = Forecaster(predictor).predict_with_trace(snapshot, bias)
# Участник 1 сохраняет trace.base_predictions вместе с forecast_id и выданным выпуском.
decision = Critic().review(
    saved_forecasts, available_observations,
    model_id=predictor.state.model_id, as_of=current_origin,
    base_batches=saved_base_batches_by_forecast_id, previous=bias,
)
# Интегратор решает, когда сохранить decision.proposed_bias через store.save_bias.
```

**Точка интеграции:** нынешний ForecastResult не содержит отдельное поле `p_base`.
Участнику 1 нужно сохранять `trace.base_predictions` (например, в manifest либо
своём sidecar) и передавать его в Critic. Наш код не меняет схемы и Store.
Если `bias_id` задан, а базового снимка нет, Critic возвращает ошибку
`BASE_PREDICTIONS_REQUIRED`. Вычитать bias из уже clipped-прогноза математически
неверно. Для выпусков без bias сами predictions являются исходным прогнозом.

Для калибровки используется последний **scheduled** origin для каждого целевого
часа, турбины и группы lead. `update` не увеличивает вес факта. Математика живёт
в models; агент лишь вызывает её и выдаёт решение. Автоматическое подключение
Critic в общий replay/Orchestrator остаётся у интегратора.

## Последовательная проверка

`TemporalFold(training, validation)` принимает готовые AsOfSnapshot.
`walk_forward(folds, observations, as_of=...)` обучает модели на каждом cutoff,
последовательно проходит origins и обновляет bias только по уже поступившим
ошибкам. Итоговые observations используются только для оценки и созревшей
коррекции; не передаются в обучение вместо training snapshot.

По умолчанию сравниваются baseline, curve и curve_bias. Для ML передать свою
фабрику в `fitters`, вызывающую `MLPredictor.fit(training_snapshot,
historical_snapshots, ...)`. Для готового ансамбля можно передать фабрику,
возвращающую проверенную ранее версию. Эти функции не управляют выбором погоды.

Одинаковая общая выборка `(origin, turbine, target_start, target_end)` для всех
моделей; дополнительно показаны доли выпущенных часов и полных origins. Неполные
и ещё не поступившие факты исключаются с подсчётом, не превращаются в нули.
Отчёт содержит MAE, RMSE, signed error (`prediction - actual`), pinball losses,
покрытие и ширину интервала, число значений и долю доступных интервалов.
Повтор одного фактического часа у разных origins допустим и явно учитывается
по выпускам. Статистическая значимость улучшения не заявляется.

Отсутствующие погодные snapshots обрабатывает интегратор; для оценки полноты
всего запланированного периода передать `expected_keys` в `compare_forecasts`.
Без него знаменатель — только объединение предоставленных выпусков.

Существует отдельная проверенная команда модуля, не добавленная в общий CLI:

```sh
python -m TwinTurbo.ai.evaluate --input prepared-evaluation.json --output reports/comparison.json
```

Формат JSON: `{"folds": [{"training": <AsOfSnapshot JSON>, "validation": [<AsOfSnapshot JSON>]}],
"observations": [<Observation JSON>], "as_of": "2026-02-01T00:15:00Z"}`.
Формировать JSON через `model_dump(mode="json")` в интеграционном слое.

## Установка и воспроизведение

P0 использует только зависимости существующего pyproject. Для необязательного ML
нужен scikit-learn (проверено 1.8.0); его добавление в общий lock — задача участника 1.
Проверенная среда этого изменения: Python 3.12, NumPy 2.3.5, pandas 2.2.3,
Pydantic 2.13.5, pytest 9.1.1. Общий Windows lock не изменялся.

```sh
python -m pip install --no-deps -e .
python -m pytest -q
python -m tests.test_metrics --output reports/forecast-models/fixture-comparison.json
```

Fixture — только синтетическая проверка: seed=42, две последовательные фазы,
14 origins, 1344 общих пары прогноза/факта. Интервалы тестируются на будущих
относительно их калибровки данных; целевые 80% не гарантируются.
Реальный cache-эксперимент воспроизводится отдельно, см. отчёт в
`reports/forecast-models/README.md`. Нельзя выдавать fixture-метрики за реальное качество.
