# Модели финальной версии TwinTurbo.ai

Одна реализация доступна через `windoracle.models` и `TwinTurbo.ai.models`.

- `TwinBuilder().build(snapshot)` — медианная кривая отдельно для турбин; параметры `bin_width` и `min_samples_per_bin` настраиваются. Совместимые defaults main: 1 м/с и 1 наблюдение на бин. Вне освоенного диапазона — крайнее значение и out_of_domain.
- `TwinBuilder().build_baseline(snapshot)` — средняя мощность train каждой турбины.
- `PersistenceBaseline(state, means)` — последний полный час, не старше часа; при устаревании явный отказ.
- `fit_ridge_predictor(...)` / `RidgePredictor` — модель NumPy из main. Существующий псевдоним MLPredictor оставлен для совместимости main.
- `models.boosting.GradientBoostingPredictor.fit(snapshot, historical_snapshots)` — HistGradientBoosting из модельной ветки: absolute_error, seed=42, без случайного early stopping. Требует архивные прогнозные признаки и уже доступные цели. Общая схема признаков теперь едина с ridge.
- Ансамбль выбирает вес по группам горизонта на отложенном доступном validation.
- Bias и интервалы используют зрелые ошибки сохранённых scheduled-выпусков; повтор не увеличивает вес факта. Нехватка истории оставляет квантили null.

```python
from windoracle.agents.twin_builder import TwinBuilder
from windoracle.models.registry import save_predictor, load_predictor
model = TwinBuilder().build(training_snapshot)
save_predictor(model, "artifacts/models/model.json")
model = load_predictor("artifacts/models/model.json")
```

Фабрика CLI читает `TWINTURBO_MODEL_ARTIFACT` или `TWINTURBO_AI_MODEL_PATH`. Отсутствие модели — ошибка. Артефакт неизменяемый, с checksum. Сериализация sklearn требует `trusted=True` при загрузке собственного артефакта и совпадения версии sklearn. Для доверенной локальной CLI-загрузки: `TWINTURBO_AI_TRUSTED_MODEL=1`.

Форматы артефактов из двух исходных веток различались. Финальный формат — из main; старую модель из feat/forecast-models нужно заново сохранить/обучить текущим API. Градиентный бустинг получил отдельное имя, чтобы не подменять существующий ridge.

Сервис сохраняет `manifest.base_predictions` для последующего Critic. Обратное восстановление базы из clipped-прогноза запрещено. `Forecaster.predict_with_trace` тоже доступен.

Последовательная оценка: `windoracle.backtest.TemporalFold` и `walk_forward_snapshots`. CLI `python -m windoracle.evaluate --input prepared.json --output report.json`; формат JSON: folds с training/validation AsOfSnapshot, observations, as_of. Для каждого fold модель фиксирована, bias видит только созревшие прошлые ошибки. Сравнение выполняется на общих ключах, доля выпущенных строк показана отдельно.

Прогноз по сеточному ветру и кривой измеренного ветра ещё требует длительной временной проверки. Успешный интеграционный запуск не доказывает превосходство над baseline.
