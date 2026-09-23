# TwinTurbo.ai — проверка модуля прогнозирования

Владелец: участник 2. Подробный API: [инструкция модуля](../../src/TwinTurbo.ai/models/README.md).

## Проверка кода

- 56 тестов: 22 исходные интеграционные и 34 проверки моделей, метрик и временной корректности.
- Проверены editable-установка, сборка wheel и импорт из распакованного wheel с изолированным Python.
- Реализованы отдельные кривые турбин, baseline, bias, интервалы, Critic, необязательный ML и фиксируемый ансамбль.
- Модель получает готовые snapshots; не читает CSV, не скачивает погоду, не выбирает run и не хранит выпуски.

```sh
python -m pip install --no-deps -e .
python -m pytest -q
```

## Синтетическая проверка

[fixture-comparison.json](fixture-comparison.json) — только синтетические данные,
генератор `tests.test_models.prepared_experiment`, seed=42. Две последовательные
фазы обучения, 14 origins, горизонт 48 часов, 1344 общих прогнозных строки.
Все данные явно synthetic; strict-экспорт fixture запрещён общим сервисом.

```sh
python -m tests.test_metrics --output reports/forecast-models/fixture-comparison.json
```

В fixture MAE у curve_bias ниже curve и baseline, но покрытие Q10–Q90 ниже целевых
80%. Не заявляется гарантированная вероятность или качество реальных турбин.
Число доступных интервалов, ширина и pinball losses приведены в JSON.

## Повторить на подготовленных реальных данных

Подробные результаты и обученный артефакт, полученные из приватных пользовательских
данных, не включены в эту ветку. Они подготовлены отдельно для владельца данных.
В Git остаётся код воспроизведения. Перед распространением результатов нужен
разрешённый владельцем набор данных и назначение публикации.

Команды подготовки ниже принадлежат участнику 1. Исходные CSV пользователь
размещает локально; погодный run скачивается существующим адаптером.

```sh
python -m TwinTurbo.ai ingest --turbine-1 data/raw/turbine_1.csv --turbine-2 data/raw/turbine_2.csv
python -m TwinTurbo.ai weather fetch --origin 2026-01-27T18:00:00Z --run 2026-01-27T12:00:00Z
python -m tests.test_metrics --real-cache --horizon 24 --origins 2026-01-27T18:00:00Z 2026-01-28T00:00:00Z 2026-01-28T06:00:00Z --output reports/forecast-models/real-cache-comparison.json
```

Эксперимент читает только готовые Store/weather cache. Он не скачивает погоду
и не импортирует исходные CSV. Обучение фиксировано на первом origin; ошибка
попадает в bias только после available_at факта. Для полной валидации нужен
существенно более длинный период и подтверждённое исходное время SCADA.

Большие CSV, SQLite, погодный кэш и локальные результаты исключены из Git.
Отдельный CLI оценки принимает готовый JSON из слоя участника 1:

```sh
python -m TwinTurbo.ai.evaluate --input prepared-evaluation.json --output reports/comparison.json
```

## Передача интегратору

- Сохранить base_predictions до bias/clipping вместе с forecast_id для Critic.
  В общем ForecastResult отдельного p_base пока нет; поддержан base_batches по ID.
- Подключить вызов Critic и сохранение BiasState в общий агентный цикл.
  Компонент предлагает состояние, но не меняет общий Store/Orchestrator.
- Для ML согласовать scikit-learn в общих зависимостях. Проверено с 1.8.0;
  основной P0 не требует sklearn.
- UI должен показывать фактическое качество, warmup интервалов и предупреждения
  временных допущений, переданные участником 1.
