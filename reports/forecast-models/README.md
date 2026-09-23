# Исторический отчёт модельной ветки

`fixture-comparison.json` сохранён без изменений из commit 7f054b3 ветки feat/forecast-models. Это синтетический эксперимент исходной реализации, не метрики окончательно объединённого кода и не оценка реальной ВЭС.

Актуальный API и отличия слияния: [модели](../../src/TwinTurbo.ai/models/README.md), [проверка релиза](../../docs/RELEASE_CHECK.md). Итоговый тестовый набор запускается `python -m pytest -q`.

Для подготовленных snapshots используется `python -m windoracle.evaluate --input prepared.json --output reports/comparison.json`. Исходные CSV, погодный кэш и обученные артефакты остаются локальными.
