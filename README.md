# Vision-Language Model

Дообучение русскоязычной VLM на открытых данных [deepvk](https://huggingface.co/collections/deepvk/vision-language-modeling-664dd7e4c257cc78e740f6bc) с оценкой на бенчмарках **GQA-ru** и **MMBench-ru**.

## Цель

Получить дообученную модель на базе `deepvk/llava-gemma-2b-lora`, которая показывает **прирост метрик** относительно исходной версии на русскоязычных бенчмарках.

## Задачи

1. Изучить архитектуру VLM и открытые ресурсы VK.
2. Подготовить данные для обучения (`LLaVA-Instruct-ru`, `GQA-ru`).
3. Дообучить модель методом LoRA.
4. Оценить качество через `lmms-eval` на GQA-ru и MMBench-ru.
5. Сравнить результаты с baseline и оформить отчёт.

## Используемые данные VK

| Ресурс | Роль в проекте |
|--------|----------------|
| `deepvk/llava-gemma-2b-lora` | Базовая модель для дообучения |
| `deepvk/llava-saiga-8b` | Референс для сравнения (не обучаем) |
| `deepvk/LLaVA-Instruct-ru` | Обучение: диалоги по изображениям |
| `deepvk/GQA-ru` | Обучение (train) + оценка (test) |
| `deepvk/MMBench-ru` | Оценка |


## Baseline

Сначала фиксируем качество исходной модели `deepvk/llava-gemma-2b-lora` без дополнительного обучения.

- Подробная инструкция: `docs/baseline_kaggle.md`
- Ожидаемые метрики:
  - `GQA-ru`: около `46.37`
  - `MMBench-ru`: около `40.19`

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```