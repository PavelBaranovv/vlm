# Ход работы

## 1. Постановка задачи

Получить русскоязычную VLM на базе `Intel/llava-gemma-2b` методом LoRA fine-tuning на датасетах VK и сравнить качество с исходной моделью Intel и `deepvk/llava-gemma-2b-lora` на `GQA-ru` и `MMBench-ru`.


Работа велась на **Kaggle (T4 x 2 accelerator)**. 

## 2. Baseline Intel

Для получения метрик написан `scripts/evaluate.py`.

Протокол:

- **MMBench-ru:** промпт с вариантами A–D, ответ — буква; парсинг первой буквы ответа.
- **GQA-ru:** вопрос + «Ответь одним словом.», сравнение первого слова prediction/gold.

Результат Intel:

| Benchmark | Accuracy |
|-----------|----------|
| GQA-ru | 0.21% |
| MMBench-ru | 59.44% |

Вывод: на русском GQA модель фактически не работает; на MMBench визуальная часть показывает относительно высокий процент даже без русской адаптации.

## 3. Подготовка данных

Скрипт: `scripts/prepare_data.py`, режим `medium`.

| Источник | Объём | Источник изображений                           |
|----------|-------|------------------------------------------------|
| GQA-ru train | 8000 | Картинки из HF-сабсета `train_balanced_images` |
| LLaVA-Instruct-ru | 10000 | COCO 2017; пути вида `coco/train2017/*.jpg`    |



Итог: 18000 примеров.

## 4. Обучение

Скрипт: `scripts/train.py`

Параметры обучения: `configs/train_config.yaml`.

Стек: `SFTTrainer` (TRL) + LoRA.

Обучение: ~6 часов на T4x2. Loss 3.87 → 1.34.

Адаптер сохранён в `checkpoints/intel-llava-gemma-2b-ru/`.

## 5. Создание модели

Скрипт: `scripts/merge_lora.py`.

LoRA-адаптер смерживается с `Intel/llava-gemma-2b`, получается самостоятельная модель `models/intel-llava-gemma-2b-ru`.
(В репозитории её нет, при необходимости собирается локально):

```bash
python scripts/merge_lora.py
```

Оценку можно запускать как по адаптеру (`checkpoints/...`), так и по целостной модели.

## 6. Оценка адаптированной модели

Тем же `evaluate.py`:

| Benchmark | Accuracy |
|-----------|----------|
| GQA-ru | 44.25% |
| MMBench-ru | 58.70% |


GQA — **главный целевой эффект достигнут: 0.21% → 44.25%**.

## 7. Референс VK тем же скриптом

`deepvk/llava-gemma-2b-lora` прогнан через тесты, получено:

| Benchmark | Accuracy |
|-----------|----------|
| GQA-ru | 48.17% |
| MMBench-ru | 63.81% |

## Воспроизвести получение метрик модели

```bash
# адаптер
python scripts/evaluate.py \
  --model checkpoints/intel-llava-gemma-2b-ru \
  --output-dir "results/Intel llava-gemma-2b ru-adapted"

# или полная модель
python scripts/merge_lora.py
python scripts/evaluate.py \
  --model models/intel-llava-gemma-2b-ru \
  --output-dir "results/Intel llava-gemma-2b ru-adapted"
```


