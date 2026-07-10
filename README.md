# ru-vlm-adaptation

Адаптация англоязычной vision-language модели `Intel/llava-gemma-2b` к русскоязычным бенчмаркам на открытых данных deepvk.

## Цель

Получить русскоязычную VLM на базе `Intel/llava-gemma-2b` методом LoRA fine-tuning на датасетах VK и сравнить качество с исходной моделью Intel и `deepvk/llava-gemma-2b-lora` на `GQA-ru` и `MMBench-ru`.

## Задачи

1. Снять базовые метрики `Intel/llava-gemma-2b` на `GQA-ru` и `MMBench-ru`.
2. Подготовить обучающую выборку из `GQA-ru` (train) и `LLaVA-Instruct-ru` с изображениями COCO 2017.
3. Обучить русскоязычную VLM на базе `Intel/llava-gemma-2b`.
4. Оценить адаптированную модель в тех же условиях, что и базовую модель.
5. Аналогично получить метрики `deepvk/llava-gemma-2b-lora`.
6. Зафиксировать метрики, артефакты обучения и выводы.

## Ожидаемые результаты

- Рост качества на `GQA-ru` относительно Intel baseline (модель почти не отвечает по-русски «из коробки»).
- Сопоставимый порядок величины с референсом VK на `GQA-ru` при одинаковой методике оценки.
- На `MMBench-ru` — сохранение или умеренное изменение accuracy.
- Воспроизводимый пайплайн: подготовка данных → LoRA train → evaluate → сравнение трёх моделей.

## Отчёты

 - [result.md](result.md)    – Описание обученной модели и итоговых метрик 
 - [steps.md](steps.md) – Подробный процесс решения

## Использованные данные

### Модели

| Модель | Источник | Как использовалась                    |
|--------|----------|---------------------------------------|
| `Intel/llava-gemma-2b` | [Hugging Face](https://huggingface.co/Intel/llava-gemma-2b) | База для русскоязычной адаптации      |
| `deepvk/llava-gemma-2b-lora` | [Hugging Face](https://huggingface.co/deepvk/llava-gemma-2b-lora) | Референс VK для сравнения показателей |


### Датасеты deepvk

| Датасет | Split / часть | Как использовался |
|---------|---------------|-------------------|
| [`deepvk/GQA-ru`](https://huggingface.co/datasets/deepvk/GQA-ru) | `train_balanced_*` | Обучение: 8000 примеров, post-prompt «Ответь одним словом.» |
| [`deepvk/GQA-ru`](https://huggingface.co/datasets/deepvk/GQA-ru) | `testdev_balanced_*` | Оценка GQA-ru (12216 примеров) |
| [`deepvk/LLaVA-Instruct-ru`](https://huggingface.co/datasets/deepvk/LLaVA-Instruct-ru) | `train` | Обучение: 10000 диалогов; изображения из COCO 2017 |
| [`deepvk/MMBench-ru`](https://huggingface.co/datasets/deepvk/MMBench-ru) | `dev` | Оценка MMBench-ru (3910 примеров) |

### Внешние данные

| Данные | Как использовались |
|--------|-------------------|
| COCO 2017 (`train2017`) | Подстановка изображений по путям из `LLaVA-Instruct-ru` (`coco/train2017/...`) |


