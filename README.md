# ru-vlm-adaptation

Адаптация `Intel/llava-gemma-2b` к русскоязычным бенчмаркам.

## Цель

Обучить русскоязычную VLM на открытых данных [deepvk](https://huggingface.co/collections/deepvk/vision-language-modeling) и сравнить с `deepvk/llava-gemma-2b-lora`.

## Задачи

1. Baseline `Intel/llava-gemma-2b` на `GQA-ru` и `MMBench-ru`
2. Подготовка train из `GQA-ru` и `LLaVA-Instruct-ru`
3. LoRA fine-tune
4. Повторная оценка
5. Сравнение с референсом VK

## Данные VK

| Датасет | Назначение |
|---------|------------|
| `GQA-ru` train | обучение |
| `LLaVA-Instruct-ru` | обучение, нужен COCO |
| `GQA-ru` testdev | оценка |
| `MMBench-ru` | оценка |

## Модели

| Модель | Роль |
|--------|------|
| `Intel/llava-gemma-2b` | база для обучения |
| `deepvk/llava-gemma-2b-lora` | референс VK |


На Kaggle: GPU T4, Internet On.
