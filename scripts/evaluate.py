from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from datasets import load_dataset
from tqdm import tqdm

from model_utils import load_model


def extract_choice(text: str) -> str:
    text = text.strip().upper()
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)
    for ch in text:
        if ch in "ABCD":
            return ch
    return ""


def extract_first_word(text: str) -> str:
    text = (text or "").strip().lower()
    match = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text)
    return match[0] if match else ""


def generate(model, processor, tokenizer, image, prompt: str, device: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[image], text=text, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def eval_mmbench(model, processor, tokenizer, device: str, limit: int | None) -> dict:
    dataset = load_dataset("deepvk/MMBench-ru", split="dev")
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    correct = 0
    results = []
    for ex in tqdm(dataset, desc="mmbench-ru"):
        hint = ex.get("hint", "")
        hint_block = f"\nПодсказка: {hint}" if hint and str(hint).strip() else ""
        prompt = (
            "<image>\n"
            "Выбери правильный вариант ответа по изображению.\n"
            "Ответь только одной буквой: A, B, C или D.\n\n"
            f"Вопрос: {ex['question']}"
            f"{hint_block}\n\n"
            f"A: {ex['A']}\nB: {ex['B']}\nC: {ex['C']}\nD: {ex['D']}\n"
        )
        raw = generate(model, processor, tokenizer, ex["image"], prompt, device, max_new_tokens=8)
        pred = extract_choice(raw)
        gold = ex["answer"].strip().upper()
        ok = int(pred == gold)
        correct += ok
        results.append({"prediction": pred, "gold": gold, "raw_output": raw, "correct": ok})

    return {
        "benchmark": "mmbench-ru",
        "accuracy": correct / len(dataset),
        "total": len(dataset),
        "correct": correct,
        "results": results,
    }


def eval_gqa(model, processor, tokenizer, device: str, limit: int | None) -> dict:
    instructions = load_dataset("deepvk/GQA-ru", "testdev_balanced_instructions", split="testdev")
    images = load_dataset("deepvk/GQA-ru", "testdev_balanced_images", split="testdev")
    img_map = {str(row["id"]): row["image"] for row in images}

    if limit:
        instructions = instructions.select(range(min(limit, len(instructions))))

    correct = 0
    results = []
    for ex in tqdm(instructions, desc="gqa-ru"):
        image = img_map[str(ex["imageId"])]
        prompt = f"<image>\n{ex['question']}\nОтветь одним словом."
        raw = generate(model, processor, tokenizer, image, prompt, device, max_new_tokens=8)
        pred = extract_first_word(raw)
        gold = extract_first_word(ex["answer"])
        ok = int(pred == gold)
        correct += ok
        results.append(
            {
                "id": ex["id"],
                "prediction": pred,
                "gold": gold,
                "raw_output": raw,
                "correct": ok,
            }
        )

    return {
        "benchmark": "gqa-ru",
        "accuracy": correct / len(instructions),
        "total": len(instructions),
        "correct": correct,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Intel/llava-gemma-2b")
    parser.add_argument("--benchmark", choices=["mmbench", "gqa", "both"], default="both")
    parser.add_argument("--output-dir", default="results/baseline/intel")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor, tokenizer = load_model(args.model, device=device)
    summary = {"model": args.model}

    if args.benchmark in ("mmbench", "both"):
        mmbench = eval_mmbench(model, processor, tokenizer, device, args.limit)
        summary["mmbench_ru"] = mmbench["accuracy"]
        with open(output_dir / "mmbench_ru_metric.json", "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in mmbench.items() if k != "results"}, f, indent=2)
        with open(output_dir / "mmbench_ru_results.json", "w", encoding="utf-8") as f:
            json.dump(mmbench["results"], f, ensure_ascii=False)

    if args.benchmark in ("gqa", "both"):
        gqa = eval_gqa(model, processor, tokenizer, device, args.limit)
        summary["gqa_ru"] = gqa["accuracy"]
        with open(output_dir / "gqa_ru_metric.json", "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in gqa.items() if k != "results"}, f, indent=2)
        with open(output_dir / "gqa_ru_results.json", "w", encoding="utf-8") as f:
            json.dump(gqa["results"], f, ensure_ascii=False)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
