from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, load_dataset
from PIL import Image
from tqdm import tqdm

GQA_POST_PROMPT = "Ответь одним словом."


def get_gqa_post_prompt(config: dict[str, Any]) -> str:
    return config.get("gqa_post_prompt", GQA_POST_PROMPT)

MODE_PRESETS = {
    "debug": {"llava_instruct_samples": 200, "gqa_train_samples": 200},
    "medium": {"llava_instruct_samples": 10_000, "gqa_train_samples": 8_000},
    "full": {"llava_instruct_samples": None, "gqa_train_samples": None},
}


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def llava_conversation_to_messages(conversations: list[dict[str, str]]) -> list[dict[str, str]]:
    messages = []
    for turn in conversations:
        role = "user" if turn["from"] == "human" else "assistant"
        messages.append({"role": role, "content": turn["value"]})
    return messages


def resolve_coco_image(coco_dir: Path, image_rel_path: str) -> Image.Image | None:
    rel_path = Path(image_rel_path)
    parts = rel_path.parts
    stripped_parts = parts[1:] if parts and parts[0] == "coco" else parts
    stripped_path = Path(*stripped_parts) if stripped_parts else rel_path
    filename = rel_path.name

    candidates = [
        coco_dir / rel_path,
        coco_dir / stripped_path,
        coco_dir / filename,
        coco_dir / "images" / stripped_path,
        coco_dir.parent / rel_path,
        coco_dir.parent / stripped_path,
    ]
    for path in candidates:
        if path.exists():
            return Image.open(path).convert("RGB")
    return None


def load_gqa_train_samples(n_samples: int | None, seed: int, post_prompt: str) -> list[dict[str, Any]]:
    instructions = load_dataset("deepvk/GQA-ru", "train_balanced_instructions", split="train")
    images = load_dataset("deepvk/GQA-ru", "train_balanced_images", split="train")

    img_map = {str(row["id"]): row["image"] for row in images}
    indices = list(range(len(instructions)))
    random.Random(seed).shuffle(indices)
    if n_samples is not None:
        indices = indices[:n_samples]

    examples = []
    skipped = 0
    for idx in tqdm(indices, desc="gqa-ru"):
        row = instructions[idx]
        image_id = str(row["imageId"])
        if image_id not in img_map:
            skipped += 1
            continue

        question = row["question"].strip()
        answer = row["answer"].strip()
        messages = [
            {"role": "user", "content": f"<image>\n{question}\n{post_prompt}"},
            {"role": "assistant", "content": answer},
        ]
        examples.append({"source": "gqa-ru", "messages": messages, "image": img_map[image_id]})

    print(f"gqa-ru: {len(examples)} kept, {skipped} skipped")
    return examples


def load_llava_instruct_samples(
    n_samples: int | None,
    seed: int,
    coco_dir: Path | None,
) -> list[dict[str, Any]]:
    if coco_dir is None:
        return []

    if not coco_dir.exists():
        raise FileNotFoundError(coco_dir)

    dataset = load_dataset("deepvk/LLaVA-Instruct-ru", split="train")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    if n_samples is not None:
        indices = indices[: n_samples * 2]

    examples = []
    skipped = 0
    for idx in tqdm(indices, desc="llava-instruct-ru"):
        if n_samples is not None and len(examples) >= n_samples:
            break

        row = dataset[idx]
        image = resolve_coco_image(coco_dir, row["image"])
        if image is None:
            skipped += 1
            continue

        messages = llava_conversation_to_messages(row["conversations"])
        if not messages:
            skipped += 1
            continue

        examples.append({"source": "llava-instruct-ru", "messages": messages, "image": image})

    print(f"llava-instruct-ru: {len(examples)} kept, {skipped} skipped")
    return examples


def save_manifest(examples: list[dict[str, Any]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            record = {
                "source": ex["source"],
                "messages": ex["messages"],
                "num_messages": len(ex["messages"]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["debug", "medium", "full"], default="debug")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--coco-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    post_prompt = get_gqa_post_prompt(config)
    preset = MODE_PRESETS[args.mode]

    llava_n = preset["llava_instruct_samples"]
    gqa_n = preset["gqa_train_samples"]
    if args.mode == "medium" and data_cfg:
        llava_n = data_cfg.get("llava_instruct_samples", llava_n)
        gqa_n = data_cfg.get("gqa_train_samples", gqa_n)

    seed = args.seed if args.seed is not None else data_cfg.get("seed", 42)
    coco_dir = args.coco_dir or data_cfg.get("coco_images_dir")
    coco_path = Path(coco_dir) if coco_dir else None

    output_dir = Path(args.output_dir or f"data/processed/{args.mode}")
    output_dir.mkdir(parents=True, exist_ok=True)

    gqa_examples = load_gqa_train_samples(gqa_n, seed, post_prompt)
    llava_examples = load_llava_instruct_samples(llava_n, seed, coco_path)
    all_examples = gqa_examples + llava_examples
    random.Random(seed).shuffle(all_examples)

    if not all_examples:
        raise RuntimeError("empty dataset")

    dataset = Dataset.from_list(all_examples)
    dataset_path = output_dir / "train"
    dataset.save_to_disk(str(dataset_path))
    save_manifest(all_examples, output_dir / "train_manifest.jsonl")

    stats = {
        "mode": args.mode,
        "total": len(all_examples),
        "gqa_ru": len(gqa_examples),
        "llava_instruct_ru": len(llava_examples),
        "seed": seed,
        "coco_dir": str(coco_path) if coco_path else None,
        "dataset_path": str(dataset_path),
    }
    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
