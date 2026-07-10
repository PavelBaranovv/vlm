from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into a full Llava checkpoint")
    parser.add_argument("--base", default="Intel/llava-gemma-2b")
    parser.add_argument("--adapter", default="checkpoints/intel-llava-gemma-2b-ru")
    parser.add_argument("--output", default="models/intel-llava-gemma-2b-ru")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    print(f"loading base {args.base}...")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map=None if args.device == "cpu" else args.device,
        low_cpu_mem_usage=True,
    )
    if args.device == "cpu":
        model = model.to("cpu")

    print(f"loading adapter {args.adapter}...")
    model = PeftModel.from_pretrained(model, args.adapter)
    print("merging...")
    model = model.merge_and_unload()

    print(f"saving to {output}...")
    model.save_pretrained(str(output), safe_serialization=True)

    processor = AutoProcessor.from_pretrained(args.adapter)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    processor.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    print("done")


if __name__ == "__main__":
    main()
