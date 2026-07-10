import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration


def load_model(model_name: str, dtype: torch.dtype = torch.float16, device: str = "cuda:0"):
    path = Path(model_name)
    if path.exists() and (path / "adapter_config.json").exists():
        adapter_config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
        base_name = adapter_config.get("base_model_name_or_path", "Intel/llava-gemma-2b")
        model = LlavaForConditionalGeneration.from_pretrained(
            base_name,
            torch_dtype=dtype,
            device_map=device,
        )
        model = PeftModel.from_pretrained(model, model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
        )
        processor = AutoProcessor.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    patch_processor(processor, tokenizer, model)
    return model, processor, tokenizer


def patch_processor(processor, tokenizer, model) -> None:
    processor.tokenizer = tokenizer
    if getattr(tokenizer, "chat_template", None):
        processor.chat_template = tokenizer.chat_template
    processor.patch_size = getattr(getattr(model.config, "vision_config", None), "patch_size", 14)
    processor.num_additional_image_tokens = 1
    processor.vision_feature_select_strategy = getattr(
        model.config, "vision_feature_select_strategy", "default"
    )
