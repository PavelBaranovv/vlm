import torch
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration


def load_model(model_name: str, dtype: torch.dtype = torch.float16, device: str = "cuda:0"):
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
