# evaluate

```bash
# Intel baseline
python scripts/evaluate.py --model Intel/llava-gemma-2b --output-dir results/baseline/intel

# Finetuned LoRA
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --model checkpoints/intel-llava-gemma-2b-ru \
  --output-dir results/intel-ru

# VK reference (same script)
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --model deepvk/llava-gemma-2b-lora \
  --output-dir results/reference/vk

# One benchmark only / smoke test
python scripts/evaluate.py --benchmark mmbench --limit 100
python scripts/evaluate.py --benchmark gqa

# Reparse MMBench from saved raw_output
python scripts/reparse_mmbench.py
```

On Kaggle: `pip uninstall -y torchao` before train/eval if PEFT fails on torchao version.
