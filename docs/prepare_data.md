# prepare_data

```bash
python scripts/prepare_data.py --mode debug
python scripts/prepare_data.py --mode medium --coco-dir /path/to/train2014
```

| mode | gqa-ru | llava-instruct-ru |
|------|--------|-------------------|
| debug | 200 | 200 |
| medium | 8000 | 10000 |
| full | all | all |

Output: `data/processed/{mode}/train/`
