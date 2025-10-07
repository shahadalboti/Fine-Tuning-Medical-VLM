# Fundus DR — Lingshu (Qwen2.5-VL) + MedGemma

This repo contains:
- **Lingshu-32B** (Qwen2.5-VL-compatible) LoRA fine-tuning + inference
- **MedGemma** LoRA fine-tuning + inference (in `medgemma/`)


## Lingshu — Train
```bash
python train.py   --csv_paths /data/a.csv,/data/b.csv   --image_folders /imgs_a,/imgs_b   --model_id lingshu-medical-mllm/Lingshu-32B   --output_dir ./outputs/fundus_lora_lingshu   --epochs 40 --train_bs 16 --eval_bs 2
```

## Lingshu — Inference
```bash
python inference_lingshu.py   --model_id lingshu-medical-mllm/Lingshu-32B   --cache_dir /path/to/.cache   --local_files_only   --image_root /data/ClinicalTrial   --ground_truth_csv /data/TestCT6-26.csv   --output_csv ./outputs/lingshu_double_prompt_results.csv   --apply_clahe --clahe_cliplimit 1.0 --clahe_tilesize 8   --max_new_tokens 48 --temperature 0.0
```

## MedGemma — Fine-Tune (LoRA or full)
```bash
python medgemma/finetune_lora.py
```

## MedGemma — Inference
```bash
python medgemma/inference.py   --model_id google/medgemma-27b-it   --cache_dir /path/to/.cache   --local_files_only   --image_root /data/ClinicalTrial   --ground_truth_csv /data/TestCT6-26.csv   --output_csv ./outputs/medgemma_double_prompt_results.csv   --apply_clahe --clahe_cliplimit 1.0 --clahe_tilesize 8   --max_new_tokens 100 --temperature 0.3
```

## Notes
- Requires `qwen-vl-utils` (for Lingshu/Qwen vision packaging).
- Set `CUDA_VISIBLE_DEVICES` to control GPUs.
- For 4-bit on Lingshu training, run with `--use_4bit` (and have `bitsandbytes` properly installed).
