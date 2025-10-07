# Fundus DR LoRA Fine-Tuning (Qwen2.5-VL)

Fine-tune a vision-language model (Qwen2.5-VL family) with LoRA for **Diabetic Retinopathy (DR)** classification on retinal **fundus** images.

> ⚠️ **Privacy**: Do **not** commit patient images or CSVs. Keep data paths local/private.

## Features
- Lazy on-the-fly image loading to reduce RAM
- Multi-CSV support with per-CSV image root folders
- LoRA via `peft` + supervised fine-tuning via `trl.SFTTrainer`
- Early stopping, gradient checkpointing, and multi-GPU via `device_map="auto"`
- Clean labels: `{'DR':'DR'}` vs `{'DR':'No DR'}`

## Quickstart

1. **Create & activate an environment** (example):
```bash
conda create -n fundus-lora python=3.10 -y
conda activate fundus-lora
pip install -r requirements.txt
```

2. **Prepare your data**:
- One or more CSV files with at least the columns: `Image`, `DR_Binary` (0=Normal, 1=AbNormal).
- Provide matching image root folders (same order as CSVs). The script will join `root/Image` and filter missing files.

3. **Run training**:
```bash
python train.py   --csv_paths /path/a.csv,/path/b.csv   --image_folders /root_a,/root_b   --model_id /path/to/qwen2.5-vl-or-lingshu-32b   --output_dir ./outputs/fundus_lora_qwen   --epochs 40   --train_bs 16   --eval_bs 2
```

### Arguments
- `--csv_paths`: Comma-separated CSV paths (same order as `--image_folders`).
- `--image_folders`: Comma-separated image root folders.
- `--model_id`: Local path or HF hub ID for the base model/processor.
- `--output_dir`: Where to save checkpoints.
- `--epochs`: Number of training epochs (default 40).
- `--train_bs`: Per-device train batch size (default 16).
- `--eval_bs`: Per-device eval batch size (default 2).

## Notes
- Set `CUDA_VISIBLE_DEVICES` to control GPUs. `device_map="auto"` will shard across available GPUs.
- If using 4-bit quantization, pass `--use_4bit` and ensure `bitsandbytes` is installed.
- The script uses conservative defaults for stability on multi-GPU systems.

## License
MIT (add your preferred license).
## Inference (Med-GEMMA Double Prompt)

Run DR detection with a two-step prompt (Fundus -> DR) using Med-GEMMA:

```bash
python inference.py   --model_id google/medgemma-27b-it   --cache_dir /path/to/.cache   --local_files_only   --image_root /data/ClinicalTrial   --ground_truth_csv /data/TestCT6-26_new.csv   --output_csv ./outputs/Local_fundus_MedGemma27_doubleprompt.csv   --apply_clahe --clahe_cliplimit 1.0 --clahe_tilesize 8   --max_new_tokens 100 --temperature 0.3
```
- CSV must contain `Image` (relative path from `--image_root`) and `label` (0/1).
- Results CSV includes raw responses and parsed labels with metrics.
