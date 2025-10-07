import os
import gc
import argparse
from typing import List, Optional

from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Optional: comment out if not needed in your environment
try:
    from qwen_vl_utils import process_vision_info
except Exception as e:
    raise ImportError("Please install `qwen-vl-utils` or provide `process_vision_info`") from e

from transformers import (
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLProcessor,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer


def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tuning for fundus DR using Qwen2.5-VL")
    p.add_argument("--csv_paths", type=str, required=True,
                   help="Comma-separated list of CSV paths (columns: Image, DR_Binary).")
    p.add_argument("--image_folders", type=str, required=True,
                   help="Comma-separated list of image root folders, same order/length as csv_paths.")
    p.add_argument("--model_id", type=str, required=True,
                   help="Base model path or HF repo (e.g., /path/to/model or Qwen/Qwen2.5-VL-7B).")
    p.add_argument("--output_dir", type=str, default="./outputs/fundus_lora_qwen",
                   help="Directory to save LoRA checkpoints.")
    p.add_argument("--epochs", type=int, default=40, help="Number of training epochs.")
    p.add_argument("--train_bs", type=int, default=16, help="Per-device train batch size.")
    p.add_argument("--eval_bs", type=int, default=2, help="Per-device eval batch size.")
    p.add_argument("--eval_steps", type=int, default=50, help="Evaluation frequency in steps.")
    p.add_argument("--save_steps", type=int, default=100, help="Checkpoint save frequency in steps.")
    p.add_argument("--num_workers", type=int, default=8, help="DataLoader workers.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--use_4bit", action="store_true", help="Enable 4-bit loading via bitsandbytes.")
    return p.parse_args()


def load_fundus_image(path: str, max_size: int = 560) -> Optional[Image.Image]:
    try:
        image_rgb = Image.open(path).convert("RGB")
        if max(image_rgb.size) > max_size:
            image_rgb.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return image_rgb
    except Exception as e:
        print(f"[!] Failed to load image: {path} ({e})")
        return None


SYSTEM_MESSAGE = (
    "You are an ophthalmologist diagnosing diabetic retinopathy from retinal images. "
    "You must analyze every image - never skip any image regardless of quality."
)

INSTRUCTION_TEXT = """
Diagnose diabetic retinopathy:
- No DR: No signs
- DR: Any stage present

You must diagnose every image, even if poor quality.

Format:
{'DR': 'DR'} or {'DR': 'No DR'}
""".strip()


class LazyVisionDataset(Dataset):
    def __init__(self, df_split: pd.DataFrame, system_message: str, instruction_text: str):
        self.df = df_split.reset_index(drop=True)
        self.system_message = system_message
        self.instruction_text = instruction_text

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_obj = load_fundus_image(row["Image"])
        if image_obj is None:
            return None

        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_message}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_obj},
                    {"type": "text", "text": self.instruction_text},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]},
        ]
        return messages


def build_dataframe(csv_paths: List[str], image_folders: List[str]) -> pd.DataFrame:
    assert len(csv_paths) == len(image_folders),         "csv_paths and image_folders must have the same length and order."

    dfs = []
    total_before = 0
    total_after = 0

    for csv_path, img_folder in zip(csv_paths, image_folders):
        if not os.path.exists(csv_path):
            print(f"⚠️ Warning: CSV file not found: {csv_path}")
            continue

        df_i = pd.read_csv(csv_path)
        total_before += len(df_i)

        if "image_exists" not in df_i.columns:
            df_i["image_exists"] = True
        if "gradability" not in df_i.columns:
            df_i["gradability"] = 0

        df_i["Image"] = df_i["Image"].apply(lambda x: os.path.join(img_folder, x))
        df_i["image_exists"] = df_i["Image"].apply(os.path.exists)
        valid_images = int(df_i["image_exists"].sum())
        total_images = len(df_i)
        print(f"📊 {os.path.basename(csv_path)}: {valid_images}/{total_images} images found")

        df_i = df_i[df_i["image_exists"]].copy()
        total_after += len(df_i)
        dfs.append(df_i)

    if len(dfs) == 0:
        raise RuntimeError("No CSVs loaded successfully. Please check paths.")

    df = pd.concat(dfs, ignore_index=True)
    print(f"📊 Rows before image filter across all CSVs: {total_before}")
    print(f"📊 Rows after image filter across all CSVs:  {total_after}")
    print(f"📊 Total combined rows in df: {len(df)}")

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Build labels
    df["Diagnosis"] = df["DR_Binary"].apply(lambda x: "Normal" if x == 0 else "AbNormal")
    df["answer"] = df.apply(
        lambda row: "{'DR': 'DR'}" if row["Diagnosis"] == "AbNormal" else "{'DR': 'No DR'}",
        axis=1,
    )
    print(f"✅ Final dataset contains {len(df)} rows (no downsampling).")
    return df


def collate_fn_builder(processor):
    def collate_fn(examples):
        # filter out None items
        valid_examples = [ex for ex in examples if ex is not None]
        if not valid_examples:
            return None
        try:
            texts = [processor.apply_chat_template(example, tokenize=False) for example in valid_examples]
            image_inputs = [process_vision_info(example)[0] for example in valid_examples]
            batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

            labels = batch["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            for token in ["<|image|>", "<|im_start|>", "<|im_end|>"]:
                token_id = processor.tokenizer.convert_tokens_to_ids(token)
                labels[labels == token_id] = -100
            batch["labels"] = labels
            return batch
        except Exception as e:
            print(f"Error in collate_fn: {e}")
            return None

    return collate_fn


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print("🔄 Step 1: Clearing CUDA cache...")
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True

    print("📂 Step 2: Defining CSV and image paths...")
    csvs = [p.strip() for p in args.csv_paths.split(",") if p.strip()]
    roots = [p.strip() for p in args.image_folders.split(",") if p.strip()]

    print("🔄 Step 3: Loading CSV files...")
    df = build_dataframe(csvs, roots)

    # Split
    print("✂️ Step 4: Splitting data...")
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=args.seed)
    del df
    gc.collect(); torch.cuda.empty_cache()
    print(f"✅ Loaded {len(train_df)} train and {len(val_df)} validation examples.")

    # Load model/processor
    print("🧠 Step 5: Loading model and processor...")
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config if args.use_4bit else None,
    )
    processor = Qwen2VLProcessor.from_pretrained(
        args.model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        use_fast=True,
    )
    print("✅ Model and processor loaded.")
    gc.collect(); torch.cuda.empty_cache()

    # LoRA
    print("⚙️ Step 6: Configuring LoRA...")
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=128,
        bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM",
    )
    lora_model = get_peft_model(model, peft_config)
    lora_model.print_trainable_parameters()

    # Datasets
    train_dataset = LazyVisionDataset(train_df, SYSTEM_MESSAGE, INSTRUCTION_TEXT)
    eval_dataset = LazyVisionDataset(val_df, SYSTEM_MESSAGE, INSTRUCTION_TEXT)

    # Training config
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        learning_rate=2e-4,
        lr_scheduler_type="constant",
        logging_steps=8,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        push_to_hub=False,
        dataloader_num_workers=args.num_workers,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    training_args.remove_unused_columns = False

    # Collator
    collate_fn = collate_fn_builder(processor)

    print("🚀 Starting training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
        processing_class=processor.tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=8)],
    )

    gc.collect(); torch.cuda.empty_cache()
    trainer.train()
    trainer.save_model(args.output_dir)
    print("[✓] Training complete and model saved.")

if __name__ == "__main__":
    main()
