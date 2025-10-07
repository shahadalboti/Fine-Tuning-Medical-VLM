"""
MedGemma Fine-tuning Script for Diabetic Retinopathy Detection

This script fine-tunes the MedGemma 4B model on your diabetic retinopathy datasets
"""

import os
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import gc
import cv2
from torchvision import transforms
import random
import logging
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")

# Transformers and training imports
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForImageTextToText,  # Added for MedGemma compatibility
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset as HFDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("🔄 Step 1: Clearing CUDA cache...")
torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True

# Custom CLAHE transformation class (same as your original)
class ApplyCLAHE:
    def __init__(self, cliplimit=1.0, tilesize=8):
        self.cliplimit = cliplimit
        self.tilesize = tilesize

    def __call__(self, image):
        image = np.array(image)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab_planes = list(cv2.split(lab))
        clahe = cv2.createCLAHE(clipLimit=self.cliplimit, tileGridSize=(self.tilesize, self.tilesize))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        image_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(image_clahe)

# Define augmentation transforms (same as your original)
train_transform = transforms.Compose([
    ApplyCLAHE(cliplimit=1.0, tilesize=8),  # Enhance original
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    transforms.RandomRotation(degrees=15),
])

# No augmentation for validation (same as your original)
val_transform = transforms.Compose([
    ApplyCLAHE(cliplimit=1.0, tilesize=8),  # Only enhance, no random augmentations
])

print("📂 Step 2: Defining CSV and image paths...")

# CSV files and image folders (your exact paths)
csv_paths = [
    "Path/To/your/data.csv",
]

image_folders = [
    "Path/To/your/data/folder"
]

print("🖼️ Step 5: Defining image loader...")
def load_Fundes_image(path, transform=None):
    try:
        image_rgb = Image.open(path).convert("RGB")
        max_size = 560
        if max(image_rgb.size) > max_size:
            image_rgb.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        if transform is not None:
            image_rgb = transform(image_rgb)
        return image_rgb
    except Exception as e:
        print(f"[!] Failed to load image: {path} ({e})")
        return None

system_message = (
    "You are an ophthalmologist diagnosing diabetic retinopathy from retinal images. "
    "You must analyze every image - never skip any image regardless of quality."
)

instruction_text = """
Diagnose diabetic retinopathy:
- No DR: No signs for diabetic retinopathy
- DR: Any stage present of diabetic retinopathy

You must diagnose every image.

Format:
{'DR': 'DR'} or {'DR': 'No DR'}
""".strip()

class MedGemmaVisionDataset(Dataset):
    def __init__(self, df_split, processor, system_message, instruction_text, transform=None, is_training=True):
        self.df = df_split.reset_index(drop=True)
        self.processor = processor
        self.system_message = system_message
        self.instruction_text = instruction_text
        self.transform = transform
        self.is_training = is_training
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        current_transform = self.transform if self.is_training else val_transform
        image_obj = load_Fundes_image(row["Image"], transform=current_transform)
        if image_obj is None:
            print(f"**************None_getitem***************")
            return None
        messages = [
            {"role": "user","content": [
                {"type": "text", "text": self.system_message + "\n\n" + self.instruction_text},
                {"type": "image", "image": image_obj}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]}
        ]
        return {"messages": messages, "image": image_obj, "image_path": row["Image"]}

class MedGemmaFineTuner:
    def __init__(self, model_name: str = "google/medgemma-4b-it", output_dir: str = "./medgemma_dr_finetuned"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.model = None
        self.processor = None
        self.tokenizer = None
        os.makedirs(output_dir, exist_ok=True)
    def setup_model_and_processor(self, use_quantization: bool = False):
        logger.info("Setting up model and processor...")
        cache_path = "/home/ncai/users/salbati@NCAI.SA/.cache"
        if use_quantization:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        else:
            quantization_config = None
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16 if not use_quantization else None,
            cache_dir=cache_path,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name, cache_dir=cache_path, local_files_only=True)
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if use_quantization:
            self.model = prepare_model_for_kbit_training(self.model)
        gc.collect(); torch.cuda.empty_cache()
        logger.info("Model and processor setup complete")
    def setup_lora_config(self):
        logger.info("Setting up LoRA configuration...")
        lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=128, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], lora_dropout=0.05, bias="none", inference_mode=False)
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        return lora_config
    def collate_fn(self, batch):
        valid_examples = [ex for ex in batch if ex is not None]
        if not valid_examples:
            print("**************None batch***************"); return None
        try:
            texts, images = [], []
            for item in valid_examples:
                conversation = item["messages"]
                text = self.processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
                texts.append(text); images.append(item["image"])
            batch_inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            labels = batch_inputs["input_ids"].clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            batch_inputs["labels"] = labels
            return batch_inputs
        except Exception as e:
            print(f"Error in collate_fn: {e}"); return None

def main():
    try:
        print("🔄 Step 3: Loading CSV files...")
        dfs = []; total_before = 0; total_after = 0
        for csv_path, img_folder in zip(csv_paths, image_folders):
            if not os.path.exists(csv_path):
                print(f"⚠️ Warning: CSV file not found: {csv_path}"); continue
            df_i = pd.read_csv(csv_path); total_before += len(df_i)
            if "image_exists" not in df_i.columns: df_i["image_exists"] = True
            if "gradability" not in df_i.columns: df_i["gradability"] = 0
            df_i["Image"] = df_i["Image"].apply(lambda x: os.path.join(img_folder, x))
            df_i["image_exists"] = df_i["Image"].apply(os.path.exists)
            valid_images = int(df_i["image_exists"].sum()); total_images = len(df_i)
            print(f"📊 {os.path.basename(csv_path)}: {valid_images}/{total_images} images found")
            df_i = df_i[df_i["image_exists"]].copy(); total_after += len(df_i); dfs.append(df_i)
        if len(dfs) == 0:
            raise RuntimeError("No CSVs loaded successfully. Please check paths.")
        df = pd.concat(dfs, ignore_index=True)
        print(f"📊 Rows before image filter across all CSVs: {total_before}")
        print(f"📊 Rows after image filter across all CSVs:  {total_after}")
        print(f"📊 Total combined rows in df: {len(df)}")
        gc.collect(); torch.cuda.empty_cache()
        print("🔄 Step 4: Processing dataframe...")
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df['Diagnosis'] = df['DR_Binary'].apply(lambda x: "Normal" if x == 0 else "AbNormal")
        df['answer'] = df.apply(lambda row: f"{{'DR': '{'DR' if row['Diagnosis'] == 'AbNormal' else 'No DR'}'}}", axis=1)
        print(f"✅ Final dataset contains {len(df)} rows (no downsampling).")
        print("✂️ Step 7: Splitting data and building datasets...")
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        del df, dfs; gc.collect(); torch.cuda.empty_cache()
        print(f"✅ Loaded {len(train_df)} train and {len(val_df)} validation examples.")
        print("🧠 Step 8: Loading MedGemma model and processor...")
        fine_tuner = MedGemmaFineTuner(output_dir="/home/ncai/users/salbati@NCAI.SA/MedGemma_DR_FineTuned")
        fine_tuner.setup_model_and_processor(use_quantization=False)
        # fine_tuner.setup_lora_config()  # enable for LoRA
        print("✅ Model and processor loaded.")
        print("⚙️ Step 9: Creating datasets...")
        train_dataset = MedGemmaVisionDataset(train_df, fine_tuner.processor, system_message, instruction_text, transform=train_transform, is_training=True)
        eval_dataset = MedGemmaVisionDataset(val_df, fine_tuner.processor, system_message, instruction_text, transform=None, is_training=False)
        training_args = TrainingArguments(
            output_dir="/MedGemma_DR_FineTuned",
            num_train_epochs=10,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            learning_rate=2e-4,
            lr_scheduler_type="constant",
            logging_steps=4,
            eval_steps=4,
            evaluation_strategy="steps",
            save_steps=4,
            save_total_limit=3,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            load_best_model_at_end=True,
            bf16=True,
            tf32=True,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            report_to=None,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
        )
        trainer = Trainer(model=fine_tuner.model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=fine_tuner.collate_fn, tokenizer=fine_tuner.processor.tokenizer, callbacks=[EarlyStoppingCallback(early_stopping_patience=8)])
        gc.collect(); torch.cuda.empty_cache()
        print("🚀 Starting training...")
        trainer.train(); trainer.save_model(); print("[✓] Training complete and model saved.")
        print("🧪 Step 10: Testing inference...")
        test_image_path = train_df.iloc[0]["Image"]
        if os.path.exists(test_image_path):
            test_image = load_Fundes_image(test_image_path, transform=val_transform)
            messages = [
                {"role": "system","content": [{"type": "text", "text": system_message}]},
                {"role": "user","content": [{"type": "text", "text": instruction_text},{"type": "image", "image": test_image}]}
            ]
            try:
                inputs = fine_tuner.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(fine_tuner.model.device, dtype=torch.bfloat16)
                input_len = inputs["input_ids"].shape[-1]
                with torch.inference_mode():
                    generated_ids = fine_tuner.model.generate(**inputs, max_new_tokens=100, temperature=0.3, do_sample=True)
                    generated_text = generated_ids[0][input_len:]
                    response = fine_tuner.processor.decode(generated_text, skip_special_tokens=True).strip()
                print(f"✅ Inference result: {response}")
            except Exception as e:
                print(f"Inference error: {e}")
        logger.info("Fine-tuning completed successfully!")
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}"); raise

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
