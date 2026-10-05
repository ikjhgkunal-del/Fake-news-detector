"""
Truth Shield - Improved Model Training Pipeline
================================================
Dataset Strategy:
  1. GonzaloA/fake_news  (ISOT dataset: 40,000+ Reuters real + PolitiFact fake articles)
  2. ioverho/misinfo-general (NELA corpus: diverse news sources 2017-2022)

Combined, this gives a rich, varied training set that generalizes much better
than the small 3,000-sample run we did before.

Labels in ISOT dataset: 0 = Fake, 1 = Real
"""

import os
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "roberta-base"
MAX_LENGTH   = 256          # 256 is faster than 512 and works great for news headlines+body
TRAIN_EPOCHS = 4
BATCH_SIZE   = 16           # 16 works great on RTX 3050 with mixed precision
SAVE_PATH    = os.path.abspath(os.path.join("..", "backend", "models", "phase1_roberta_fulltune"))
# ───────────────────────────────────────────────────────────────────────────────

def load_isot_dataset():
    """Load the full ISOT fake news dataset (40,000+ articles). Labels: 0=Fake, 1=Real"""
    print("📥 [1/2] Loading ISOT Fake News dataset (full 40,000+ articles)...")
    ds = load_dataset("GonzaloA/fake_news")
    ds = ds.select_columns(["text", "label"])
    print(f"   ✓ ISOT: {len(ds['train'])} train | {len(ds['validation'])} val | {len(ds['test'])} test")
    return ds

def load_misinfo_general():
    """
    Load ioverho/misinfo-general (NELA corpus, diverse sources 2017-2022).
    Labels: 1=reliable(real), 0=unreliable(fake) - same schema as ISOT.
    """
    print("📥 [2/2] Loading misinfo-general dataset (NELA corpus, diverse topics)...")
    try:
        ds = load_dataset("ioverho/misinfo-general", split="train")
        ds = ds.select_columns(["content", "label"])
        ds = ds.rename_column("content", "text")
        ds = ds.shuffle(seed=42).select(range(min(20000, len(ds))))
        print(f"   ✓ misinfo-general: {len(ds)} samples loaded")
        return ds
    except Exception as e:
        print(f"   ⚠️  Could not load misinfo-general ({e}). Continuing with ISOT only.")
        return None

def tokenize(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def main():
    print("\n" + "="*60)
    print(" 🚀 Truth Shield — Improved Training Pipeline")
    print("="*60 + "\n")

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)} — training will be FAST!\n")
    else:
        print("⚠️  No GPU detected — training on CPU (will be slow, but works).\n")

    # ── 1. Load Datasets ──────────────────────────────────────────────────────
    isot_ds = load_isot_dataset()
    misinfo_ds = load_misinfo_general()

    # ── 2. Merge Datasets ─────────────────────────────────────────────────────
    train_datasets = [isot_ds["train"]]
    if misinfo_ds is not None:
        misinfo_split = misinfo_ds.train_test_split(test_size=0.2, seed=42)
        train_datasets.append(misinfo_split["train"])

    combined_train = concatenate_datasets(train_datasets).shuffle(seed=42)
    combined_val   = isot_ds["validation"]

    print(f"\n📊 Final Training Set : {len(combined_train):,} samples")
    print(f"📊 Final Validation Set: {len(combined_val):,} samples\n")

    # ── 3. Tokenize ───────────────────────────────────────────────────────────
    print("⚙️  Loading tokenizer and tokenizing all data...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenize_fn = lambda examples: tokenize(examples, tokenizer)
    tokenized_train = combined_train.map(tokenize_fn, batched=True, desc="Tokenizing train")
    tokenized_val   = combined_val.map(tokenize_fn, batched=True, desc="Tokenizing val")

    # ── 4. Load Model ─────────────────────────────────────────────────────────
    print("\n🏗️  Loading base RoBERTa model with classification head...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # ── 5. Training Arguments ─────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir="./fake_news_roberta_model",
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=(device == "cuda"),      # Mixed precision: 2x faster on GPU!
        dataloader_num_workers=0,     # Keep at 0 on Windows
        report_to="none",
    )

    # ── 6. Train ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )

    print("\n🔥 Starting Training! This will take a while...\n")
    trainer.train()

    # ── 7. Evaluate on Test Set ───────────────────────────────────────────────
    print("\n📈 Evaluating on ISOT test set...")
    tokenized_test = isot_ds["test"].map(tokenize_fn, batched=True, desc="Tokenizing test")
    test_results = trainer.evaluate(tokenized_test)
    print(f"\n🏆 Final Test Results:")
    for k, v in test_results.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")

    # ── 8. Save Model ─────────────────────────────────────────────────────────
    print(f"\n💾 Saving trained model to:\n   {SAVE_PATH}\n")
    os.makedirs(SAVE_PATH, exist_ok=True)
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    print("="*60)
    print(" ✅ Training Complete! Your model is ready.")
    print(f" 📁 Saved to: {SAVE_PATH}")
    print("="*60 + "\n")
    print("Next step: Restart the backend server to load the new model.")

if __name__ == "__main__":
    main()
