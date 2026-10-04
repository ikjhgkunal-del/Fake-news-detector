import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def main():
    print("🚀 Starting Fake News Model Training Pipeline...")
    
    # 1. Load the Dataset
    # We use the famous ISOT Fake News Dataset (pre-packaged on HuggingFace by GonzaloA)
    # Labels: 0 = Fake, 1 = Real
    print("📥 Downloading ISOT Fake News dataset from HuggingFace...")
    dataset = load_dataset("GonzaloA/fake_news")
    
    # Let's take a subset for faster training (Good for testing your code).
    # NOTE: Once your code works, remove `.select(range(...))` to train on all 40,000+ articles!
    train_dataset = dataset["train"].shuffle(seed=42).select(range(3000))
    val_dataset = dataset["validation"].shuffle(seed=42).select(range(500))

    # 2. Load the Tokenizer
    print("⚙️ Loading roberta-base tokenizer...")
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. Tokenize the text data
    def tokenize_function(examples):
        # Truncate to 512 tokens to fit RoBERTa's max length
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    print("🧠 Tokenizing the dataset (this might take a minute)...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)

    # 4. Load the Base Model
    print("🏗️ Loading base RoBERTa model with a new Classification Head...")
    # num_labels=2 because we are classifying Fake (0) vs Real (1)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 5. Define Evaluation Metrics
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # 6. Setup Training Arguments
    # These hyperparameters are standard for fine-tuning transformers on NLP tasks.
    training_args = TrainingArguments(
        output_dir="./fake_news_roberta_model",
        num_train_epochs=3,              # 3 epochs is usually enough for RoBERTa
        per_device_train_batch_size=8,   # Reduce to 4 if you get 'Out Of Memory' errors
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,               # Helps prevent overfitting
        logging_steps=50,
        eval_strategy="epoch",           # Evaluate at the end of each epoch
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # 7. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )

    # 8. Train!
    print("🔥 Starting Training...")
    trainer.train()

    # 9. Save the Final Model
    print("💾 Saving the trained model...")
    # Save the model directly into the backend folder so the web app can use it
    save_path = os.path.abspath(os.path.join("..", "backend", "models", "phase1_roberta_fulltune"))
    os.makedirs(save_path, exist_ok=True)
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"✅ Training Complete! Model and tokenizer saved to {save_path}")
    print("Now update your backend/.env file to point PHASE1_DIR to this new path!")

if __name__ == "__main__":
    main()
