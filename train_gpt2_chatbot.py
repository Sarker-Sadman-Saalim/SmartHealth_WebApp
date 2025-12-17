# train_gpt2_chatbot.py
print("=== TRAIN GPT2 SCRIPT STARTED ===")

import os
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

END_TOKEN = "<END>"


def pick_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def compute_top_factors_from_row(row):
    factors = []

    # BMI
    try:
        bmi = float(row["Weight"]) / (float(row["Height"]) ** 2)
    except Exception:
        bmi = None

    if bmi is not None:
        if bmi >= 30:
            factors.append("High BMI")
        elif bmi >= 25:
            factors.append("Overweight BMI")

    # FAF
    if "FAF" in row and pd.notna(row["FAF"]):
        try:
            if float(row["FAF"]) <= 1:
                factors.append("Low physical activity")
        except Exception:
            pass

    # CH2O
    if "CH2O" in row and pd.notna(row["CH2O"]):
        try:
            if float(row["CH2O"]) < 2:
                factors.append("Low water intake")
        except Exception:
            pass

    # Family history
    if "family_history_with_overweight" in row:
        try:
            if str(row["family_history_with_overweight"]).strip().lower() == "yes":
                factors.append("Family history of overweight")
        except Exception:
            pass

    return (factors + ["", "", ""])[:3]


def format_example(question, answer, context):
    tf = context.get("top_factors", ["", "", ""])
    tf = (tf + ["", "", ""])[:3]

    return (
        f"Question: {question}\n"
        f"Prediction: {context.get('prediction', 'Unknown')}\n"
        f"BMI: {float(context.get('bmi', 0)):.1f}\n"
        f"FAF: {context.get('faf', 'N/A')}\n"
        f"FCVC: {context.get('fcvc', 'N/A')}\n"
        f"CH2O: {context.get('ch2o', 'N/A')}\n"
        f"TopFactors: {tf[0]} | {tf[1]} | {tf[2]}\n"
        f"Answer: {answer}\n"
        f"{END_TOKEN}"
    )


def main():
    print(f"CUDA available: {torch.cuda.is_available()}")

    # --- Find dataset files ---
    qa_path = pick_first_existing([
        "data/synthetic_qa_data_v2.csv",
        "data/synthetic_qa_data.csv",
    ])

    profiles_path = pick_first_existing([
        "data/merged_obesity.csv",
        "data/merged_obesity_clean.csv",
    ])

    if qa_path is None:
        raise FileNotFoundError(
            "Could not find QA CSV. Expected: data/synthetic_qa_data_v2.csv or data/synthetic_qa_data.csv"
        )
    if profiles_path is None:
        raise FileNotFoundError(
            "Could not find profiles CSV. Expected: data/merged_obesity.csv or data/merged_obesity_clean.csv"
        )

    print(f"Using QA file: {qa_path}")
    print(f"Using profiles file: {profiles_path}")

    qa_df = pd.read_csv(qa_path)
    profiles = pd.read_csv(profiles_path)

    if "question" not in qa_df.columns or "answer" not in qa_df.columns:
        raise ValueError("QA CSV must contain columns: question, answer")

    if len(profiles) == 0:
        raise ValueError("Profiles CSV is empty.")

    # =========================================================
    # FAST MODE (CPU friendly)
    # =========================================================
    MAX_TRAIN_EXAMPLES = 500  # ✅ fast
    EPOCHS = 1                # ✅ fast
    MAX_LENGTH = 128          # ✅ fast

    qa_df = qa_df.sample(n=min(MAX_TRAIN_EXAMPLES, len(qa_df)), random_state=42).reset_index(drop=True)

    # Match profiles length to qa_df length (reuse rows if needed)
    profiles = profiles.sample(n=len(qa_df), random_state=42, replace=True).reset_index(drop=True)

    # --- Build training texts ---
    texts = []
    for i in range(len(qa_df)):
        p = profiles.loc[i]

        # BMI safely
        try:
            bmi = float(p["Weight"]) / (float(p["Height"]) ** 2)
        except Exception:
            bmi = 0.0

        prediction = "Obesity" if bmi >= 30 else "Overweight" if bmi >= 25 else "Normal"

        context = {
            "prediction": prediction,
            "bmi": bmi,
            "faf": p.get("FAF", "N/A"),
            "fcvc": p.get("FCVC", "N/A"),
            "ch2o": p.get("CH2O", "N/A"),
            "top_factors": compute_top_factors_from_row(p),
        }

        texts.append(format_example(
            qa_df.loc[i, "question"],
            qa_df.loc[i, "answer"],
            context
        ))

    dataset = Dataset.from_pandas(pd.DataFrame({"text": texts}))

    # --- Load GPT-2 ---
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- Tokenize ---
    def tokenize_function(examples):
        enc = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    print("Tokenizing...")
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- Train ---
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=1000,
        save_total_limit=1,
        report_to="none",
        learning_rate=5e-5,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("Training GPT-2 (FAST MODE)...")
    trainer.train()

    # --- Save ---
    out_dir = "./fine_tuned_obesity_model"
    os.makedirs(out_dir, exist_ok=True)

    print("Saving fine-tuned model...")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"✅ Fine-tuning complete! Model saved to '{out_dir}'")


if __name__ == "__main__":
    main()
