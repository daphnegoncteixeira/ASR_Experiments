# -*- coding: utf-8 -*-
"""
Pipeline de fine-tuning du modèle Gervasio (decoder-only, pt-PT) pour post-traitement de transcriptions en Kriol.
Ce script adapte un modèle pré-entraîné portugais pour corriger des transcriptions générées automatiquement (Wav2Vec2).

Pré-requis :
- Un fichier TSV (ou CSV) avec deux colonnes : `input` (brut CTC) et `target` (transcription Kriol corrigée)
- Modèle Gervasio disponible sur Hugging Face

Auteur : Daphne Teixeira
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import torch

# === PARAMÈTRES ===
model_checkpoint = "PORTULAN/gervasio-7b-portuguese-ptpt-decoder"
train_file = "kriol_finetune.tsv"  # Doit contenir colonnes "input" et "target"
output_dir = "gervasio-kriol-finetuned"
epochs = 5
batch_size = 2
max_length = 128

# === TOKENISEUR ET MODÈLE ===
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

# === CHARGEMENT DU DATASET ===
dataset = load_dataset("csv", data_files={"train": train_file}, delimiter="\t")

# === PRÉPARATION DES PROMPTS ===
def preprocess_function(example):
    prompt = f"Corrige le Kriol :\nInput: {example['input']}\nOutput:"
    full_text = prompt + " " + example["target"]
    tokenized = tokenizer(full_text, truncation=True, max_length=max_length, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

encoded = dataset.map(preprocess_function, batched=False)

# === ENTRAÎNEMENT ===
args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    num_train_epochs=epochs,
    logging_dir=f"{output_dir}/logs",
    save_strategy="epoch",
    logging_strategy="epoch",
    evaluation_strategy="no",
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded["train"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("\nFine-tuning terminé. Modèle sauvegardé dans :", output_dir)
