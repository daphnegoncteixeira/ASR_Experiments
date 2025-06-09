# -*- coding: utf-8 -*-
"""
Script d'entraînement du modèle Whisper avec Hugging Face `Trainer`.

Ce script suppose que vous avez déjà :
1. Un fichier CSV contenant vos données, transformé en `datasets.DatasetDict`
2. Les colonnes "input_features" (caractéristiques audio extraites) et "labels" (séquences tokenisées)
3. Le jeu de données est divisé en "train" et "test" (avec `train_test_split`)

Entrée attendue :
- `dataset`: un objet `datasets.DatasetDict` contenant deux sous-ensembles : `train` et `test`
- Chaque exemple doit contenir les champs :
    - `input_features`: tenseurs audio préparés avec `WhisperProcessor`
    - `labels`: séquences tokenisées du texte de transcription

Sortie :
- Modèle Whisper fine-tuné sauvegardé dans le répertoire `OUTPUT_DIR`
- Processeur sauvegardé dans le même dossier
- Score de WER calculé 

Auteur : Daphne Teixeira
"""

from transformers import WhisperForConditionalGeneration, TrainingArguments, Trainer
import torch
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# === PARAMÈTRES ===
MODEL_NAME = "openai/whisper-small"
OUTPUT_DIR = "whisper-kriol-finetuned"

# === CHARGER LE MODÈLE ===
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

# === ARGUMENTS D'ENTRAÎNEMENT ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=100,
    max_steps=1000,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    push_to_hub=False
)

# === COLLECTEUR DE DONNÉES PERSONNALISÉ ===
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [f["labels"] for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad({"input_ids": label_features}, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

# === ÉVALUATION — WER ===
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

# === ENTRAÎNEUR ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor.feature_extractor,
    data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor),
    compute_metrics=compute_metrics
)

# === LANCEMENT ===
trainer.train()

# === SAUVEGARDE ===
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
