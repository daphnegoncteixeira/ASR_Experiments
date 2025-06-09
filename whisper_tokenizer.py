# -*- coding: utf-8 -*-
"""
Préparation du jeu de données Hugging Face (audio + texte) pour le fine-tuning de Whisper.
Utilise le WhisperProcessor (feature extractor + tokenizer).

Auteur : Daphne Teixeira
"""

from transformers import WhisperProcessor
import torch

# === PARAMÈTRES ===
WHISPER_MODEL = "openai/whisper-medium"  # Peut être changé pour 'small', 'large', etc.
LANGUAGE = "kriol"  # Nom libre, juste pour marquage éventuel
TASK = "transcribe"

# === CHARGER LE PROCESSOR ===
processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)

# === FONCTION DE PRÉTRAITEMENT ===
def prepare_example(example):
    audio = example["audio"]
    inputs = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=example["text"],
        return_tensors="pt"
    )

    example["input_features"] = inputs.input_features[0]
    example["labels"] = inputs.labels[0]
    return example

# === APPLICATION AU DATASET ===
dataset = dataset.map(prepare_example, remove_columns=dataset["train"].column_names)

# === AFFICHAGE DE CONTRÔLE ===
print("Exemple prétraité :")
print(dataset["train"][0])
