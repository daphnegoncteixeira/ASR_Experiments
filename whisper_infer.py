# -*- coding: utf-8 -*-
"""
Script d'inférence pour tester un modèle Whisper fine-tuné sur un fichier audio Kriol.

Entrées :
- Chemin vers le modèle fine-tuné (`model_dir`)
- Fichier audio `.wav` en 16 kHz mono

Sortie :
- Transcription avec timecodes sauvegardée en fichier JSON

Auteur : Daphne Teixeira
"""

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torch
import json
import os

# === PARAMÈTRES ===
model_dir = "whisper-kriol-finetuned"  # Dossier contenant le modèle fine-tuné
wav_file = "sample.wav"  # Fichier audio à transcrire (mono, 16 kHz)
output_json = "transcription_output.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

# === CHARGER MODÈLE ET PROCESSOR ===
processor = WhisperProcessor.from_pretrained(model_dir)
model = WhisperForConditionalGeneration.from_pretrained(model_dir).to(device)

# === CHARGER L'AUDIO ===
waveform, sample_rate = torchaudio.load(wav_file)
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

# === TRANSCRIPTION AVEC TIME STAMPS ===
inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
input_features = inputs.input_features.to(device)

with torch.no_grad():
    predicted_ids = model.generate(input_features, return_timestamps=True)

decoded = processor.batch_decode(predicted_ids, skip_special_tokens=False)

# === FORMATAGE DES SORTIES ===
transcription_data = {
    "audio_file": wav_file,
    "transcription": decoded[0]
}

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(transcription_data, f, ensure_ascii=False, indent=2)

print(f"\nTranscription sauvegardée dans : {output_json}")
