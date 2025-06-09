# -*- coding: utf-8 -*-
"""
Script de normalisation orthographique pour un corpus de transcriptions Kriol.
Objectif : homogénéiser les transcriptions pour faciliter l'entraînement et la comparaison.

- Mise en minuscule
- Suppression des ponctuations accolées aux mots (e.g. "bu," -> "bu")
- Ajout d'espaces entre les mots si besoin
- Nettoyage des espaces superflus

Auteur : Daphne Teixeira
"""

import csv
import re

# === PARAMÈTRES ===
INPUT_CSV = 'whisper_kriol_cm.csv'  # Fichier d'entrée
OUTPUT_CSV = 'whisper_kriol_cm_normalized.csv'  # Fichier de sortie
COLUMN_NAME = 'text'

# === FONCTION DE NORMALISATION ===
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"([.,!?;:])", r" \1 ", text)  # Isoler la ponctuation
    text = re.sub(r"[^\w\s.,!?;:']", "", text)  # Enlever tout autre caractère indésirable
    text = re.sub(r"\s+", " ", text)  # Nettoyer les espaces multiples
    return text.strip()

# === TRAITEMENT ===
def normalize_corpus(input_csv, output_csv):
    with open(input_csv, newline='', encoding='utf-8') as f_in, \
         open(output_csv, 'w', newline='', encoding='utf-8') as f_out:

        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            original = row[COLUMN_NAME]
            row[COLUMN_NAME] = normalize_text(original)
            writer.writerow(row)

    print(f"Corpus normalisé sauvegardé dans : {output_csv}")

# === EXÉCUTION ===
if __name__ == '__main__':
    normalize_corpus(INPUT_CSV, OUTPUT_CSV)
