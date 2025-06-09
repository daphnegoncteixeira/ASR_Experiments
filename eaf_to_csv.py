# -*- coding: utf-8 -*-
"""
Script de conversion des fichiers .eaf (ELAN) en fichier .csv
Chaque ligne contient : chemin vers l'audio, temps de début, temps de fin, transcription, et un tag de variété dialectale (ici fixé à 'CM').

Ce script est à modifier pour changer le tag dialectal à 'GB' pour les corpus correspondants.

Auteur : Daphne Teixeira
"""

import pympi
import csv
import os

# === PARAMÈTRES ===
EAF_FILE = 'Emilie_K.eaf'  # Nom du fichier EAF
AUDIO_FILENAME = 'Emilie_K.wav'  # Fichier audio correspondant
OUTPUT_CSV = 'whisper_kriol_cm.csv'  # Nom du fichier CSV de sortie
DIALECT_TAG = 'CM'  # Tag de variété : 'CM' pour Casamance, 'GB' pour Guinée-Bissau

# === EXTRACTION DES DONNÉES ===

def extract_transcriptions(eaf_file, audio_filename, dialect_tag, output_csv):
    eaf = pympi.Elan.Eaf(eaf_file)
    
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['audio', 'start', 'end', 'text', 'variety'])

        for tier in eaf.get_tier_names():
            for start, end, text in eaf.get_annotation_data_for_tier(tier):
                if text.strip():  # Ne pas inclure les annotations vides
                    writer.writerow([
                        audio_filename,
                        start / 1000.0,  # Conversion en secondes
                        end / 1000.0,
                        text.strip(),
                        dialect_tag
                    ])

    print(f"Fichier CSV créé avec succès : {output_csv}")

# === EXÉCUTION ===
if __name__ == '__main__':
    extract_transcriptions(EAF_FILE, AUDIO_FILENAME, DIALECT_TAG, OUTPUT_CSV)
