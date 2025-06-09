# ASR_Experiments
Cette branche contient plusieurs expériences liées à la **reconnaissance automatique de la parole (ASR)**.

---
# Pipeline de Prétraitement et d'Entraînement Auto-Supervisé avec Wav2Vec2

## `vad_pyannote.py` — Détection d'Activité Vocale (VAD) avec PyAnnote

Ce script applique la **détection d’activité vocale (VAD)** en utilisant le modèle pré-entraîné [pyannote-audio](https://github.com/pyannote/pyannote-audio).

### Objectif

Ce traitement est lancé **avant `pre-process.py`** pour filtrer les zones non parlées dans les fichiers `.wav`. Il permet de :

- Supprimer les parties silencieuses
- Conserver uniquement les segments contenant de la parole
- Générer des fichiers `.rttm` avec les timestamps détectés
- Sauvegarder des `.wav` nettoyés pour l'étape suivante

---

## `pre-process.py` — Préparation des Données Audio

Ce script prépare les fichiers audio pour l'entraînement auto-supervisé Wav2Vec2.

### Étapes du pipeline

- Parcourt récursivement un répertoire contenant des fichiers `.wav`
- Ignore les fichiers contenant certains noms (`Emilie_K`, `Fabiano`, etc.)
- Rééchantillonne tous les fichiers à **16 kHz mono**
- Découpe les longs fichiers en segments de **10 à 20 secondes**, avec **2 secondes de recouvrement**
- Sauvegarde les segments dans un répertoire de sortie structuré

Ce traitement garantit la conformité du corpus audio avec les exigences de format du modèle Wav2Vec2.

---

## `train_wav2vec.py` — Entraînement Auto-Supervisé avec Transformers

Ce script lance l'entraînement auto-supervisé du modèle `Wav2Vec2ForPreTraining` via HuggingFace.

### Fonctions principales

- Définit une **configuration sur mesure** du modèle (couches, têtes, masquage, etc.)
- Charge un corpus audio prétraité (`load_from_disk`)
- Configure les **paramètres d'entraînement** (batch size, logs, fp16, etc.)
- Lance l’entraînement avec la classe `Trainer`
- Apprend directement à partir du **signal audio brut**, sans besoin de transcriptions

Ce pipeline permet d’adapter un modèle Wav2Vec2 à une langue ou un domaine spécifique en l’absence de données annotées.

- 

# Pipeline de Fine-Tuning de Whisper pour le Kriol

Ce dépôt contient un pipeline complet pour fine-tuner le modèle ASR Whisper d'OpenAI sur le Kriol, une langue créole à base portugaise parlée en Guinée-Bissau et en Casamance. Le pipeline est conçu pour des scénarios à faibles ressources, avec peu de données transcrites, et optimisé pour la reproductibilité, l'analyse linguistique et le déploiement sur le terrain.

## Structure du dépôt

- `whisper_trainer.py` — fine-tuning de Whisper avec Hugging Face Trainer
- `whisper_infer.py` — inférence à partir du modèle entraîné
- `whisper_train.slurm` — script SLURM pour entraînement sur cluster avec LETO
- `whisper_kriol_cm.csv` — corpus segmenté avec transcriptions et étiquettes de variétés
- `simple_normalization.py`— outils de nettoyage et vérification orthographique

## Prérequis

- Python 3.8+
- PyTorch avec support GPU
- `transformers`, `datasets`, `torchaudio`, `evaluate`, `accelerate`
- Environnement Conda (ex: `training`)

Installation :
```bash
pip install transformers datasets torchaudio evaluate accelerate

```

## Aperçu du pipeline

### 1. Préparer le CSV

Créer un fichier CSV (`whisper_kriol_cm.csv`) avec les colonnes suivantes :

- `audio` — nom du fichier audio complet (ex : `Emilie_K.wav`)
- `start` / `end` — timestamps des segments (en secondes)
- `text` — transcription
- `variety` — étiquette de dialecte (ex : `CM`, `GB`)

### 2. Normaliser et vérifier

Exécuter les scripts suivants pour nettoyer les transcriptions :

```bash
python simple_normalization.py

```

### 3. Charger en dataset Hugging Face

Charger le fichier CSV dans un objet `datasets.DatasetDict` et le diviser en jeux d'entraînement et de test :

```python
from hf_dataset_loader import dataset  # prépare et divise en train/test

```

### 4. Tokeniser pour Whisper

Appliquer le `WhisperProcessor` pour générer les champs `input_features` et `labels` :

```python
from whisper_tokenizer import dataset  # ajoute input_features et labels
```
### 5. Entraîner

Soumettre l'entraînement sur SLURM :

```bash
sbatch whisper_train.slurm
```

Le modèle fine-tuné sera sauvegardé dans :

whisper-kriol-finetuned/

### 6. Inférence

Lancer une transcription sur un fichier `.wav` :

```bash
python whisper_infer.py
```
Les résultats sont sauvegardés dans un fichier JSON

# Pipeline de Fine-Tuning avec Gervasio
à rediger





