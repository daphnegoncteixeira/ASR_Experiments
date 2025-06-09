# ASR_Experiments
Cette branche contient plusieurs expériences liées à la **reconnaissance automatique de la parole (ASR)**.

---

### `pre-process.py`

Ce script prépare les données audio pour l'entraînement auto-supervisé :

- 🔍 Parcourt un répertoire de fichiers `.wav` récursivement.
- 🚫 Ignore automatiquement les fichiers contenant certains noms (par défaut : `Emilie_K`, `Fabiano`).
- 🔄 Rééchantillonne les fichiers à 16 kHz s’ils ne le sont pas déjà.
- ✂️ Découpe les fichiers audio longs en segments de 10 à 20 secondes, avec un **recouvrement de 2 secondes** entre les morceaux pour éviter la perte d'information.
- 💾 Sauvegarde les fichiers traités dans un répertoire de sortie spécifié.

✅ Ce traitement est essentiel pour garantir que les données audio respectent les contraintes de format et de durée attendues par le modèle Wav2Vec2.

---
### `train_wav2vec.py`

Ce script lance **l'entraînement auto-supervisé** d’un modèle `Wav2Vec2ForPreTraining` à l’aide de la bibliothèque 🤗 **Transformers** :

- ⚙️ Définit une **configuration personnalisée** du modèle (taille des couches, nombre de têtes d’attention, masquage temporel, etc.).
- 📦 Charge un **jeu de données audio prétraité** au format HuggingFace (`load_from_disk`).
- 🛠️ Configure les **paramètres d'entraînement** (batch size, accumulation de gradient, logs TensorBoard, précision mixte, etc.).
- 🚀 Utilise la classe `Trainer` pour gérer l’entraînement.
- 🧠 Entraîne le modèle **sans transcriptions**, uniquement à partir du signal audio (apprentissage auto-supervisé).

🎯 Ce script permet de développer un modèle Wav2Vec2 adapté à un domaine ou à une langue spécifique, même en l’absence de données annotées.

---
## `vad_pyannote.py` – Voice Activity Detection (VAD) avec PyAnnote

Ce script applique la **détection d’activité vocale (VAD)** en utilisant le modèle pré-entraîné [pyannote-audio](https://github.com/pyannote/pyannote-audio) pour filtrer les zones non parlées dans les fichiers audio. Il génère des fichiers `.wav` contenant uniquement les segments de parole ainsi que des fichiers `.rttm` avec les timestamps des segments détectés. 

### 📌 But

Ce script doit être lancé **avant `pre-process.py`**. Il permet de réduire le bruit et l’audio non pertinent en :
- Supprimant les parties silencieuses ou non parlées
- Conservant uniquement les régions où la parole est présente
- Exportant les timestamps des segments de parole détectés au format **RTTM**
- 

# Pipeline de Fine-Tuning de Whisper pour le Kriol

Ce dépôt contient un pipeline complet pour fine-tuner le modèle ASR Whisper d'OpenAI sur le Kriol, une langue créole à base portugaise parlée en Guinée-Bissau et en Casamance. Le pipeline est conçu pour des scénarios à faibles ressources, avec peu de données transcrites, et optimisé pour la reproductibilité, l'analyse linguistique et le déploiement sur le terrain.

## 📁 Structure du dépôt

- `whisper_trainer.py` — fine-tuning de Whisper avec Hugging Face Trainer
- `whisper_infer.py` — inférence à partir du modèle entraîné
- `whisper_train.slurm` — script SLURM pour entraînement sur cluster avec LETO
- `whisper_kriol_cm.csv` — corpus segmenté avec transcriptions et étiquettes de variétés
- `simple_normalization.py`— outils de nettoyage et vérification orthographique

## 🧠 Prérequis

- Python 3.8+
- PyTorch avec support GPU
- `transformers`, `datasets`, `torchaudio`, `evaluate`, `accelerate`
- Environnement Conda (ex: `training`)

Installation :
```bash
pip install transformers datasets torchaudio evaluate accelerate



