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
