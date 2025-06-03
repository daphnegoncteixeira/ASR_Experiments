# Importation des bibliothèques nécessaires
import os
from transformers import (
    Wav2Vec2Config,              # Pour configurer le modèle Wav2Vec2
    Wav2Vec2ForPreTraining,      # Modèle Wav2Vec2 prêt pour l'entraînement auto-supervisé
    Trainer,                     # Classe pour gérer l’entraînement
    TrainingArguments,           # Classe pour spécifier les arguments d'entraînement
    set_seed                     # Pour fixer une graine aléatoire et garantir la reproductibilité
)
from datasets import load_from_disk  # Pour charger un jeu de données sauvegardé localement
import torch

# 1. Configuration
# Fixe la graine aléatoire pour assurer la reproductibilité (résultats constants)
set_seed(42)

# Création d'une configuration personnalisée pour le modèle Wav2Vec2
config = Wav2Vec2Config(
    hidden_size=768,                 # Taille des vecteurs cachés (représentations internes)
    num_hidden_layers=12,           # Nombre de couches de transformeurs dans l'encodeur
    num_attention_heads=12,         # Nombre de têtes d'attention par couche
    intermediate_size=3072,         # Taille de la couche intermédiaire dans le feed-forward
    hidden_act="gelu",              # Fonction d'activation utilisée dans les couches cachées
    hidden_dropout=0.1,             # Pourcentage de dropout appliqué aux couches cachées
    mask_time_prob=0.065,           # Proportion du signal audio à masquer pendant l'entraînement
    mask_time_length=10,            # Longueur des blocs à masquer (en trames audio)
    num_negatives=100,              # Nombre d'exemples négatifs utilisés pour la perte contrastive
    contrastive_logits_temperature=0.1  # Température pour la distribution des logits contrastifs
)

# 2. Initialisation du modèle
# Instancie le modèle Wav2Vec2 avec la configuration définie ci-dessus
model = Wav2Vec2ForPreTraining(config)

# 3. Préparation du jeu de données
# Fonction pour charger un dataset depuis le disque (prétraité au format HuggingFace)
def prepare_dataset(data_dir):
    dataset = load_from_disk(data_dir)  # Charge le dataset à partir du répertoire donné
    return dataset

# 4. Définition des arguments d'entraînement
training_args = TrainingArguments(
    output_dir="/home/dgoncalves/wav2vec_output",         # Répertoire de sortie pour le modèle sauvegardé
    logging_dir="/home/dgoncalves/wav2vec_output/logs",   # Répertoire de logs (pour TensorBoard, etc.)
    per_device_train_batch_size=16,                       # Taille du batch par GPU (ou CPU si pas de GPU)
    gradient_accumulation_steps=2,                        # Nombre d'étapes d'accumulation de gradient avant mise à jour
    learning_rate=5e-5,                                   # Taux d’apprentissage initial
    warmup_steps=1000,                                    # Nombre d'étapes pour le *warmup* du taux d'apprentissage
    max_steps=50000,                                      # Nombre total d’étapes d'entraînement
    save_steps=10000,                                     # Fréquence de sauvegarde du modèle
    logging_steps=100,                                    # Fréquence d'enregistrement des logs
    fp16=True,                                            # Active le calcul en précision mixte (float16) pour accélérer l'entraînement
    dataloader_num_workers=8,                             # Nombre de workers pour charger les données en parallèle
    report_to="tensorboard",                              # Spécifie l’outil de suivi utilisé (ici TensorBoard)
    remove_unused_columns=False                           # Ne supprime pas les colonnes inutilisées du dataset
)

# 5. Lancement de l'entraînement
print("===== DÉBUT DE L'ENTRAÎNEMENT =====")

# Création d'une instance de Trainer, qui gère le processus d'entraînement
trainer = Trainer(
    model=model,                                          # Le modèle à entraîner
    args=training_args,                                   # Les paramètres d’entraînement définis plus haut
    train_dataset=prepare_dataset("/home/dgoncalves/wav_files/processed_16k")  # Chargement du dataset
)

# Démarre réellement l'entraînement du modèle
trainer.train()

print("===== ENTRAÎNEMENT TERMINÉ =====")
