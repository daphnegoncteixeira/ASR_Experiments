import os
import wave
import argparse
import numpy as np
from pydub import AudioSegment
import soundfile as sf
import librosa

def check_dependencies():
    """Vérifie si toutes les dépendances nécessaires sont installées"""
    required = ['numpy', 'pydub', 'soundfile', 'librosa']
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    # Si des dépendances sont manquantes, lève une erreur
    if missing:
        raise ImportError(f"Dépendances manquantes : {', '.join(missing)}. "
                         f"Veuillez les installer avec : pip install {' '.join(missing)}")

def process_wav_files(input_dir, output_dir, min_duration=10, max_duration=20, sample_rate=16000):
    """
    Traite les fichiers WAV pour un entraînement wav2vec :
    - Exclut les fichiers contenant 'Emilie_K' ou 'Fabiano' dans leur nom
    - Coupe les fichiers en segments de 10 à 20 secondes
    - Rééchantillonne à 16 kHz si nécessaire
    - Sauvegarde les fichiers traités dans un répertoire de sortie
    """
    # Crée le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    processed_files = 0  # Compteur de fichiers traités
    excluded_files = 0   # Compteur de fichiers exclus
    
    # Parcours récursif du répertoire d’entrée
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith('.wav'):
                # Ignore les fichiers contenant les noms exclus
                if 'Emilie_K' in filename or 'Fabiano' in filename:
                    excluded_files += 1
                    continue
                
                input_path = os.path.join(root, filename)
                
                try:
                    # Charge le fichier audio avec librosa
                    audio, sr = librosa.load(input_path, sr=None, mono=True)
                    
                    # Rééchantillonne à 16 kHz si ce n’est pas déjà le cas
                    if sr != sample_rate:
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
                    
                    # Calcule la durée en secondes
                    duration = len(audio) / sample_rate
                    
                    if duration <= max_duration:
                        # Si le fichier est plus court que max_duration, l’enregistre tel quel
                        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.wav")
                        sf.write(output_path, audio, sample_rate)
                        processed_files += 1
                    else:
                        # Sinon, le découpe en segments de longueur maximale
                        chunk_size = sample_rate * max_duration  # Taille d’un segment en échantillons
                        overlap = sample_rate * 2  # Recouvrement de 2 secondes
                        start = 0
                        chunk_num = 0
                        
                        while start + chunk_size <= len(audio):
                            end = start + chunk_size
                            chunk = audio[start:end]
                            
                            # Vérifie si le segment respecte la durée minimale
                            if len(chunk) / sample_rate >= min_duration:
                                output_path = os.path.join(
                                    output_dir, 
                                    f"{os.path.splitext(filename)[0]}_chunk{chunk_num}.wav"
                                )
                                sf.write(output_path, chunk, sample_rate)
                                processed_files += 1
                                chunk_num += 1
                            
                            # Décale le point de départ en prenant en compte le recouvrement
                            start = end - overlap
                
                except Exception as e:
                    print(f"Erreur lors du traitement de {filename} : {str(e)}")
                    continue
    
    # Affiche un résumé du traitement
    print(f"Traitement terminé. {processed_files} fichiers traités, {excluded_files} fichiers exclus.")

if __name__ == "__main__":
    # Définit les arguments en ligne de commande attendus
    parser = argparse.ArgumentParser(description='Traite des fichiers WAV pour un entraînement wav2vec')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Répertoire d’entrée contenant les fichiers WAV')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Répertoire de sortie pour les fichiers traités')
    args = parser.parse_args()
    
    try:
        check_dependencies()
        print("Toutes les dépendances sont installées.")
        process_wav_files(args.input_dir, args.output_dir)
    except ImportError as e:
        print(e)
