import os
import torch
import argparse
import numpy as np
import soundfile as sf
import librosa
from pyannote.audio import Pipeline
from pyannote.core import Segment, Annotation

# Charger le pipeline VAD
pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                                    use_auth_token=True)

def write_rttm(filename, annotation):
    """Écrit les segments dans un fichier RTTM"""
    with open(filename, 'w') as f:
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            f.write(f"SPEAKER {os.path.splitext(os.path.basename(filename))[0]} 1 {turn.start:.3f} {turn.duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n")

def apply_vad(input_dir, output_dir, sample_rate=16000, min_segment_duration=0.5):
    os.makedirs(output_dir, exist_ok=True)
    rttm_dir = os.path.join(output_dir, "rttm")
    os.makedirs(rttm_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for filename in files:
            if not filename.lower().endswith(".wav"):
                continue

            input_path = os.path.join(root, filename)
            try:
                # Charger l'audio
                audio, sr = librosa.load(input_path, sr=sample_rate)
                vad_result = pipeline({"waveform": torch.tensor(audio).unsqueeze(0), "sample_rate": sample_rate})

                # Annotation pyannote pour RTTM
                annotation = Annotation(uri=os.path.splitext(filename)[0])

                speech_segments = []
                for i, turn in enumerate(vad_result.get_timeline()):
                    start = max(0, int(turn.start * sample_rate))
                    end = min(len(audio), int(turn.end * sample_rate))
                    duration = (end - start) / sample_rate
                    if duration >= min_segment_duration:
                        speech_segments.append(audio[start:end])
                        annotation[Segment(turn.start, turn.end)] = f"SPEAKER_{i:02d}"

                if not speech_segments:
                    print(f"[!] Aucun segment valide détecté dans {filename}")
                    continue

                # Sauvegarder audio filtré
                output_audio_path = os.path.join(output_dir, filename)
                sf.write(output_audio_path, np.concatenate(speech_segments), sample_rate)

                # Sauvegarder le fichier RTTM
                rttm_path = os.path.join(rttm_dir, f"{os.path.splitext(filename)[0]}.rttm")
                write_rttm(rttm_path, annotation)

                print(f"[✓] {filename} traité : {len(speech_segments)} segments parlés détectés")

            except Exception as e:
                print(f"[X] Erreur avec {filename} : {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Appliquer la détection de voix (VAD) avec pyannote-audio")
    parser.add_argument("--input_dir", type=str, required=True, help="Répertoire des fichiers WAV d'entrée")
    parser.add_argument("--output_dir", type=str, required=True, help="Répertoire de sortie pour les fichiers WAV filtrés et RTTM")
    args = parser.parse_args()

    apply_vad(args.input_dir, args.output_dir)
