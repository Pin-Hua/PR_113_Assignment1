import os
import numpy as np
import json
import cv2
import librosa
import moviepy.editor as mp
from scipy.signal import medfilt
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

# Set file paths
video_path = "C:/Users/user/PR_113_Assignment1/videos/video2.mp4"
gt_path = "C:/Users/user/PR_113_Assignment1/labels/video2.json"

# --- Feature Extraction Functions ---
def extract_audio_features(video_path):
    audio_path = video_path.replace(".mp4", ".wav")
    video_clip = mp.VideoFileClip(video_path)
    if video_clip.audio is None:
        return np.zeros(1000), np.zeros(1000)
    video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le', verbose=False, logger=None)
    y, sr = librosa.load(audio_path, sr=None)
    rms = librosa.feature.rms(y=y)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    return rms.mean(axis=0), spectral_contrast.mean(axis=0)

def extract_video_features(video_path, smoothing_window=5):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return np.zeros(1000)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    flow_magnitudes = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_magnitudes.append(mag.mean())
        prev_gray = gray
    cap.release()
    if not flow_magnitudes:
        return np.zeros(1000)
    flow_magnitudes = np.convolve(flow_magnitudes, np.ones(smoothing_window) / smoothing_window, mode='same')
    return np.array(flow_magnitudes)

def smooth_predictions(predicted_summary, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    return medfilt(predicted_summary, kernel_size)

def evaluate_highlights(predicted_summary, ground_truth_path):
    with open(ground_truth_path, 'r') as f:
        gt_data = json.load(f)
        gt_summary = np.array(gt_data["user_summary"]).mean(axis=0) >= 0.5
    min_len = min(len(gt_summary), len(predicted_summary))
    return precision_recall_fscore_support(
        gt_summary[:min_len], predicted_summary[:min_len], average='binary')[:3]

# --- Ablation Experiments ---
experiments = {
    "A": {"motion": True, "audio": True, "smooth": True},
    "B": {"motion": True, "audio": False, "smooth": True},
    "C": {"motion": False, "audio": True, "smooth": True},
    "D": {"motion": True, "audio": True, "smooth": False},
    "E": {"motion": True, "audio": False, "smooth": False},
    "F": {"motion": False, "audio": True, "smooth": False},
}

results = []

for label, cfg in experiments.items():
    motion_scores = extract_video_features(video_path) if cfg["motion"] else np.zeros(1000)
    energy, contrast = extract_audio_features(video_path) if cfg["audio"] else (np.zeros(1000), np.zeros(1000))

    min_len = min(len(motion_scores), len(energy), len(contrast))
    combined = motion_scores[:min_len] + energy[:min_len] + contrast[:min_len]
    predicted = (combined > np.percentile(combined, 75)).astype(int)

    if cfg["smooth"]:
        predicted = smooth_predictions(predicted)

    precision, recall, f1 = evaluate_highlights(predicted, gt_path)
    results.append((label, cfg["motion"], cfg["audio"], cfg["smooth"], precision, recall, f1))

# Output result
ablation_df = pd.DataFrame(results, columns=["Exp", "Motion", "Audio", "Smoothing", "Precision", "Recall", "F1-score"])
print(ablation_df)