import os
import numpy as np
import json
import cv2
import librosa
import moviepy.editor as mp
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

# Set file paths
video_path = "C:/Users/user/PR_113_Assignment1/videos/video2.mp4"
gt_path = "C:/Users/user/PR_113_Assignment1/labels/video2.json"

# --- Traditional Feature Extraction ---
def extract_traditional_audio_features(video_path):
    audio_path = video_path.replace(".mp4", ".wav")
    video_clip = mp.VideoFileClip(video_path)
    if video_clip.audio is None:
        return np.zeros(1000), np.zeros(1000)
    video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le', verbose=False, logger=None)
    y, sr = librosa.load(audio_path, sr=None)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return zcr.mean(axis=0), centroid.mean(axis=0)

def extract_traditional_video_features(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_diffs = []
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return np.zeros(1000)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        frame_diffs.append(np.mean(diff))
        prev_gray = gray
    cap.release()
    if not frame_diffs:
        return np.zeros(1000)
    return np.array(frame_diffs)

# Modern feature extraction
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

def evaluate_highlights(predicted_summary, ground_truth_path):
    with open(ground_truth_path, 'r') as f:
        gt_data = json.load(f)
        gt_summary = np.array(gt_data["user_summary"]).mean(axis=0) >= 0.5
    min_len = min(len(gt_summary), len(predicted_summary))
    return precision_recall_fscore_support(
        gt_summary[:min_len], predicted_summary[:min_len], average='binary')[:3]

# --- Cross Feature Ablation Experiments ---
experiments = {
    "Modern_Video + Modern_Audio": {"video": "modern", "audio": "modern"},
    "Modern_Video + Traditional_Audio": {"video": "modern", "audio": "traditional"},
    "Traditional_Video + Modern_Audio": {"video": "traditional", "audio": "modern"},
    "Traditional_Video + Traditional_Audio": {"video": "traditional", "audio": "traditional"},
}

results = []

for label, cfg in experiments.items():
    if cfg["video"] == "modern":
        motion_scores = extract_video_features(video_path)
    else:
        motion_scores = extract_traditional_video_features(video_path)

    if cfg["audio"] == "modern":
        energy, contrast = extract_audio_features(video_path)
    else:
        energy, contrast = extract_traditional_audio_features(video_path)

    min_len = min(len(motion_scores), len(energy), len(contrast))
    combined = motion_scores[:min_len] + energy[:min_len] + contrast[:min_len]
    predicted = (combined > np.percentile(combined, 75)).astype(int)

    precision, recall, f1 = evaluate_highlights(predicted, gt_path)
    results.append((label, cfg["video"], cfg["audio"], precision, recall, f1))

ablation_df = pd.DataFrame(results, columns=["Exp", "Video Method", "Audio Method", "Precision", "Recall", "F1-score"])
print(ablation_df)
