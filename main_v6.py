import cv2
import numpy as np
import librosa
import librosa.display
import json
import matplotlib.pyplot as plt
import argparse
import subprocess
import os
import moviepy.editor as mp
from sklearn.metrics import precision_recall_fscore_support

# Function to extract audio features using RMS energy and spectral contrast
def extract_audio_features(video_path):
    # Convert MP4 to WAV using MoviePy
    if video_path.endswith(".mp4"):
        audio_path = video_path.replace(".mp4", ".wav")
        video_clip = mp.VideoFileClip(video_path)
        if video_clip.audio is None:
            print(f"Warning: No audio track found in {video_path}. Skipping audio feature extraction.")
            return None, None
        video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le')
    else:
        audio_path = video_path
    
    # Process extracted or provided WAV audio
    y, sr = librosa.load(audio_path, sr=None)
    rms = librosa.feature.rms(y=y)  # Root Mean Square (RMS) Energy
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)  # Spectral Contrast
    
    return rms, spectral_contrast

# Function to extract motion-based video features using Optical Flow
def extract_video_features(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []
    ret, prev_frame = cap.read()
    if not ret:
        print(f"Error: Cannot read video {video_path}")
        cap.release()
        return np.array(features)
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        features.append(mag.mean())
        prev_gray = gray
    
    cap.release()
    return np.array(features)

# Function to evaluate the generated summary
def evaluate_highlights(predicted_summary, ground_truth_path):
    with open(ground_truth_path, 'r') as f:
        ground_truth_data = json.load(f)
        ground_truth_summary = np.array(ground_truth_data["user_summary"])  # Selecting best annotation
    
    if ground_truth_summary.size == 0:
        print("Error: Ground truth summary is empty.")
        return 0, 0, 0
    
    best_gt_summary = ground_truth_summary.mean(axis=0) >= 0.5
    
    min_length = min(len(best_gt_summary), len(predicted_summary))
    best_gt_summary = best_gt_summary[:min_length]
    predicted_summary = predicted_summary[:min_length]
    
    precision, recall, f1, _ = precision_recall_fscore_support(best_gt_summary, predicted_summary, average='binary')
    return precision, recall, f1

# Function to visualize prediction vs ground truth
def visualize_pred_gt(predicted_summary, ground_truth_path):
    with open(ground_truth_path, 'r') as f:
        ground_truth_data = json.load(f)
        ground_truth_summary = np.array(ground_truth_data["user_summary"]).mean(axis=0) >= 0.5
    
    min_length = min(len(ground_truth_summary), len(predicted_summary))
    ground_truth_summary = ground_truth_summary[:min_length]
    predicted_summary = predicted_summary[:min_length]
    
    plt.figure(figsize=(10, 2))
    plt.title("Visual Comparison of Video Summarization")
    plt.xlabel("Frames")
    
    plt.bar(np.where(ground_truth_summary)[0], 1, color='green', label='Ground Truth', alpha=0.8, height=0.5)
    plt.bar(np.where(predicted_summary)[0], 0.5, color='orange', label='Prediction', alpha=0.8, height=0.3)
    
    plt.yticks([0.5, 1], ["Predictions", "Ground Truth"])
    plt.legend()
    plt.show()

# Function to generate 30s and 60s video summarization
def get_summary(video_path, summary_length):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    summary_frames = int(summary_length * fps)
    
    output_video_path = f"summary_{summary_length}s.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    frame_indices = np.linspace(0, total_frames - 1, summary_frames, dtype=int)
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        out.write(frame)
    
    cap.release()
    out.release()
    print(f"Generated summary saved at {output_video_path}")

def main():
    parser = argparse.ArgumentParser(description='Video Summarization and Evaluation')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--ground_truth', type=str, required=False, help='Path to the ground truth JSON file (if available)')
    parser.add_argument('--summary_30s', action='store_true', help='Generate a 30-second summary')
    parser.add_argument('--summary_60s', action='store_true', help='Generate a 60-second summary')
    args = parser.parse_args()
    
    # Extract features
    audio_features = extract_audio_features(args.video)
    video_features = extract_video_features(args.video)
    
    # Simulated summary (random for now, should be generated based on extracted features)
    predicted_summary = np.random.randint(0, 2, len(video_features))
    
    # Evaluate the highlights if ground truth exists
    if args.ground_truth:
        precision, recall, f1 = evaluate_highlights(predicted_summary, args.ground_truth)
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")
        visualize_pred_gt(predicted_summary, args.ground_truth)
    else:
        print("Ground truth not provided. Skipping evaluation and visualization.")
    
    # Generate video summaries
    if args.summary_30s:
        get_summary(args.video, 30)
    if args.summary_60s:
        get_summary(args.video, 60)

if __name__ == "__main__":
    main()
