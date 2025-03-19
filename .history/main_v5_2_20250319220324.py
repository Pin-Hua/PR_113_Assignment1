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

def extract_video_features(video_path, smoothing_window=5):
    cap = cv2.VideoCapture(video_path)
    features = []
    ret, prev_frame = cap.read()
    if not ret:
        print(f"Error: Cannot read video {video_path}")
        cap.release()
        return np.array(features)

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

    # 平滑化光流變化數據
    flow_magnitudes = np.convolve(flow_magnitudes, np.ones(smoothing_window) / smoothing_window, mode='same')
    return np.array(flow_magnitudes)

from scipy.signal import medfilt

def smooth_predictions(predicted_summary, kernel_size=5):
    """
    使用中值濾波 (Median Filter) 平滑化預測結果，避免零散的預測片段。
    
    :param predicted_summary: 預測的 highlight (二進制陣列)
    :param kernel_size: 濾波器的大小，必須為奇數
    :return: 平滑後的 highlight 預測結果
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # 確保 kernel_size 為奇數
    
    return medfilt(predicted_summary, kernel_size)

def evaluate_highlights(predicted_summary, ground_truth_path):
    with open(ground_truth_path, 'r') as f:
        ground_truth_data = json.load(f)
        ground_truth_summary = np.array(ground_truth_data["user_summary"]).mean(axis=0) >= 0.5
    
    min_length = min(len(ground_truth_summary), len(predicted_summary))
    ground_truth_summary = ground_truth_summary[:min_length]
    predicted_summary = predicted_summary[:min_length]

    # 平滑化預測結果，避免過度選取
    predicted_summary = smooth_predictions(predicted_summary)

    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth_summary, predicted_summary, average='binary')
    return precision, recall, f1


def visualize_pred_gt(predicted, ground_truth_path, output_path="comparison.png"):
    with open(ground_truth_path, 'r') as f:
        ground_truth_data = json.load(f)
        ground_truth = np.array(ground_truth_data["user_summary"]).mean(axis=0) >= 0.5

    min_length = min(len(ground_truth), len(predicted))
    ground_truth = ground_truth[:min_length]
    predicted = predicted[:min_length]

    fig, ax = plt.subplots(figsize=(12, 2))

    gt_transitions = np.diff(np.concatenate([[0], ground_truth, [0]]))
    gt_starts = np.where(gt_transitions == 1)[0]
    gt_ends = np.where(gt_transitions == -1)[0]

    pred_transitions = np.diff(np.concatenate([[0], predicted, [0]]))
    pred_starts = np.where(pred_transitions == 1)[0]
    pred_ends = np.where(pred_transitions == -1)[0]

    for start, end in zip(gt_starts, gt_ends):
        width = end - start
        ax.barh(y=1, width=width, left=start, color='green', height=0.4, alpha=0.8)
    for start, end in zip(pred_starts, pred_ends):
        width = end - start
        ax.barh(y=0, width=width, left=start, color='orange', height=0.4, alpha=0.8)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Predictions', 'Ground Truth'])
    ax.set_xlabel('Frames')
    ax.set_title('Visual Comparison of Video Summarization')

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color='green', label='Ground Truth'),
        plt.Rectangle((0, 0), 1, 1, color='orange', label='Prediction')
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(0.3, -0.15), ncol=2)

    ax.set_xlim(0, len(predicted))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)

    plt.savefig(output_path)
    plt.show()

def get_summary(video_path, summary_length):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    summary_frames = int(summary_length * fps)

    output_video_path = f"summary_{summary_length}s.mp4"
    
    # Extract video frames
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
    
    # Reattach audio using MoviePy
    video_clip = mp.VideoFileClip(video_path)
    summary_clip = mp.VideoFileClip(output_video_path)

    if video_clip.audio is not None:
        summary_clip = summary_clip.set_audio(video_clip.audio)
        output_video_path_with_audio = f"summary_{summary_length}s_with_audio.mp4"
        summary_clip.write_videofile(output_video_path_with_audio, codec="libx264", audio_codec="aac")
        print(f"Generated summary with audio saved at {output_video_path_with_audio}")
    else:
        print(f"Warning: No audio found in {video_path}. Summary generated without audio.")

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
        
        # Visualize predicted summary vs ground truth
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
