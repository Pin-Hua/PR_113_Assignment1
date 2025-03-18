import cv2
import numpy as np
import json
import argparse
import librosa
import librosa.display
import moviepy.editor as mp
import os
import matplotlib.pyplot as plt

class HighlightDetector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video = cv2.VideoCapture(video_path)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    '''
    def extract_audio_features(self):
        """Extracts audio features (RMS energy, spectral contrast)."""
        audio_path = self.video_path.replace(".mp4", ".wav")
        video_clip = mp.VideoFileClip(self.video_path)
        video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le')

        y, sr = librosa.load(audio_path, sr=None)
        energy = librosa.feature.rms(y=y)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=0)

        os.remove(audio_path)  # Clean up extracted audio
        return energy[:self.frame_count], spectral_contrast[:self.frame_count]
    '''
    '''
    def extract_audio_features(self):
        """Extracts audio features (RMS energy, spectral contrast) and aligns them to video frames."""
        audio_path = self.video_path.replace(".mp4", ".wav")
        video_clip = mp.VideoFileClip(self.video_path)
        video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le')

        y, sr = librosa.load(audio_path, sr=None)
        energy = librosa.feature.rms(y=y)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=0)

        os.remove(audio_path)  # Clean up extracted audio

        # Ensure energy & spectral contrast match video frame count
        target_length = self.frame_count
        energy = np.interp(np.linspace(0, len(energy)-1, target_length), np.arange(len(energy)), energy)
        spectral_contrast = np.interp(np.linspace(0, len(spectral_contrast)-1, target_length), np.arange(len(spectral_contrast)), spectral_contrast)

        return energy, spectral_contrast
    '''

    def extract_audio_features(self):
        """Extracts key audio features (RMS energy, spectral contrast) and aligns them to video frames."""
        audio_path = self.video_path.replace(".mp4", ".wav")
        video_clip = mp.VideoFileClip(self.video_path)
        video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le')

        y, sr = librosa.load(audio_path, sr=None)
        energy = librosa.feature.rms(y=y)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=0)

        os.remove(audio_path)  # Clean up extracted audio

        # Ensure energy & spectral contrast match video frame count

        '''
        target_length = self.frame_count - 1  # Match motion_scores length
        energy = np.interp(np.linspace(0, len(energy)-1, target_length), np.arange(len(energy)), energy)
        spectral_contrast = np.interp(np.linspace(0, len(spectral_contrast)-1, target_length), np.arange(len(spectral_contrast)), spectral_contrast)
        '''

        return energy, spectral_contrast


    def extract_video_features(self):
        """Extracts motion-based video features using Optical Flow."""
        motion_magnitudes = []
        prev_frame = None

        while True:
            ret, frame = self.video.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_frame is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                motion_magnitudes.append(np.mean(np.abs(flow)))
            prev_frame = gray

        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video position
        return np.array(motion_magnitudes)
    
    def generate_summary(self, duration=30):
        """Generates a summarized video of specified duration."""
        motion_scores = self.extract_video_features()
        energy, spectral_contrast = self.extract_audio_features()
        combined_scores = motion_scores + energy + spectral_contrast

        # Lower the selection threshold to ensure enough frames
        threshold = np.percentile(combined_scores, 80)  # Reduced from 85 to 80

        # Ensure selection covers full duration by adjusting segment count
        top_segments = np.argsort(combined_scores)[-int((duration * self.fps)):]
        top_segments.sort()

        video_clip = mp.VideoFileClip(self.video_path)
        selected_clips = [video_clip.subclip(max(0, i / self.fps), min(self.duration, (i + 1) / self.fps)) for i in top_segments]

        # Ensure proper concatenation method
        final_clip = mp.concatenate_videoclips(selected_clips, method="compose")

        output_path = self.video_path.replace(".mp4", f"_summary_{duration}s.mp4")
        final_clip.write_videofile(output_path, codec="libx264")

        return output_path

    def evaluate_highlights(self, highlight_mask, gt_path):
        """Evaluate highlight detection against ground truth annotations"""
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)

        if 'user_summary' not in gt_data:
            raise ValueError("Error: 'user_summary' key not found in JSON file.")

        user_summary = np.array(gt_data['user_summary'], dtype=np.float32)

        # If multiple users, average their annotations
        if user_summary.ndim == 2:
            user_summary = user_summary.mean(axis=0)

        # Ensure length matches the video frame count
        if user_summary.shape[0] != self.frame_count:
            raise ValueError(f"Mismatch: Ground truth frames ({user_summary.shape[0]}) != Video frames ({self.frame_count})")

        highlight_mask = np.asarray(highlight_mask, dtype=bool)

        # Check if data is valid before computation
        if user_summary is None or highlight_mask is None:
            raise ValueError("Error: Ground truth (gt) or highlight mask is None.")

        tp = np.logical_and(highlight_mask, user_summary).sum()
        fp = np.logical_and(highlight_mask, np.logical_not(user_summary)).sum()
        fn = np.logical_and(np.logical_not(highlight_mask), user_summary).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fscore = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Evaluation Results:")
        print(f"F-score: {fscore:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")

        return user_summary

    '''
    def evaluate_highlights(self, highlight_mask, gt_path):
        """Evaluate highlight detection against ground truth annotations"""
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)

        user_summary = np.array(gt_data['user_summary'], dtype=np.float32)

        if user_summary.shape[1] != self.frame_count:
            raise ValueError(f"Ground truth frames ({user_summary.shape[1]}) don't match video frames ({self.frame_count})")

        best_fscore, best_precision, best_recall = 0.0, 0.0, 0.0
        gt_c = None
        for gt in user_summary:
            tp = np.logical_and(highlight_mask, gt).sum()
            fp = np.logical_and(highlight_mask, np.logical_not(gt)).sum()
            fn = np.logical_and(np.logical_not(highlight_mask), gt).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            fscore = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            if fscore > best_fscore:
                best_fscore, gt_c = fscore, gt
            if precision > best_precision:
                best_precision = precision
            if recall > best_recall:
                best_recall = recall

        print("\nEvaluation Results:")
        print(f"Best F-score: {best_fscore:.4f}")
        print(f"Best Precision: {best_precision:.4f}")
        print(f"Best Recall: {best_recall:.4f}")

        return gt_c
    '''

def visualize_pred_gt(predicted, ground_truth, output_path="comparison.png"):
    """Visualizes predicted and ground truth moments in a comparison plot."""
    fig, ax = plt.subplots(figsize=(12, 2))

    gt_transitions = np.diff(np.concatenate([[0], ground_truth, [0]]))
    gt_starts, gt_ends = np.where(gt_transitions == 1)[0], np.where(gt_transitions == -1)[0]

    predicted = np.array(predicted, dtype=int)

    if predicted is None or len(predicted) == 0:
        raise ValueError("Error: `predicted` is None or empty.")

    predicted = np.array(predicted, dtype=int)

    pred_transitions = np.diff(np.concatenate([[0], predicted, [0]]))

    # pred_transitions = np.diff(np.concatenate([[0], predicted, [0]]))

    # pred_transitions = np.diff(np.concatenate([[0], predicted, [0]]))
    pred_starts, pred_ends = np.where(pred_transitions == 1)[0], np.where(pred_transitions == -1)[0]

    for start, end in zip(gt_starts, gt_ends):
        ax.barh(y=1, width=end - start, left=start, color='green', height=0.4)
    for start, end in zip(pred_starts, pred_ends):
        ax.barh(y=0, width=end - start, left=start, color='orange', height=0.4)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Predictions', 'Ground Truth'])
    ax.set_xlabel('Frames')
    ax.set_title('Visual Comparison of Video Summarization')

    ax.legend([
        plt.Rectangle((0, 0), 1, 1, color='green', label='Ground Truth'),
        plt.Rectangle((0, 0), 1, 1, color='orange', label='Prediction')
    ])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def main(args):
    detector = HighlightDetector(args.video)
    #highlight_mask = np.zeros(detector.frame_count, dtype=bool)  # Initialize as empty mask
    highlight_mask = None
    # Generate summaries
    summary_30s = detector.generate_summary(duration=30)
    summary_1min = detector.generate_summary(duration=60)

    print(f"Generated Summaries:\n30s: {summary_30s}\n1min: {summary_1min}")

    if args.ground_truth:
        user_summary = detector.evaluate_highlights(highlight_mask, args.ground_truth)
        if user_summary is not None:
            visualize_pred_gt(highlight_mask, user_summary)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--ground_truth", type=str, help="Path to ground truth JSON file")
    args = parser.parse_args()
    main(args)
