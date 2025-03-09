import cv2
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from scipy.signal import resample

class HighlightDetector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video = cv2.VideoCapture(video_path)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def extract_audio_features(self):

        video = VideoFileClip(self.video_path)
        audio = video.audio
        
        if audio is None:
            print("Warning: No audio found in video")
            return np.zeros(self.frame_count)
            
        audio_array = audio.to_soundarray()
        sample_rate = audio.fps
        
        # Convert to mono if stereo
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
            
        samples_per_frame = int(sample_rate / self.fps)
        
        rms_energies = []

        ###########################
        # Add other audio features# 
        ###########################

        for i in range(0, len(audio_array), samples_per_frame):
            frame = audio_array[i:i + samples_per_frame]
            if len(frame) < samples_per_frame:
                frame = np.pad(frame, (0, samples_per_frame - len(frame)))
            
            rms = np.sqrt(np.mean(frame**2))
            rms_energies.append(rms)
        
        feature_length = len(rms_energies)
        if feature_length != self.frame_count:
            rms_energies = resample(rms_energies, self.frame_count)
        
        audio_features = self._normalize(rms_energies)
        return np.array(audio_features)
    
    def extract_video_features(self):
        histogram_diffs = []
        prev_frame = None
        fs = 0
        while self.video.isOpened():
            ret, frame = self.video.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                  
        ###############################
        # Add other video features.   #
        # Consider static analysis or # 
        # dynamic (temporal) analysis #               
        ###############################

            if prev_frame is not None:
                hist_diff = self.calculate_histogram_difference(prev_frame, gray)
                histogram_diffs.append(hist_diff)
            else:
                histogram_diffs.append(0)
                
            prev_frame = gray
            fs += 1
            
        self.video.release()
        self.frame_count = fs
        features = self._normalize(histogram_diffs)
        return np.array(features)
    
    def calculate_histogram_difference(self, frame1, frame2):
        """Calculate difference between histograms of consecutive frames"""
        hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
        diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return abs(1 - diff)
    
    def _normalize(self, feature):
        """Normalize feature to range [0, 1]"""
        min_val = np.min(feature)
        max_val = np.max(feature)
        if max_val - min_val == 0:
            return np.zeros_like(feature)
        return (feature - min_val) / (max_val - min_val)
    
    def detect_highlights(self, weights=None):
            
        print("Extracting video features...")
        video_features = self.extract_video_features()

        print("Extracting audio features...")
        audio_features = self.extract_audio_features()

        ###########################
        # More advanced classifer #
        # are allowed to consider.#
        ###########################

        # weighted sum
        highlight_scores = weights['audio'] * audio_features
        highlight_scores += weights['histogram'] * video_features
        
        return audio_features, video_features, highlight_scores
    
    def plot_highlight_curve(self, audio_features, video_features, highlight_scores, weights, window_size=5):
        """Plot individual features and the highlight detection curve"""
        
        time_axis = np.linspace(0, self.duration, len(highlight_scores))
        
        fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        axs[0].plot(time_axis, audio_features, 'g-', label=f'RMS Energy (weight: {weights["audio"]:.2f})')
        axs[0].set_title('Audio Features')
        axs[0].set_ylabel('Normalized Amplitude')
        axs[0].grid(True)
        axs[0].legend()
        
        axs[1].plot(time_axis, video_features, 'b-', label=f'Differences of Histogram (weight: {weights["histogram"]:.2f})')
        axs[1].set_title('Video Features')
        axs[1].set_ylabel('Normalized Changes')
        axs[1].grid(True)
        axs[1].legend()
        
        axs[2].plot(time_axis, highlight_scores, 'r-', label='Combined Highlight Score')
        axs[2].set_title('Combined Highlight Detection Score')
        axs[2].set_xlabel('Time (seconds)')
        axs[2].set_ylabel('Score')
        axs[2].grid(True)
        
        threshold = np.mean(highlight_scores) + np.std(highlight_scores)
        axs[2].axhline(y=threshold, color='k', linestyle='--', label='Highlight Threshold')
        
        highlight_moments = time_axis[highlight_scores > threshold]
        highlight_values = highlight_scores[highlight_scores > threshold]
        axs[2].scatter(highlight_moments, highlight_values, 
                      color='red', marker='o', label='Highlight Moments')
        
        axs[2].legend()
        
        plt.tight_layout()
        plt.savefig('highlight_analysis.png')
        plt.show()
        
        highlight_mask = highlight_scores > threshold
        return threshold, highlight_mask

    def identify_highlight_segments(self, highlight_mask, min_segment_length=10):
        """
        Identify continuous segments of highlight frames
        
        Parameters:
        -----------
        highlight_mask : numpy.ndarray
            Boolean mask indicating whether each frame is a highlight (True) or not (False)
        min_segment_length : int
            Minimum number of consecutive frames to consider as a highlight segment
            
        Returns:
        --------
        list of tuples (start_frame, end_frame)
            List of frame ranges for each highlight segment
        """

        transitions = np.diff(highlight_mask.astype(int))
        start_indices = np.where(transitions == 1)[0] + 1
        end_indices = np.where(transitions == -1)[0] + 1
        
        # check the first frame
        if highlight_mask[0]:
            start_indices = np.insert(start_indices, 0, 0)

        # check the last frame
        if highlight_mask[-1]:
            end_indices = np.append(end_indices, len(highlight_mask))
            
        highlight_segments = []
        for start, end in zip(start_indices, end_indices):
            if end - start >= min_segment_length:
                highlight_segments.append((int(start), int(end)))
                

        return highlight_segments

    def create_highlight_video(self, video_path, highlight_segments, output_path='highlight_video.mp4', 
                              padding_seconds=1.5):
        """
        Create a video containing only the highlight segments
        
        Parameters:
        -----------
        video_path : str
            Path to the original video
        highlight_segments : list of tuples
            List of (start_frame, end_frame) tuples for each highlight segment
        output_path : str
            Path where the highlight video will be saved
        padding_seconds : float
            Additional seconds to include before and after each highlight segment
        """
        if not highlight_segments:
            raise ValueError("No highlight segments found. No video created.")
            
        padding_frames = int(self.fps * padding_seconds)
        
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        for segment_idx, (start_frame, end_frame) in enumerate(highlight_segments):
            padded_start = max(0, start_frame - padding_frames)
            padded_end = min(self.frame_count, end_frame + padding_frames)
            
            print(f"Processing highlight segment {segment_idx+1}/{len(highlight_segments)}: "
                 f"Frames {padded_start}-{padded_end} (duration: {(padded_end-padded_start)/self.fps:.2f}s)")
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, padded_start)
            
            for _ in range(padded_start, padded_end):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                cv2.putText(frame, f"Highlight {segment_idx+1}", (30, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                out.write(frame)
                
        cap.release()
        out.release()
        
        print(f"Highlight video created at: {output_path}")
        print(f"Total highlight segments: {len(highlight_segments)}")
        
        total_frames = sum([end - start + 2 * padding_frames for start, end in highlight_segments])
        total_frames = min(total_frames, self.frame_count)
        total_duration = total_frames / self.fps
        
        print(f"Original video duration: {self.duration:.2f}s")
        print(f"Highlight video duration: {total_duration:.2f}s")
        print(f"Reduction: {100 * (1 - total_duration/self.duration):.1f}%")
    
    def evaluate_highlights(self, highlight_mask, gt_path):
        """Evaluate highlight detection against ground truth annotations"""
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)

        user_summary = np.array(gt_data['user_summary'], dtype=np.float32)

        if user_summary.shape[1] != self.frame_count:
            raise ValueError(f"Ground truth frames ({user_summary.shape[1]}) don't match video frames ({self.frame_count})")
        
        best_fscore = 0.0
        best_precision = 0.0
        best_recall = 0.0
        for i, gt in enumerate(user_summary):
            tp = np.logical_and(highlight_mask, gt).sum()
            fp = np.logical_and(highlight_mask, np.logical_not(gt)).sum()
            fn = np.logical_and(np.logical_not(highlight_mask), gt).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            fscore = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            if fscore > best_fscore:
                best_fscore = fscore
                
            if precision > best_precision:
                best_precision = precision

            if recall > best_recall:
                best_recall = recall

        print(f"\nEvaluation Results:")
        print(f"Best F-score: {best_fscore:.4f}")
        print(f"Best Precision: {best_precision:.4f}")
        print(f"Best Recall: {best_recall:.4f}")

def main(args):
    
    # For reproduction
    np.random.seed(42)

    video_path = args.video
    detector = HighlightDetector(video_path)
    
    weights = {
        'audio': args.rms,        # RMS
        'histogram': args.hist,   # Histogram Differences: Visual scene changes
    }
    
    audio_features, video_features, highlight_scores = detector.detect_highlights(weights)
    
    print("Plotting highlight curves...")
    threshold, highlight_mask = detector.plot_highlight_curve(audio_features, video_features, highlight_scores, weights)
    print("Visualizations saved as 'highlight_analysis.png'")

    if args.ground_truth:
        detector.evaluate_highlights(highlight_mask, args.ground_truth)
    
    if args.save:
        print("Creating highlight video...")
        highlight_segments = detector.identify_highlight_segments(highlight_mask, min_segment_length=args.min_segment_frames)
        detector.create_highlight_video(args.video, highlight_segments, output_path=args.output, padding_seconds=args.padding_seconds)
        print(f"Highlight video saved as {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--rms", default=0.5, type=float, help="Weight of RMS energy")
    parser.add_argument("--hist", default=0.5, type=float, help="Weight of histogram")
    parser.add_argument("--save", action="store_true", help="Create highlight video")
    parser.add_argument("--output", default="output_video.mp4", help="Output highlight video path")
    parser.add_argument("--min_segment_frames", default=15, type=int, help="Minimum number of frames to consider as a highlight segment")
    parser.add_argument("--padding_seconds", default=1.5, type=float, help="Padding in seconds to add before and after each highlight segment")
    parser.add_argument("--ground_truth", type=str, help="Path to ground truth JSON file")
    args = parser.parse_args()
    main(args)