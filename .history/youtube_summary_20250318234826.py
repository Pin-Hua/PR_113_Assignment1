import cv2
import numpy as np
import json
import argparse
import moviepy.editor as mp
import os
import scipy.io.wavfile as wav
import trace

class HighlightDetector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video = cv2.VideoCapture(video_path)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps

    def extract_audio_features(self):
        """Extracts RMS energy and spectral contrast, ensuring they match video frame count."""
        audio_path = self.video_path.replace(".mp4", ".wav")
        video_clip = mp.VideoFileClip(self.video_path)
        video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le')

        sample_rate, audio_data = wav.read(audio_path)
        os.remove(audio_path)  # Clean up extracted audio

        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)  # Convert to mono if stereo

        frame_size = sample_rate // int(self.fps)  # Align with video frame rate
        energy = np.array([np.sqrt(np.mean(audio_data[i:i+frame_size]**2)) for i in range(0, len(audio_data), frame_size)])
        spectral_variance = np.array([np.var(audio_data[i:i+frame_size]) for i in range(0, len(audio_data), frame_size)])

        # Ensure audio features match the number of video frames
        energy = np.interp(np.linspace(0, len(energy) - 1, self.frame_count), np.arange(len(energy)), energy)
        spectral_variance = np.interp(np.linspace(0, len(spectral_variance) - 1, self.frame_count), np.arange(len(spectral_variance)), spectral_variance)

        return energy, spectral_variance

    def extract_video_features(self):
        """Extracts motion intensity using Optical Flow."""
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
        """Generates a summarized video of specified duration using motion and audio features."""
        motion_scores = self.extract_video_features()
        energy, spectral_contrast = self.extract_audio_features()
        combined_scores = motion_scores + energy + spectral_contrast

        # Select top frames based on combined scores
        top_segments = np.argsort(combined_scores)[-int((duration * self.fps)):]
        top_segments.sort()

        video_clip = mp.VideoFileClip(self.video_path)
        selected_clips = [video_clip.subclip(max(0, i / self.fps), min(self.duration, (i + 1) / self.fps)) for i in top_segments]
        final_clip = mp.concatenate_videoclips(selected_clips, method="compose")

        output_path = self.video_path.replace(".mp4", f"_summary_{duration}s.mp4")
        final_clip.write_videofile(output_path, codec="libx264")
        return output_path


def main(args):
    detector = HighlightDetector(args.video)
    summary_30s = detector.generate_summary(duration=30)
    summary_1min = detector.generate_summary(duration=60)

    print(f"Generated Summaries:\n30s: {summary_30s}\n1min: {summary_1min}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    args = parser.parse_args()
    main(args)

    tracer = trace.Trace(count=False, trace=True)  # 初始化 trace 來追蹤執行過程
    tracer.run('main(args)')  # 使用 trace 來執行主函式