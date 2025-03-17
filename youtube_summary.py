import cv2
import numpy as np
import moviepy.editor as mp
import librosa
import librosa.display
import os

def extract_audio_features(video_path):
    video = mp.VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path, codec='pcm_s16le')
    
    y, sr = librosa.load(audio_path, sr=None)
    rms_energy = librosa.feature.rms(y=y)  # RMS 能量
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)  # 頻譜對比度
    
    os.remove(audio_path)
    return rms_energy, spectral_contrast

def extract_video_features(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    flow_magnitude = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_magnitude.append(np.sum(magnitude))
        prev_gray = gray
    
    cap.release()
    return flow_magnitude

def get_summary(video_path, duration=30):
    video = mp.VideoFileClip(video_path)
    total_duration = video.duration
    
    rms_energy, spectral_contrast = extract_audio_features(video_path)
    flow_magnitude = extract_video_features(video_path)
    
    importance_scores = np.array(flow_magnitude) + np.mean(rms_energy) + np.mean(spectral_contrast)
    top_indices = np.argsort(importance_scores)[-duration:]
    top_indices.sort()
    
    clips = []
    frame_rate = total_duration / len(flow_magnitude)
    
    for idx in top_indices:
        start_time = idx * frame_rate
        end_time = min(start_time + frame_rate, total_duration)
        clips.append(video.subclip(start_time, end_time))
    
    summary = mp.concatenate_videoclips(clips)
    output_filename = f"summary_{duration}sec.mp4"
    summary.write_videofile(output_filename, codec='libx264')
    
    return output_filename

# 主程式
video_path = "C:/Users/user/PR_113_Assignment1/videos/youtube.mp4"
summarized_30s = get_summary(video_path, duration=30)
summarized_60s = get_summary(video_path, duration=60)

print(f"30秒摘要視訊已儲存為: {summarized_30s}")
print(f"60秒摘要視訊已儲存為: {summarized_60s}")
