import cv2
import numpy as np
import json
import argparse

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

        return None
    
    def extract_video_features(self):

        return None
    
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
        gt_c = None
        for i, gt in enumerate(user_summary):
            tp = np.logical_and(highlight_mask, gt).sum()
            fp = np.logical_and(highlight_mask, np.logical_not(gt)).sum()
            fn = np.logical_and(np.logical_not(highlight_mask), gt).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            fscore = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            if fscore > best_fscore:
                best_fscore = fscore
                gt_c = gt
            if precision > best_precision:
                best_precision = precision

            if recall > best_recall:
                best_recall = recall

        print(f"\nEvaluation Results:")
        print(f"Best F-score: {best_fscore:.4f}")
        print(f"Best Precision: {best_precision:.4f}")
        print(f"Best Recall: {best_recall:.4f}")

        return gt_c
    
def visualize_pred_gt(predicted, ground_truth, output_path="comparison.png"):
    """
    Visualize predicted moments and ground truth moments as horizontal bars.

    Parameters:
    ----------
    predicted : numpy.ndarray
        Binary array indicating predicted highlight moments (1 for highlight, 0 otherwise).
    ground_truth : numpy.ndarray
        Binary array indicating ground truth highlight moments (1 for highlight, 0 otherwise).
    output_path : str
        Path to save the visualization image.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    

    fig, ax = plt.subplots(figsize=(12, 2))
    
    gt_transitions = np.diff(np.concatenate([[0], ground_truth, [0]]))
    gt_starts = np.where(gt_transitions == 1)[0]
    gt_ends = np.where(gt_transitions == -1)[0]
    

    pred_transitions = np.diff(np.concatenate([[0], predicted, [0]]))
    pred_starts = np.where(pred_transitions == 1)[0]
    pred_ends = np.where(pred_transitions == -1)[0]
    
    for start, end in zip(gt_starts, gt_ends):
        width = end - start
        ax.barh(y=1, width=width, left=start, color='green', height=0.4)
    for start, end in zip(pred_starts, pred_ends):
        width = end - start
        ax.barh(y=0, width=width, left=start, color='orange', height=0.4)
    
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Predictions', 'Ground Truth'])
    ax.set_xlabel('Frames')
    ax.set_title('Visual Comparison of video summarization')
    
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

def main(args):

    np.random.seed(42)

    video_path = args.video
    detector = HighlightDetector(video_path)
    # highlight_mask = None
    highlight_mask = np.zeros(detector.frame_count, dtype=bool)  # Initialize as an empty mask

    if args.ground_truth:
        gt_c = detector.evaluate_highlights(highlight_mask, args.ground_truth)
        visualize_pred_gt(highlight_mask, gt_c)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--ground_truth", type=str, help="Path to ground truth JSON file")
    args = parser.parse_args()
    main(args)