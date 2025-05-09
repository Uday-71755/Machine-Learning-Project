# metrics.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime

class TrainingMetrics:
    def __init__(self):
        self.scores = []
        self.episode_lengths = []
        self.losses = []
        self.timestamps = []
        self.start_time = datetime.now()
        
    def record_episode(self, score, length, loss=None):
        self.scores.append(score)
        self.episode_lengths.append(length)
        if loss is not None:
            self.losses.append(loss)
        self.timestamps.append((datetime.now() - self.start_time).total_seconds())
        
    def save_training_curves(self):
        os.makedirs('metrics_plots', exist_ok=True)
        
        plt.figure(figsize=(15, 5))
        
        # Score plot
        plt.subplot(1, 3, 1)
        plt.plot(self.scores)
        plt.title('Training Scores')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        
        # Loss plot
        if self.losses:
            plt.subplot(1, 3, 2)
            plt.plot(self.losses)
            plt.title('Training Loss')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
        
        # Length plot
        plt.subplot(1, 3, 3)
        plt.plot(self.episode_lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Frames')
        
        plt.tight_layout()
        plt.savefig('metrics_plots/training_curves.png')
        plt.close()
        
    def get_summary_stats(self):
        return {
            'total_episodes': len(self.scores),
            'max_score': max(self.scores) if self.scores else 0,
            'mean_score': np.mean(self.scores) if self.scores else 0,
            'median_score': np.median(self.scores) if self.scores else 0,
            'training_time_sec': self.timestamps[-1] if self.timestamps else 0,
            'last_100_avg': np.mean(self.scores[-100:]) if len(self.scores) >= 100 else 0
        }
    
    def save_to_csv(self):
        df = pd.DataFrame({
            'episode': range(1, len(self.scores)+1),
            'score': self.scores,
            'length': self.episode_lengths,
            'timestamp': self.timestamps
        })
        if self.losses:
            df['loss'] = self.losses[:len(self.scores)]  # Align lengths
            
        os.makedirs('metrics_data', exist_ok=True)
        path = 'metrics_data/training_metrics.csv'
        df.to_csv(path, index=False)
        return path