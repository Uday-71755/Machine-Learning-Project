import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import seaborn as sns
from collections import defaultdict

class GameEvaluator:
    def __init__(self):
        self.true_actions = []
        self.predicted_actions = []
        self.predicted_probs = []
        self.class_names = ['Left', 'Straight', 'Right']
        self.action_counts = defaultdict(int)

    def record_prediction(self, true_action, predicted_action, probs):
        self.true_actions.append(true_action)
        self.predicted_actions.append(predicted_action)
        self.predicted_probs.append(probs)
        self.action_counts[predicted_action] += 1

    def generate_reports(self):
        reports = {}
        
        # Ensure we always have all classes represented
        unique_classes = set(self.true_actions + self.predicted_actions)
        labels = sorted(unique_classes)
        
        reports['classification'] = classification_report(
            self.true_actions,
            self.predicted_actions,
            labels=labels,
            target_names=[self.class_names[i] for i in labels],
            output_dict=True,
            zero_division=0
        )
        reports['accuracy'] = accuracy_score(self.true_actions, self.predicted_actions)
        reports['action_distribution'] = dict(self.action_counts)
        
        # Plot action distribution
        self.plot_action_distribution()
        
        # Plot confusion matrix if we have predictions
        if len(self.true_actions) > 0:
            self.plot_confusion_matrix(self.true_actions, self.predicted_actions)
        
        return reports

    def plot_action_distribution(self):
        plt.figure(figsize=(8, 6))
        # Ensure all actions are represented even if count is 0
        action_counts = {i: self.action_counts.get(i, 0) for i in range(3)}
        pd.Series(action_counts).plot(kind='bar')
        plt.title('Action Distribution')
        plt.xlabel('Action')
        plt.ylabel('Count')
        plt.xticks(ticks=[0, 1, 2], labels=self.class_names, rotation=0)
        plt.tight_layout()
        
        import os
        os.makedirs('metrics', exist_ok=True)
        plt.savefig('metrics/action_distribution.png')
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[self.class_names[i] for i in sorted(set(y_true + y_pred))],
                    yticklabels=[self.class_names[i] for i in sorted(set(y_true + y_pred))])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('metrics/confusion_matrix.png')
        plt.close()