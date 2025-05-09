import torch
import random
import pandas as pd
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model1 import Linear_QNet, QTrainer
from evalution import GameEvaluator
from metrics import TrainingMetrics
import torch.nn.functional as F
import json
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 80  # Start with higher randomness
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.best_score = -float('inf')
        self.evaluator = GameEvaluator()
        self.metrics = TrainingMetrics()

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=np.float32)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(np.array(states), actions, rewards, np.array(next_states), dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(np.array([state]), [action], [reward], np.array([next_state]), [done])

    def get_action(self, state, eval_mode=False):
        # Linear epsilon decay
        self.epsilon = max(0, 80 - self.n_games)
        
        final_move = [0, 0, 0]
        
        # Exploration (random action)
        if not eval_mode and random.random() < (self.epsilon / 100):
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Exploitation (model prediction)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                prediction = self.model(state_tensor)
                probs = F.softmax(prediction, dim=1).numpy()[0]
            
            # During evaluation, we want to record the model's confidence
            if eval_mode:
                optimal_action = torch.argmax(prediction).item()
                self.evaluator.record_prediction(optimal_action, optimal_action, probs)
            
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            
        return final_move

    def evaluate(self, game, num_episodes=20):
        """Evaluation mode with metrics collection"""
        # Reset evaluator for fresh metrics
        self.evaluator = GameEvaluator()

        for _ in range(num_episodes):
            state = self.get_state(game)
            
            # In evaluation, we want to see the model's true predictions
            # without exploration noise
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                prediction = self.model(state_tensor)
                probs = F.softmax(prediction, dim=1).numpy()[0]
                move = torch.argmax(prediction).item()
            
            # Record both the predicted and "optimal" action (same in eval mode)
            self.evaluator.record_prediction(move, move, probs)
            
            # Convert to one-hot action
            action = [0, 0, 0]
            action[move] = 1
            
            _, done, score = game.play_step(action)
            self.metrics.record_episode(score, game.frame_iteration)
            if done:
                game.reset()
                
        reports = self.evaluator.generate_reports()
        
        # Save reports
        os.makedirs('metrics', exist_ok=True)
        with open('metrics/evaluation_report.json', 'w') as f:
            json.dump(reports, f, indent=2)
            
        print("\n=== Evaluation Results ===")
        print(f"Average Score: {np.mean(self.metrics.scores):.2f}")
        print("\nAction Distribution:")
        for i, count in reports['action_distribution'].items():
            print(f"{self.evaluator.class_names[i]}: {count}")
        
        if 'classification' in reports:
            print("\nClassification Report:")
            print(pd.DataFrame(reports['classification']).transpose())
        
        return reports

def train(num_episodes=500):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    
    agent = Agent()
    game = SnakeGameAI()
    
    try:
        for episode in range(1, num_episodes + 1):
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old)
            
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)
            
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)
            
            if done:
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()
                
                if score > agent.best_score:
                    agent.best_score = score
                    agent.model.save('best_model.pth')
                
                total_score += score
                mean_score = total_score / agent.n_games
                plot_scores.append(score)
                plot_mean_scores.append(mean_score)
                
                print(f'Episode {episode}, Score: {score}, Best: {agent.best_score}, ε: {agent.epsilon:.1f}')
                
        # Final evaluation and save metrics
        final_reports = agent.evaluate(game)
        agent.metrics.save_training_curves()
        agent.metrics.save_to_csv()
        
        return agent
        
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model...")
        agent.model.save('interrupted_model.pth')
        return agent

if __name__ == '__main__':
    trained_agent = train()