import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import gym
import random
import time
import matplotlib.pyplot as plt
from collections import deque
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

# ------------------ Hill Climbing Policy ------------------
class HillClimbingPolicy:
    def _init_(self, state_dim, action_dim):
        self.w = np.random.randn(state_dim, action_dim)

    def act(self, state):
        return np.argmax(np.dot(state, self.w))

def train_hill_climbing(env, policy, episodes=300, threshold=180):
    print("\n🔧 Training with Hill Climbing...\n")
    best_R = -np.inf
    best_w = policy.w.copy()
    scores = []
    accuracies = []
    noise_scale = 1e-2

    for ep in range(episodes):
        state = env.reset()
        if isinstance(state, tuple): state = state[0]
        rewards = []
        for _ in range(500):
            action = policy.act(state)
            result = env.step(action)
            if len(result) == 4:
                state, reward, done, _ = result
            else:
                state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            rewards.append(reward)
            if done:
                break
        total_reward = sum(rewards)
        accuracy = total_reward / 500
        scores.append(total_reward)
        accuracies.append(accuracy)

        if total_reward > best_R:
            best_R = total_reward
            best_w = policy.w.copy()
            noise_scale = max(1e-3, noise_scale / 2)
            policy.w += noise_scale * np.random.randn(*policy.w.shape)
        else:
            noise_scale = min(2, noise_scale * 2)
            policy.w = best_w + noise_scale * np.random.randn(*policy.w.shape)

        print(f"HC Ep {ep+1} | Reward: {total_reward:.2f} | Accuracy: {accuracy*100:.2f}% | Best: {best_R:.2f}")

        if np.mean(scores[-100:]) >= threshold:
            print("✅ Hill Climbing converged!")
            break

    policy.w = best_w
    return policy, scores, accuracies

# ------------------ Q-Network and DDQN Agent ------------------
class QNetwork(nn.Module):
    def _init_(self, state_dim, action_dim):
        super(QNetwork, self)._init_()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class DDQNAgent:
    def _init_(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qnet = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.eps_decay = 0.995
        self.eps_min = 0.01
        self.action_dim = action_dim
        self.update_target()

    def update_target(self):
        self.target_net.load_state_dict(self.qnet.state_dict())

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            return self.qnet(state).argmax().item()

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) >= self.batch_size:
            self.learn()

    def learn(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.qnet(states).gather(1, actions)
        next_actions = self.qnet(next_states).argmax(1).unsqueeze(1)
        next_q_values = self.target_net(next_states).gather(1, next_actions).detach()
        target = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

def train_ddqn(env, agent, episodes=300, desc="DDQN"):
    print(f"\n🚀 Training {desc}...\n")
    scores = []
    accuracies = []
    for ep in range(episodes):
        state = env.reset()
        if isinstance(state, tuple): state = state[0]
        total_reward = 0
        for _ in range(500):
            action = agent.act(state)
            result = env.step(action)
            if len(result) == 4:
                next_state, reward, done, _ = result
            else:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        accuracy = total_reward / 500
        scores.append(total_reward)
        accuracies.append(accuracy)
        agent.update_target()
        print(f"{desc} Ep {ep+1} | Reward: {total_reward:.2f} | Accuracy: {accuracy*100:.2f}% | Eps: {agent.epsilon:.3f}")
        if np.mean(scores[-100:]) >= 195:
            print(f"✅ {desc} solved in {ep+1} episodes!")
            break
    return scores, accuracies

# ------------------ Pre-fill Replay Memory ------------------
def prefill_memory(agent, env, policy, episodes=50):
    for _ in range(episodes):
        state = env.reset()
        if isinstance(state, tuple): state = state[0]
        for _ in range(500):
            action = policy.act(state)
            result = env.step(action)
            if len(result) == 4:
                next_state, reward, done, _ = result
            else:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

# ------------------ Visualize Agent ------------------
def watch_trained_agent(env, agent, episodes=3):
    print("\n🎥 Watching trained agent...\n")
    for ep in range(episodes):
        state = env.reset()
        if isinstance(state, tuple): state = state[0]
        done = False
        total_reward = 0
        step_count = 0
        pygame.init()
        screen = pygame.display.set_mode((600, 400))
        font = pygame.font.SysFont('Arial', 20)

        while not done:
            time.sleep(0.02)
            action = agent.act(state)
            result = env.step(action)
            if len(result) == 4:
                next_state, reward, done, _ = result
            else:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            state = next_state
            total_reward += reward
            step_count += 1

            screen.fill((255, 255, 255))
            env.render()
            overlay = [
                f"Episode: {ep + 1}",
                f"Step: {step_count}",
                f"Action: {action}",
                f"Reward: {total_reward:.2f}",
                f"Status: {'Done' if done else 'Running'}"
            ]
            for i, line in enumerate(overlay):
                text = font.render(line, True, (0, 0, 0))
                screen.blit(text, (10, 10 + 25 * i))
            pygame.display.flip()
        print(f"🎯 Episode {ep+1}: Total Reward = {total_reward:.2f}")
    pygame.quit()

# ------------------ MAIN ------------------
env = gym.make('CartPole-v1', render_mode='human')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 1. Hill Climbing only
hc_policy = HillClimbingPolicy(state_dim, action_dim)
hc_policy, hc_scores, hc_accuracies = train_hill_climbing(env, hc_policy)

# 2. DDQN only (NO HC)
ddqn_agent_plain = DDQNAgent(state_dim, action_dim)
ddqn_scores_plain, ddqn_accuracies_plain = train_ddqn(env, ddqn_agent_plain, desc="DDQN (no HC)")

# 3. DDQN + Hill Climbing prefill
ddqn_agent_combo = DDQNAgent(state_dim, action_dim)
prefill_memory(ddqn_agent_combo, env, hc_policy)
ddqn_scores_combined, ddqn_accuracies_combined = train_ddqn(env, ddqn_agent_combo, desc="DDQN + HC")

# ------------------ REWARD Graphs ------------------
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.plot(hc_scores)
plt.title("Hill Climbing - Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.axhline(180, color="r", linestyle="--", label="HC Threshold")
plt.legend()
plt.grid()

plt.subplot(1, 3, 2)
plt.plot(ddqn_scores_plain, color="orange")
plt.title("DDQN Only - Reward")
plt.xlabel("Episode")
plt.axhline(195, color="g", linestyle="--", label="Solved")
plt.legend()
plt.grid()

plt.subplot(1, 3, 3)
plt.plot(ddqn_scores_combined, color="blue")
plt.title("DDQN + HC Pre-fill - Reward")
plt.xlabel("Episode")
plt.axhline(195, color="g", linestyle="--", label="Solved")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# ------------------ ACCURACY Graphs ------------------
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.plot(hc_accuracies)
plt.title("Hill Climbing - Accuracy")
plt.xlabel("Episode")
plt.ylabel("Accuracy")
plt.grid()

plt.subplot(1, 3, 2)
plt.plot(ddqn_accuracies_plain, color="orange")
plt.title("DDQN Only - Accuracy")
plt.xlabel("Episode")
plt.ylabel("Accuracy")
plt.grid()

plt.subplot(1, 3, 3)
plt.plot(ddqn_accuracies_combined, color="blue")
plt.title("DDQN + HC Pre-fill - Accuracy")
plt.xlabel("Episode")
plt.ylabel("Accuracy")
plt.grid()

plt.tight_layout()
plt.show()

# ------------------ OVERALL ACCURACY SUMMARY ------------------
overall_hc_accuracy = np.mean(hc_accuracies)
overall_ddqn_accuracy = np.mean(ddqn_accuracies_plain)
overall_combo_accuracy = np.mean(ddqn_accuracies_combined)

print("\n📊 ===== Overall Model Accuracy Summary =====")
print(f"🔹 Hill Climbing Only Accuracy     : {overall_hc_accuracy * 100:.2f}%")
print(f"🔹 DDQN Only (no HC) Accuracy      : {overall_ddqn_accuracy * 100:.2f}%")
print(f"🔹 DDQN + HC Prefill Accuracy      : {overall_combo_accuracy * 100:.2f}%")

# Final animation
watch_trained_agent(env, ddqn_agent_combo)