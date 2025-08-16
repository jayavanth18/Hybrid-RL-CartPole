# 🏋️ Hybrid Reinforcement Learning for CartPole

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-CartPole-green)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Comparing Hill Climbing, Double Deep Q-Networks (DDQN), and a novel Hybrid approach for solving the classic CartPole problem.**

🎥 [Demo Video (Coming Soon)](#)

</div>  

---

## 🌟 Overview

This project explores **different reinforcement learning strategies** to master the `CartPole-v1` environment.
The unique contribution is a **Hybrid RL technique**: bootstrapping a DDQN with experience from a simple Hill Climbing agent, accelerating learning and improving stability.

### ✨ Highlights

* 🎯 **Algorithmic Comparison**: Hill Climbing, DDQN, Hybrid (Hill Climbing + DDQN).
* 🚀 **Hybrid Warm-Start**: Replay buffer pre-filled with Hill Climbing trajectories.
* 📊 **Analytics & Visualization**: Matplotlib plots comparing rewards & accuracy.
* 🖥️ **Live Rendering**: Watch the best trained agent perform via Pygame.

---

## 📂 Project Structure

```plaintext
CartPole-Hybrid-RL/
├── code.py                         # Main script (training + evaluation + visualization)
├── Project_Overview_and_Literature_Review.pdf
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

* Python 3.9+
* pip package manager
* Git

```bash
# 1️⃣ Clone the repository
git clone https://github.com/YOUR_USERNAME/CartPole-Hybrid-RL.git
cd CartPole-Hybrid-RL

# 2️⃣ (Optional) Setup virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3️⃣ Install dependencies
pip install -r requirements.txt
```

**requirements.txt**

```
numpy
gym[classic_control]
matplotlib
pygame
torch
```

---

## 🚀 Usage

Run the full experiment (training + plotting + visualization):

```bash
python code.py
```

### Execution Flow

1. 🧗 Train **Hill Climbing agent**.
2. 🧠 Train **DDQN agent** from scratch.
3. 💡 Train **Hybrid DDQN** (warm-started with Hill Climbing data).
4. 📈 Plot **rewards & accuracy** comparisons.
5. 🎥 Render **live agent performance** (Pygame).

---

## 📊 Results

<div align="center">

### Reward & Accuracy Comparison

Hybrid (blue) often converges faster and outperforms DDQN (orange).

![Comparison 1](https://github.com/user-attachments/assets/e34c7303-5ea4-47ec-8fd2-c02a4f17ddb3)
![Comparison 2](https://github.com/user-attachments/assets/07601e5b-3eb3-4ce0-ad87-070e06681832)

### Live Agent Demo

*A GIF/video here would boost the README a lot.*

</div>  

---

## 🧠 Technical Details

### Algorithms

* **Hill Climbing** → Linear policy, optimized with noisy perturbations.
* **DDQN** → PyTorch model with two hidden layers (64 neurons, ReLU).
* **Hybrid DDQN** → Replay buffer pre-filled by Hill Climbing rollouts.

### Training Setup

* Replay Memory: `deque(maxlen=10,000)`
* Optimizer: Adam
* Loss: MSE
* Exploration: Epsilon-greedy (`1.0 → 0.01`)
* Hybrid Pre-fill: \~50 episodes from Hill Climbing agent

---

## 🔮 Future Work

* [ ] Add PPO / A2C baselines
* [ ] Hyperparameter tuning (Optuna/Ray Tune)
* [ ] Extend to harder environments (`LunarLander-v2`, `MountainCar-v0`)
* [ ] Save/load trained models
* [ ] Web UI demo (Streamlit/Gradio)

---

## 🤝 Contributing

1. Fork the repo
2. Create your branch (`feature/AmazingFeature`)
3. Commit changes (`git commit -m "Add AmazingFeature"`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open PR 🚀

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

* **OpenAI Gymnasium** for CartPole environment
* **PyTorch** for deep learning framework

---

## 👨‍💻 Author

<div align="center">

**[A. Jayavanth](https://github.com/jayavanth18)**

[![GitHub](https://img.shields.io/badge/GitHub-jayavanth18-black?logo=github)](https://github.com/jayavanth18)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/jayavanth18/)

⭐ If this repo helped you, consider giving it a **star**!

</div>  

---
