# 🏋️ Hybrid Reinforcement Learning for CartPole

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-CartPole-green)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Comparing Hill Climbing, Double Deep Q-Networks (DDQN), and a novel Hybrid approach for solving the classic CartPole problem.**

</div>  

---

## 🌟 Overview

This project explores **different reinforcement learning strategies** to master the `CartPole-v1` environment.
The unique contribution is a **Hybrid RL technique**: bootstrapping a DDQN with experience from a simple Hill Climbing agent, accelerating learning and improving stability.

### ✨ Highlights

* 🎯 **Algorithmic Comparison**: Hill Climbing, DDQN, Hybrid (Hill Climbing + DDQN)
* 🚀 **Hybrid Warm-Start**: Replay buffer pre-filled with Hill Climbing trajectories
* 📊 **Analytics & Visualization**: Matplotlib plots comparing rewards & accuracy
* 🖥️ **Live Rendering**: Watch the best trained agent perform via Pygame

---

## 📂 Project Structure

```plaintext
CartPole-Hybrid-RL/
├── code.py                         # Main script (training + evaluation + visualization)
├── Project_Overview_and_Literature_Review.pdf
├── Report.pdf
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
git clone https://github.com/jayavanth18/Hybrid-RL-CartPole.git
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

1. 🧗 Train **Hill Climbing agent**
2. 🧠 Train **DDQN agent** from scratch
3. 💡 Train **Hybrid DDQN** (warm-started with Hill Climbing data)
4. 📈 Plot **rewards & accuracy** comparisons
5. 🎥 Render **live agent performance** (Pygame)

---

## 📊 Results

<div align="center">

### Reward & Accuracy Comparison

![Reward](https://github.com/user-attachments/assets/4d1c4326-d61a-4780-b3b4-c671547a35ba)
![Episode Accuracy](https://github.com/user-attachments/assets/ee541a05-f48d-4cb7-91d1-6ff4856bffc1) <img width="1673" height="921" alt="Output" src="https://github.com/user-attachments/assets/5356c67d-c90d-4638-8deb-c3982d6af578" />

---

### Live Agent Demo

**Watch the final trained agent (Hybrid DDQN) successfully balancing the pole in real-time.**

![Live Demo of the Trained Agent](https://github.com/user-attachments/assets/879d8be0-8ea7-4244-8b6a-d0d68a3515e7)

*This visualization is rendered using Pygame and shows the agent's performance after completing the full training cycle.*

</div>  

---

## 🧠 Technical Details

### Algorithms

* **Hill Climbing** → A simple linear policy optimized by adding noise and adopting changes that improve rewards
* **DDQN** → A PyTorch-based neural network with two hidden layers (64 neurons each, ReLU activation)
* **Hybrid DDQN** → Replay buffer is pre-filled with experiences from the trained Hill Climbing agent

### Training Setup

* **Replay Memory**: `deque(maxlen=10,000)`
* **Optimizer**: Adam
* **Loss Function**: Mean Squared Error (MSE)
* **Exploration**: Epsilon-greedy, decaying from `1.0` → `0.01`
* **Hybrid Pre-fill**: \~50 episodes from Hill Climbing agent populate initial memory

---

## 🔮 Future Work

* [ ] Add PPO / A2C algorithms for a broader comparison
* [ ] Hyperparameter tuning with Optuna or Ray Tune
* [ ] Extend hybrid concept to environments like `LunarLander-v2`, `MountainCar-v0`
* [ ] Save/load trained model weights
* [ ] Web-based demo with Streamlit or Gradio

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m "Add AmazingFeature"`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request 🚀

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

* **OpenAI Gymnasium** for the CartPole environment
* **PyTorch** for the deep learning framework

---

## 👨‍💻 Author

<div align="center">

**[A. Jayavanth](https://github.com/jayavanth18)**

[![GitHub](https://img.shields.io/badge/GitHub-jayavanth18-black?logo=github)](https://github.com/jayavanth18)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/jayavanth18/)

⭐ If this repo helped you, consider giving it a **star**!

</div>  

---
