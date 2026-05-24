# 🚗 DAVE-2 Edge AI — Self-Driving Car System
### IU International University · Course DLBAIPEAI · Project Edge AI

[![Model](https://img.shields.io/badge/Model-DAVE--2%20CNN-00ff9f?style=flat-square)](https://arxiv.org/pdf/1604.07316.pdf)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow.js-orange?style=flat-square)](https://www.tensorflow.org/js)
[![Edge](https://img.shields.io/badge/Edge-iPhone%20Safari-blue?style=flat-square)](https://developer.apple.com/safari/)
[![Live](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-green?style=flat-square)](https://yerkebttt.github.io/driving-app)

> End-to-end deep learning for autonomous driving, deployed directly on a mobile device (iPhone) using TensorFlow.js — no cloud, no server, pure Edge AI.

---

## 📱 Live Demo

**Open on your iPhone Safari:**  
👉 `https://yerkebttt.github.io/driving-app`

> Allow camera and microphone access when prompted.

---

## 🎯 Project Overview

This project implements **Task 3: A Self-Driving Car Simulated System** from the IU course DLBAIPEAI (Project Edge AI).

The system:
1. Trains NVIDIA's **DAVE-2 CNN** on the Udacity self-driving car dataset
2. Converts the model to **TensorFlow.js** format
3. Deploys it as a **web application running entirely on iPhone Safari** (Edge AI)
4. Uses the **iPhone camera** as input and **Web Speech API** for voice control
5. Calculates the **autonomy grade** on real road videos

---

## 🏗️ Architecture

```
Camera Frame (iPhone)
        │
        ▼
  Preprocessing
  ┌─────────────────────────┐
  │  Crop rows 60-135       │
  │  Resize → 200×66 px     │
  │  BGR → YUV color space  │
  │  Normalize: ÷127.5 – 1  │
  └─────────────────────────┘
        │
        ▼
  DAVE-2 CNN (TF.js)
  ┌─────────────────────────┐
  │  Lambda  (normalize)    │
  │  Conv2D  24 × 5×5 s2   │
  │  Conv2D  36 × 5×5 s2   │
  │  Conv2D  48 × 5×5 s2   │
  │  Conv2D  64 × 3×3      │
  │  Conv2D  64 × 3×3      │
  │  Flatten               │
  │  Dense 100 + Dropout   │
  │  Dense 50              │
  │  Dense 10              │
  │  Dense 1 (output)      │
  └─────────────────────────┘
        │
        ▼
  Steering Angle [-1.0 … +1.0]
        │
        ▼
  HUD Overlay + Voice Control
```

**Total Parameters:** 252,219 (985 KB)

---

## 📊 Results

### Training Performance

| Platform | Training Time | Final MSE | Final MAE |
|---|---|---|---|
| CPU (estimated) | ~10.5 hours | 0.0041 | 0.0401 |
| GPU (Google Colab T4) | **0.7 minutes** | 0.0041 | 0.0401 |
| Edge Device (iPhone Safari) | inference only | — | — |

### Autonomy Grade Results

Evaluated using the formula from Bojarski et al. (2016):

$$\text{autonomy} = \left(1 - \frac{\text{interventions} \times 6}{\text{elapsed seconds}}\right) \times 100$$

| Video | Conditions | Duration | Interventions | Autonomy |
|---|---|---|---|---|
| 1 | Daytime | 60s | — | —% |
| 2 | Daytime | 60s | — | —% |
| 3 | Nighttime | 60s | — | —% |
| 4 | Nighttime | 60s | — | —% |
| **Average** | | | | **—%** |
| Bojarski et al. (2016) | Real roads (NJ) | — | — | **98%** |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Model Training | Python, TensorFlow/Keras, Google Colab (T4 GPU) |
| Model Format | TensorFlow.js (browser-compatible) |
| Edge Deployment | iPhone Safari, WebGL acceleration |
| Voice Control | Web Speech API (`webkitSpeechRecognition`) |
| Dataset | Udacity Self-Driving Car Dataset (8,036 images) |
| Frontend | Vanilla HTML/CSS/JS — no frameworks |

---

## 📁 Repository Structure

```
driving-app/
│
├── index.html          # Main application UI
├── style.css           # Dark HUD styling
├── app.js              # TF.js inference + voice control logic
│
└── tfjs_model/
    ├── model.json      # Model architecture
    └── group1-shard1of1.bin   # Trained weights (~985 KB)
```

---

## 🚀 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/yerkebttt/driving-app.git
cd driving-app

# Serve locally (required for camera/microphone access)
python -m http.server 8080

# Open in browser
# http://localhost:8080
```

> **Important:** The app requires HTTPS or localhost to access the camera. GitHub Pages provides HTTPS automatically.

---

## 🎤 Voice Commands

| Command | Action |
|---|---|
| **"left"** | Steer left (1.2 second override) |
| **"right"** | Steer right (1.2 second override) |
| **"stop"** / **"brake"** | Emergency stop |
| **"go"** / **"drive"** / **"start"** | Resume autonomous mode |
| **"straight"** / **"forward"** | Centre correction |

---

## 🔬 Training Details

**Dataset:** Udacity Self-Driving Car Simulator Dataset
- 8,036 center camera images
- Steering angles: min −0.94, max 1.0, mean 0.004
- 80% training / 20% validation split

**Training Configuration:**
- Optimizer: Adam (adaptive learning rate)
- Loss function: MSE (Mean Squared Error)
- Epochs: 20
- Batch size: 32
- Dropout: 0.5 (anti-overfitting)

**Preprocessing:**
- Crop rows 60–135 (remove sky and hood)
- Resize to 200×66 (NVIDIA DAVE-2 input size)
- Color: BGR → YUV (better lane detection)
- Normalize: pixel ÷ 127.5 − 1.0

---

## 📚 References

- Bojarski, M., Del Testa, D., Dworakowski, D., Firner, B., Flepp, B., Goyal, P., Jackel, L. D., Monfort, M., Muller, U., Zhang, J., Zhang, X., Zhao, J., & Zieba, K. (2016). *End to end learning for self-driving cars*. arXiv. https://arxiv.org/pdf/1604.07316.pdf

- Mikhailiuk, A. (2022). *On the edge — deploying deep learning applications on mobile*. Towards Data Science. https://towardsdatascience.com/on-the-edge-deploying-deep-applications-on-constrained-devices-f2dac997dd4d

- Apple Inc. (n.d.). *Machine learning*. Apple Developer. https://developer.apple.com/machine-learning/

- SIP-Lab. (n.d.). *Deep-learning-mobile* [GitHub repository]. GitHub. https://github.com/SIP-Lab/Deep-Learning-Mobile

---

## 🤖 AI Tools Statement

This project was developed with assistance from Claude (Anthropic) and ChatGPT (OpenAI) for code generation, debugging, and technical explanation. All experimental design, data collection, analysis, and report writing represent the student's own academic work.

---

## 👩‍💻 Author

**Yerkezhan** · IU International University of Applied Sciences  
Course: DLBAIPEAI – Project Edge AI · 2026

[![GitHub](https://img.shields.io/badge/GitHub-yerkebttt-181717?style=flat-square&logo=github)](https://github.com/yerkebttt)
