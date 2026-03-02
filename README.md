Link:(https://dnids-dashboard.onrender.com/)
# 🛡️ Distributed Network Intrusion Detection System (DNIDS)

## Using Apache Storm + Python + Docker

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Step-by-Step Setup](#step-by-step-setup)
6. [How It Works](#how-it-works)
7. [Running the System](#running-the-system)
8. [Understanding the Output](#understanding-the-output)
9. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

This project implements a **Distributed Network Intrusion Detection System (DNIDS)** that processes network traffic data in real-time to detect potential cyber attacks. It uses:

- **Apache Storm** (via Docker) for distributed real-time stream processing
- **Python** for all topology components (Spout + Bolts)
- **KDD Cup 99 Dataset** (41 network features) for training and testing
- **Decision Tree Classifier** (scikit-learn) for ML-based attack detection

### What is NIDS?

A **Network Intrusion Detection System (NIDS)** monitors network traffic and identifies suspicious activity that could indicate a cyber attack. Our system classifies each network connection as either:

- **Normal (0)** → Safe traffic ✅
- **Attack (1)** → Potential intrusion 🚨

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Apache Storm Cluster (Docker)                │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌─────────┐    ┌────────┐ │
│  │  KDD     │───▶│  Preprocess  │───▶│   ML    │───▶│ Alert  │ │
│  │  Spout   │    │    Bolt      │    │  Bolt   │    │  Bolt  │ │
│  │          │    │              │    │         │    │        │ │
│  │ Reads    │    │ Scales &     │    │ Predict │    │ Logs   │ │
│  │ CSV data │    │ Normalizes   │    │ using   │    │ alerts │ │
│  │ line by  │    │ 41 features  │    │ Decision│    │ for    │ │
│  │ line     │    │              │    │ Tree    │    │ attacks│ │
│  └──────────┘    └──────────────┘    └─────────┘    └────────┘ │
│                                                                 │
│  ┌───────────┐  ┌───────────┐  ┌────────────────┐              │
│  │ Zookeeper │  │  Nimbus   │  │  Supervisor(s) │              │
│  └───────────┘  └───────────┘  └────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

### Component Flow:

1. **KDD Spout** → Reads `kddcup.data_10_percent.csv` line-by-line, emits raw CSV rows
2. **Preprocess Bolt** → Receives raw data, encodes categorical features, scales numerical features using StandardScaler
3. **ML Bolt** → Loads pre-trained `decision_tree_model.pkl`, predicts Normal (0) or Attack (1)
4. **Alert Bolt** → If prediction = Attack, logs `🚨 INTRUSION DETECTED!` with details

---

## 📁 Project Structure

```
DNIDS/
├── README.md                          # This file
├── docker-compose.yml                 # Docker Storm cluster setup
├── requirements.txt                   # Python dependencies
│
├── dataset/
│   └── download_dataset.py            # Script to download KDD Cup 99 data
│
├── model/
│   ├── train_model.py                 # Train Decision Tree & save .pkl
│   └── decision_tree_model.pkl        # Pre-trained model (generated)
│
├── topology/
│   ├── ids_topology.py                # Main topology definition
│   ├── kdd_spout.py                   # Spout: reads CSV data
│   ├── preprocess_bolt.py             # Bolt: feature scaling/normalization
│   ├── ml_bolt.py                     # Bolt: ML prediction
│   └── alert_bolt.py                  # Bolt: intrusion alerting
│
├── simulation/
│   └── run_simulation.py              # Standalone simulation (no Storm needed)
│
└── docs/
    └── STEP_BY_STEP_GUIDE.md          # Detailed beginner guide
```

---

## ⚙️ Prerequisites

Before starting, make sure you have:

1. **Python 3.8+** installed
2. **Docker Desktop** installed and running
3. **pip** (Python package manager)

---

## 🚀 Step-by-Step Setup

### Step 1: Install Python Dependencies

```bash
cd d:\DNIDS
pip install -r requirements.txt
```

### Step 2: Download the KDD Cup 99 Dataset

```bash
python dataset/download_dataset.py
```

This downloads `kddcup.data_10_percent.csv` (~2MB compressed) from the UCI repository.

### Step 3: Train the ML Model

```bash
python model/train_model.py
```

This trains a **DecisionTreeClassifier** on the KDD data and saves:
- `model/decision_tree_model.pkl` (trained model)
- `model/scaler.pkl` (fitted StandardScaler)
- `model/label_encoders.pkl` (fitted LabelEncoders for categorical features)

### Step 4: Start the Docker Storm Cluster

```bash
docker-compose up -d
```

This starts:
- **Zookeeper** (port 2181) - Cluster coordination
- **Nimbus** (port 6627) - Storm master node
- **Supervisor** - Storm worker node
- **Storm UI** (port 8080) - Web dashboard

### Step 5: Run the Simulation

```bash
python simulation/run_simulation.py
```

This runs the complete IDS pipeline locally, simulating the Storm topology.

### Step 6: Access Storm UI

Open your browser and go to: **http://localhost:8080**

---

## 🔍 How It Works

### 1. Data Reading (KDD Spout)
The Spout reads the KDD Cup 99 dataset CSV file line-by-line. Each line contains 41 features describing a network connection, plus a label (e.g., "normal.", "smurf.", "neptune.").

### 2. Preprocessing (Preprocess Bolt)
- **Categorical features** (protocol_type, service, flag) are encoded using LabelEncoder
- **Numerical features** (38 features) are scaled using StandardScaler
- The label is converted to binary: `normal.` → 0, everything else → 1

### 3. ML Prediction (ML Bolt)
- Loads the pre-trained DecisionTreeClassifier
- Takes the 41 preprocessed features as input
- Outputs prediction: **0 (Normal)** or **1 (Attack)**

### 4. Alert Generation (Alert Bolt)
- If prediction == 1 → Logs `🚨 INTRUSION DETECTED!` with connection details
- If prediction == 0 → Logs `✅ Normal traffic`
- Maintains running statistics (total processed, attacks detected, accuracy)

---

## 📊 Understanding the Output

When running, you'll see output like:

```
============================================================
🛡️  DISTRIBUTED NETWORK INTRUSION DETECTION SYSTEM (DNIDS)
============================================================

[SPOUT] Emitting record #1: 0,tcp,http,SF,181,5450,...
[PREPROCESS] Scaled 41 features for record #1
[ML] Prediction for record #1: NORMAL (confidence: 0.98)
✅ [ALERT] Record #1: Normal traffic - no action needed

[SPOUT] Emitting record #2: 0,tcp,private,S0,0,0,...
[PREPROCESS] Scaled 41 features for record #2
[ML] Prediction for record #2: ATTACK (confidence: 0.95)
🚨 [ALERT] Record #2: INTRUSION DETECTED! Attack type suspected: neptune

────────────────────────────────────────
📊 Running Statistics:
   Total Processed: 100
   Normal:          35 (35.0%)
   Attacks:         65 (65.0%)
   Model Accuracy:  97.2%
────────────────────────────────────────
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Docker containers won't start | Make sure Docker Desktop is running. Try `docker-compose down` then `docker-compose up -d` |
| Dataset download fails | Check internet connection. Try manual download from http://kdd.ics.uci.edu/databases/kddcup99/ |
| Model training is slow | The 10% dataset has ~500K records. Training takes 1-2 minutes on average hardware |
| Import errors | Run `pip install -r requirements.txt` to install all dependencies |
| Port 8080 in use | Change the Storm UI port in `docker-compose.yml` |

---

## 📚 References

- [KDD Cup 99 Dataset](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
- [Apache Storm Documentation](https://storm.apache.org/documentation.html)
- [Streamparse - Python for Storm](https://streamparse.readthedocs.io/)
- [Docker Storm by denverdino](https://github.com/denverdino/docker-storm)
- [Scikit-learn Decision Trees](https://scikit-learn.org/stable/modules/tree.html)

---

**Made with ❤️ for Network Security**
#   D N I D S 
 
 
