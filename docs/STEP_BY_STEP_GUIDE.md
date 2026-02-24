# 📖 DNIDS - Step-by-Step Beginner Guide

## Complete Walkthrough for Beginners

This guide explains every concept, file, and step in detail. No prior experience with Storm, ML, or Docker is assumed.

---

## 🧠 What Are We Building?

Imagine you're a security guard watching thousands of people enter a building through multiple doors simultaneously. You need to identify who's a visitor and who's an intruder — **in real-time**.

That's exactly what our **Network Intrusion Detection System (NIDS)** does, but for computer networks:
- **People** = Network connections (HTTP requests, FTP sessions, etc.)
- **Building doors** = Network ports
- **Security guard** = Our ML model (Decision Tree)
- **Multiple guards working together** = Apache Storm (distributed processing)

---

## 📚 Key Concepts Explained

### 1. Apache Storm (What & Why?)

**What:** Apache Storm is a distributed, real-time computation system. Think of it like a factory assembly line, but for data.

**Why:** When you have millions of network connections per second, one computer can't analyze them all. Storm splits the work across many computers.

**Key Terms:**
| Term | Meaning | Our Example |
|------|---------|-------------|
| **Topology** | The overall program/pipeline | Our IDS pipeline |
| **Spout** | Data source (reads data) | Reads CSV file |
| **Bolt** | Data processor (transforms data) | Preprocess, ML, Alert |
| **Tuple** | One piece of data flowing through | One network connection record |
| **Stream** | Flow of tuples between components | CSV row → features → prediction → alert |

### 2. KDD Cup 99 Dataset

The **KDD Cup 1999** dataset is the most famous dataset for network intrusion detection research. It contains:
- **~5 million** network connection records (we use the 10% subset: ~494K)
- **41 features** describing each connection
- **1 label** telling us if it's normal or which attack type

### 3. The 41 Features (Grouped)

#### Basic Features (1-9): Information about the connection itself
```
duration, protocol_type, service, flag, src_bytes, dst_bytes,
land, wrong_fragment, urgent
```

#### Content Features (10-22): Information about the data content
```
hot, num_failed_logins, logged_in, num_compromised, root_shell,
su_attempted, num_root, num_file_creations, num_shells,
num_access_files, num_outbound_cmds, is_host_login, is_guest_login
```

#### Traffic Features (23-41): Statistics about network traffic patterns
```
count, srv_count, serror_rate, srv_serror_rate, rerror_rate,
srv_rerror_rate, same_srv_rate, diff_srv_rate, srv_diff_host_rate,
dst_host_count, dst_host_srv_count, dst_host_same_srv_rate,
dst_host_diff_srv_rate, dst_host_same_src_port_rate,
dst_host_srv_diff_host_rate, dst_host_serror_rate,
dst_host_srv_serror_rate, dst_host_rerror_rate,
dst_host_srv_rerror_rate
```

### 4. Attack Types in KDD

| Category | Attacks | Description |
|----------|---------|-------------|
| **DOS** | smurf, neptune, back, teardrop, pod, land | Overwhelm the target with traffic |
| **R2L** | warezclient, guess_passwd, warezmaster | Remote attacker gains local access |
| **U2R** | buffer_overflow, rootkit, loadmodule | Local user gains root privileges |
| **PROBE** | satan, ipsweep, portsweep, nmap | Surveillance/scanning attacks |

### 5. Decision Tree Classifier

A **Decision Tree** is like a flowchart of yes/no questions:
```
Is src_bytes > 1000?
├── YES → Is service == 'http'?
│         ├── YES → NORMAL ✅
│         └── NO  → Is count > 50?
│                   ├── YES → ATTACK 🚨
│                   └── NO  → NORMAL ✅
└── NO  → Is flag == 'S0'?
          ├── YES → ATTACK 🚨
          └── NO  → NORMAL ✅
```

---

## 🔧 Step 1: Install Dependencies

```bash
# Navigate to project directory
cd d:\DNIDS

# Install Python packages
pip install -r requirements.txt
```

**What gets installed:**
- `numpy` - Math operations on arrays
- `pandas` - Data loading and manipulation
- `scikit-learn` - Machine learning (Decision Tree)
- `joblib` - Save/load ML models
- `matplotlib` - Visualization
- `requests` - HTTP downloads
- `colorama` - Colored terminal output

---

## 🔧 Step 2: Download the Dataset

```bash
python dataset/download_dataset.py
```

**What happens:**
1. Downloads `kddcup.data_10_percent.gz` (~2MB) from UCI repository
2. Extracts it to `dataset/kddcup.data_10_percent.csv` (~75MB)
3. Verifies the file (should have ~494,021 records)

**Sample CSV line:**
```
0,tcp,http,SF,181,5450,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,8,8,0.00,0.00,0.00,0.00,1.00,0.00,0.00,9,9,1.00,0.00,0.11,0.00,0.00,0.00,0.00,0.00,normal.
```

Each value maps to one of the 41 features, and the last value is the label.

---

## 🔧 Step 3: Train the Model

```bash
python model/train_model.py
```

**What happens:**
1. **Loads** the CSV dataset into a pandas DataFrame
2. **Encodes** categorical features (protocol_type, service, flag) using LabelEncoder
   - Example: `tcp` → 0, `udp` → 1, `icmp` → 2
3. **Converts labels** to binary: `normal.` → 0, everything else → 1
4. **Splits** data: 70% training, 30% testing
5. **Scales** features using StandardScaler (mean=0, std=1)
6. **Trains** DecisionTreeClassifier
7. **Evaluates** on test set (typically ~99% accuracy)
8. **Saves** three .pkl files:
   - `decision_tree_model.pkl` - The trained model
   - `scaler.pkl` - The feature scaler
   - `label_encoders.pkl` - The categorical encoders

---

## 🔧 Step 4: Run the Simulation

```bash
# Quick test (100 records)
python simulation/run_simulation.py --records 100

# Demo mode (slow, see each record)
python simulation/run_simulation.py --records 50 --delay 0.1

# Full run
python simulation/run_simulation.py --records 0
```

**What happens:**
The simulation runs the complete Storm-like pipeline locally:

```
Record from CSV → Spout → Preprocess Bolt → ML Bolt → Alert Bolt → Log
```

---

## 🔧 Step 5: Docker Storm Cluster

```bash
# Start the Storm cluster
docker-compose up -d

# Check status
docker-compose ps

# View Storm UI
# Open browser: http://localhost:8080

# View logs
docker-compose logs -f

# Stop the cluster
docker-compose down
```

**What gets started:**
| Container | Role | Port |
|-----------|------|------|
| dnids-zookeeper | Cluster coordination | 2181 |
| dnids-nimbus | Storm master node | 6627 |
| dnids-supervisor | Storm worker node | - |
| dnids-stormui | Web dashboard | 8080 |

---

## 📝 Code Walkthrough

### File 1: `kdd_spout.py` (Data Source)

```python
# The Spout reads CSV data line-by-line
# Key method: next_tuple()
#   - Reads one CSV line
#   - Returns: {'raw_data': "0,tcp,http,...", 'record_id': 1}
```

### File 2: `preprocess_bolt.py` (Feature Engineering)

```python
# The Preprocess Bolt transforms raw data
# Key method: process(input_tuple)
#   - Input: raw CSV string
#   - Encodes: tcp→0, http→1, SF→2 (categorical encoding)
#   - Scales: all 41 features to mean=0, std=1
#   - Returns: {'features': [0.1, -0.3, ...], 'original_label': 'normal.'}
```

### File 3: `ml_bolt.py` (ML Prediction)

```python
# The ML Bolt predicts Normal/Attack
# Key method: process(input_tuple)
#   - Input: 41 scaled features
#   - Runs model.predict() using DecisionTree
#   - Returns: {'prediction': 0 or 1, 'confidence': 0.95}
```

### File 4: `alert_bolt.py` (Alerting)

```python
# The Alert Bolt generates alerts
# Key method: process(input_tuple)
#   - Input: prediction result
#   - If attack: prints "🚨 INTRUSION DETECTED!"
#   - If normal: prints "✅ Normal traffic"
#   - Tracks: accuracy, precision, recall statistics
```

---

## 🎯 Expected Results

When you run the simulation, expect:
- **Model Accuracy:** ~99% (Decision Trees work very well on KDD data)
- **Attack Detection Rate:** ~65% of the 10% dataset are attacks
- **Processing Speed:** ~50,000+ records/second on modern hardware
- **Main Attack Types:** smurf (DOS) and neptune (DOS) are most common

---

## ❓ FAQ

**Q: Do I need Docker to run this?**
A: No! The simulation (`run_simulation.py`) works without Docker. Docker is only needed for the actual Storm cluster deployment.

**Q: Why Decision Tree and not Deep Learning?**
A: Decision Trees are simple, fast, interpretable, and achieve ~99% accuracy on KDD data. Perfect for a beginner project. You can swap in any sklearn classifier later.

**Q: Can I use a different dataset?**
A: Yes! As long as your CSV has the same 41 features. The modern equivalent is the CICIDS dataset.

**Q: How does this compare to real-world IDS?**
A: Real IDS systems (like Snort, Suricata) use rule-based + ML approaches, process live network traffic (not CSV files), and handle much higher throughput.

---

## 🚀 Next Steps

1. **Try different ML models**: Random Forest, SVM, Neural Networks
2. **Add live packet capture**: Use `scapy` to capture real network traffic
3. **Deploy on actual Storm cluster**: Use `streamparse` for real Storm deployment
4. **Add a dashboard**: Build a real-time web dashboard with Flask/Streamlit
5. **Multi-class classification**: Instead of binary, predict specific attack types
