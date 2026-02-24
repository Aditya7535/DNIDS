"""
=============================================================
 DNIDS Web Dashboard - Flask Application
=============================================================
 A beautiful web-based dashboard for the Distributed Network
 Intrusion Detection System. Runs the ML pipeline and shows
 real-time intrusion detection results.

 Endpoints:
   /              - Main dashboard
   /api/simulate  - Run simulation (POST)
   /api/stats     - Get current statistics
   /api/predict   - Predict single record
   /api/sample    - Get sample records from dataset
=============================================================
"""

import os
import sys
import json
import time
import random
import threading
import numpy as np
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from topology.kdd_spout import KDDSpout
from topology.preprocess_bolt import PreprocessBolt
from topology.ml_bolt import MLBolt
from topology.alert_bolt import AlertBolt

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────
simulation_state = {
    'running': False,
    'progress': 0,
    'total_records': 0,
    'records_processed': 0,
    'attacks_detected': 0,
    'normals_detected': 0,
    'correct_predictions': 0,
    'accuracy': 0.0,
    'precision': 0.0,
    'recall': 0.0,
    'true_positives': 0,
    'false_positives': 0,
    'true_negatives': 0,
    'false_negatives': 0,
    'recent_alerts': [],
    'attack_types': {},
    'timeline': [],
    'start_time': None,
    'elapsed': 0,
    'rate': 0,
}

# Initialize components (lazy loading)
_spout = None
_preprocess = None
_ml_bolt = None
_alert_bolt = None
_components_ready = False


def init_components():
    """Initialize all topology components."""
    global _preprocess, _ml_bolt, _components_ready

    if _components_ready:
        return True

    try:
        # Check if dataset exists
        dataset_path = os.path.join(PROJECT_ROOT, 'dataset', 'kddcup.data_10_percent.csv')
        if not os.path.exists(dataset_path):
            # Try to download
            from dataset.download_dataset import download_dataset
            download_dataset()

        # Check if model exists
        model_path = os.path.join(PROJECT_ROOT, 'model', 'decision_tree_model.pkl')
        if not os.path.exists(model_path):
            # Try to train
            from model.train_model import main as train_main
            train_main()

        _preprocess = PreprocessBolt()
        _preprocess.prepare()

        _ml_bolt = MLBolt()
        _ml_bolt.prepare()

        _components_ready = True
        return True
    except Exception as e:
        print(f"Error initializing components: {e}")
        return False


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main dashboard."""
    return render_template('index.html')


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'project': 'DNIDS',
        'timestamp': datetime.now().isoformat(),
        'components_ready': _components_ready,
    })


@app.route('/api/init', methods=['POST'])
def initialize():
    """Initialize ML components."""
    success = init_components()
    return jsonify({
        'success': success,
        'message': 'Components initialized' if success else 'Failed to initialize'
    })


@app.route('/api/stats')
def get_stats():
    """Return current simulation statistics."""
    return jsonify(simulation_state)


@app.route('/api/simulate', methods=['POST'])
def run_simulation():
    """Start a simulation run."""
    global simulation_state

    if simulation_state['running']:
        return jsonify({'error': 'Simulation already running'}), 400

    data = request.get_json() or {}
    num_records = min(data.get('records', 500), 5000)  # Cap at 5000 for web

    # Initialize components if needed
    if not init_components():
        return jsonify({'error': 'Failed to initialize ML components'}), 500

    # Start simulation in background thread
    thread = threading.Thread(target=_run_simulation_worker, args=(num_records,))
    thread.daemon = True
    thread.start()

    return jsonify({'message': f'Simulation started with {num_records} records', 'records': num_records})


def _run_simulation_worker(num_records):
    """Background worker for simulation."""
    global simulation_state

    # Reset state
    simulation_state = {
        'running': True,
        'progress': 0,
        'total_records': num_records,
        'records_processed': 0,
        'attacks_detected': 0,
        'normals_detected': 0,
        'correct_predictions': 0,
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'true_positives': 0,
        'false_positives': 0,
        'true_negatives': 0,
        'false_negatives': 0,
        'recent_alerts': [],
        'attack_types': {},
        'timeline': [],
        'start_time': time.time(),
        'elapsed': 0,
        'rate': 0,
    }

    # Attack type descriptions
    attack_types_map = {
        'neptune.': 'DOS - Neptune (SYN Flood)',
        'smurf.': 'DOS - Smurf (ICMP Flood)',
        'back.': 'DOS - Back (Apache Attack)',
        'teardrop.': 'DOS - Teardrop',
        'pod.': 'DOS - Pod (Ping of Death)',
        'land.': 'DOS - Land (IP Spoofing)',
        'satan.': 'PROBE - Satan Scan',
        'ipsweep.': 'PROBE - IP Sweep',
        'portsweep.': 'PROBE - Port Sweep',
        'nmap.': 'PROBE - Nmap Scan',
        'warezclient.': 'R2L - Warezmaster Client',
        'guess_passwd.': 'R2L - Password Guessing',
        'buffer_overflow.': 'U2R - Buffer Overflow',
        'rootkit.': 'U2R - Rootkit',
    }

    try:
        spout = KDDSpout(max_records=num_records)
        spout.open()

        preprocess = PreprocessBolt()
        preprocess.prepare()

        ml_bolt = MLBolt()
        ml_bolt.prepare()

        batch_attacks = 0
        batch_normals = 0
        batch_size = max(1, num_records // 50)  # ~50 timeline points

        for i in range(num_records):
            # Get next record
            raw_tuple = spout.next_tuple()
            if raw_tuple is None:
                break

            # Preprocess
            preprocessed = preprocess.process(raw_tuple)
            if preprocessed is None:
                continue

            # ML Prediction
            result = ml_bolt.process(preprocessed)
            if result is None:
                continue

            prediction = result['prediction']
            confidence = result['confidence']
            original_label = result['original_label'].strip()
            actual_binary = 0 if original_label == 'normal.' else 1

            # Update stats
            simulation_state['records_processed'] = i + 1
            simulation_state['progress'] = int((i + 1) / num_records * 100)

            if prediction == 1:
                simulation_state['attacks_detected'] += 1
                batch_attacks += 1

                # Track attack type
                attack_name = attack_types_map.get(original_label, original_label)
                simulation_state['attack_types'][attack_name] = \
                    simulation_state['attack_types'].get(attack_name, 0) + 1

                # Add to recent alerts (keep last 50)
                alert = {
                    'record_id': result['record_id'],
                    'confidence': round(confidence, 4),
                    'attack_type': attack_name,
                    'actual_label': original_label,
                    'correct': prediction == actual_binary,
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                }
                simulation_state['recent_alerts'].insert(0, alert)
                if len(simulation_state['recent_alerts']) > 50:
                    simulation_state['recent_alerts'] = simulation_state['recent_alerts'][:50]
            else:
                simulation_state['normals_detected'] += 1
                batch_normals += 1

            # True/False Positives/Negatives
            if prediction == 1 and actual_binary == 1:
                simulation_state['true_positives'] += 1
            elif prediction == 1 and actual_binary == 0:
                simulation_state['false_positives'] += 1
            elif prediction == 0 and actual_binary == 0:
                simulation_state['true_negatives'] += 1
            elif prediction == 0 and actual_binary == 1:
                simulation_state['false_negatives'] += 1

            if prediction == actual_binary:
                simulation_state['correct_predictions'] += 1

            # Calculate metrics
            total = simulation_state['records_processed']
            simulation_state['accuracy'] = round(
                simulation_state['correct_predictions'] / max(1, total) * 100, 2
            )
            tp = simulation_state['true_positives']
            fp = simulation_state['false_positives']
            fn = simulation_state['false_negatives']
            simulation_state['precision'] = round(tp / max(1, tp + fp) * 100, 2)
            simulation_state['recall'] = round(tp / max(1, tp + fn) * 100, 2)

            # Timing
            elapsed = time.time() - simulation_state['start_time']
            simulation_state['elapsed'] = round(elapsed, 1)
            simulation_state['rate'] = round(total / max(0.001, elapsed), 0)

            # Timeline data point
            if (i + 1) % batch_size == 0 or i == num_records - 1:
                simulation_state['timeline'].append({
                    'record': i + 1,
                    'attacks': batch_attacks,
                    'normals': batch_normals,
                    'accuracy': simulation_state['accuracy'],
                })
                batch_attacks = 0
                batch_normals = 0

            # Small delay for web visualization smoothness
            if num_records <= 200:
                time.sleep(0.02)

        spout.close()

    except Exception as e:
        print(f"Simulation error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        simulation_state['running'] = False
        simulation_state['progress'] = 100


@app.route('/api/predict', methods=['POST'])
def predict_single():
    """Predict a single network record."""
    if not init_components():
        return jsonify({'error': 'Components not initialized'}), 500

    data = request.get_json()
    raw_data = data.get('raw_data', '')

    if not raw_data:
        return jsonify({'error': 'No raw_data provided'}), 400

    try:
        # Process through pipeline
        input_tuple = {'raw_data': raw_data, 'record_id': 0}
        preprocessed = _preprocess.process(input_tuple)
        if preprocessed is None:
            return jsonify({'error': 'Preprocessing failed'}), 400

        result = _ml_bolt.process(preprocessed)
        if result is None:
            return jsonify({'error': 'Prediction failed'}), 400

        return jsonify({
            'prediction': 'Attack' if result['prediction'] == 1 else 'Normal',
            'prediction_code': result['prediction'],
            'confidence': round(result['confidence'], 4),
            'original_label': result['original_label'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sample')
def get_sample():
    """Get sample records from the dataset."""
    dataset_path = os.path.join(PROJECT_ROOT, 'dataset', 'kddcup.data_10_percent.csv')
    if not os.path.exists(dataset_path):
        return jsonify({'error': 'Dataset not found'}), 404

    samples = []
    try:
        with open(dataset_path, 'r') as f:
            lines = f.readlines()

        # Pick random samples
        indices = random.sample(range(min(len(lines), 10000)), min(10, len(lines)))
        for idx in indices:
            line = lines[idx].strip()
            fields = line.split(',')
            if len(fields) >= 42:
                samples.append({
                    'raw_data': line,
                    'protocol': fields[1],
                    'service': fields[2],
                    'flag': fields[3],
                    'label': fields[41],
                    'index': idx,
                })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'samples': samples})


@app.route('/api/model-info')
def model_info():
    """Get model metadata."""
    metadata_path = os.path.join(PROJECT_ROOT, 'model', 'model_metadata.txt')
    model_path = os.path.join(PROJECT_ROOT, 'model', 'decision_tree_model.pkl')

    info = {
        'model_type': 'DecisionTreeClassifier',
        'features': 41,
        'classification': 'Binary (Normal / Attack)',
        'dataset': 'KDD Cup 99 (10%)',
        'records': '~494,021',
        'model_exists': os.path.exists(model_path),
    }

    if os.path.exists(model_path):
        info['model_size'] = f"{os.path.getsize(model_path) / 1024:.1f} KB"

    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            info['metadata_raw'] = f.read()

    return jsonify(info)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    print(f"\n{'=' * 60}")
    print(f"  DNIDS Web Dashboard")
    print(f"  http://localhost:{port}")
    print(f"{'=' * 60}\n")
    app.run(host='0.0.0.0', port=port, debug=debug)
