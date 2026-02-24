"""
=============================================================
 DNIDS Simulation Runner
=============================================================
 This script runs the complete DNIDS pipeline WITHOUT requiring
 a Storm cluster. It simulates the Storm topology locally.

 This is perfect for:
 - Testing the complete pipeline
 - Demonstrations and presentations
 - Development and debugging
 - Understanding the data flow

 The simulation:
 1. Downloads the KDD dataset (if needed)
 2. Trains the ML model (if not already trained)
 3. Runs the full topology pipeline
 4. Generates detailed statistics and reports

 Usage:
   python simulation/run_simulation.py
   python simulation/run_simulation.py --records 500 --delay 0.05
=============================================================
"""

import os
import sys
import time
import argparse
import io

# Fix Windows console encoding for Unicode output
if sys.platform == 'win32' and not isinstance(sys.stdout, io.TextIOWrapper):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

# Add parent directory to path
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, PROJECT_ROOT)


def print_banner():
    """Print the DNIDS banner."""
    banner = """
============================================================
  DISTRIBUTED NETWORK INTRUSION DETECTION SYSTEM (DNIDS)
============================================================

  [KDD Spout] --> [Preprocess Bolt] --> [ML Bolt] --> [Alert Bolt]

  Apache Storm + Python + DecisionTree Classifier
  Dataset: KDD Cup 99 (41 features, binary classification)
============================================================
"""
    print(banner)


def check_prerequisites():
    """Check that all prerequisites are met."""
    print("=" * 60)
    print("  Checking Prerequisites")
    print("=" * 60)
    
    all_good = True
    
    # Check 1: Python packages
    print("\n  [1/4] Python packages...")
    required_packages = ['numpy', 'pandas', 'sklearn', 'joblib']
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"    [OK] {pkg}")
        except ImportError:
            print(f"    [MISSING] {pkg} (run: pip install -r requirements.txt)")
            all_good = False
    
    # Check 2: Dataset
    print("\n  [2/4] KDD Cup 99 dataset...")
    dataset_path = os.path.join(PROJECT_ROOT, 'dataset', 'kddcup.data_10_percent.csv')
    if os.path.exists(dataset_path):
        size = os.path.getsize(dataset_path) / 1024 / 1024
        print(f"    [OK] Found ({size:.1f} MB)")
    else:
        print(f"    [!]  Not found - will download now...")
        try:
            from dataset.download_dataset import download_dataset
            download_dataset()
            print(f"    [OK] Downloaded successfully!")
        except Exception as e:
            print(f"    [FAIL] Download failed: {e}")
            all_good = False
    
    # Check 3: Trained model
    print("\n  [3/4] Trained ML model...")
    model_path = os.path.join(PROJECT_ROOT, 'model', 'decision_tree_model.pkl')
    scaler_path = os.path.join(PROJECT_ROOT, 'model', 'scaler.pkl')
    encoders_path = os.path.join(PROJECT_ROOT, 'model', 'label_encoders.pkl')
    
    if all(os.path.exists(p) for p in [model_path, scaler_path, encoders_path]):
        model_size = os.path.getsize(model_path) / 1024
        print(f"    [OK] Model found ({model_size:.1f} KB)")
    else:
        print(f"    [!]  Model not trained - training now...")
        try:
            from model.train_model import main as train_main
            train_main()
            print(f"    [OK] Model trained successfully!")
        except Exception as e:
            print(f"    [FAIL] Training failed: {e}")
            all_good = False
    
    # Check 4: Topology components
    print("\n  [4/4] Topology components...")
    components = {
        'KDD Spout': 'topology/kdd_spout.py',
        'Preprocess Bolt': 'topology/preprocess_bolt.py',
        'ML Bolt': 'topology/ml_bolt.py',
        'Alert Bolt': 'topology/alert_bolt.py',
        'IDS Topology': 'topology/ids_topology.py',
    }
    for name, path in components.items():
        full_path = os.path.join(PROJECT_ROOT, path)
        if os.path.exists(full_path):
            print(f"    [OK] {name}")
        else:
            print(f"    [MISSING] {name} ({path} not found)")
            all_good = False
    
    print(f"\n{'-' * 60}")
    if all_good:
        print("  [OK] All prerequisites met! Ready to run.")
    else:
        print("  [FAIL] Some prerequisites are missing. Fix the issues above.")
    print(f"{'-' * 60}")
    
    return all_good


def run_simulation(records=1000, delay=0.0, report_interval=500, quiet=False):
    """
    Run the DNIDS simulation.
    
    Args:
        records (int): Number of records to process (0 = all).
        delay (float): Delay between records (seconds).
        report_interval (int): Records between summary reports.
        quiet (bool): Reduce output verbosity.
    """
    from topology.ids_topology import IDSTopology
    
    print(f"\n{'=' * 60}")
    print(f"  Starting DNIDS Simulation")
    print(f"{'=' * 60}")
    print(f"  Records to process: {'all' if records == 0 else f'{records:,}'}")
    print(f"  Emit delay: {delay}s")
    print(f"  Report interval: every {report_interval} records")
    print(f"  Verbose: {'No' if quiet else 'Yes'}")
    print(f"{'=' * 60}\n")
    
    # Small pause for user to read
    time.sleep(1)
    
    # Build and run topology
    topology = IDSTopology(name="dnids-simulation")
    topology.build(
        max_records=records,
        emit_delay=delay,
        report_interval=report_interval,
        verbose=not quiet
    )
    topology.run()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='DNIDS Simulation Runner - Run the IDS pipeline locally',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (100 records)
  python simulation/run_simulation.py --records 100

  # Demo mode (slow, 50 records)
  python simulation/run_simulation.py --records 50 --delay 0.1

  # Full run (all ~494K records)
  python simulation/run_simulation.py --records 0

  # Production-like (quiet, all records)
  python simulation/run_simulation.py --records 0 --quiet
        """
    )
    
    parser.add_argument(
        '--records', '-n', type=int, default=1000,
        help='Records to process (0=all). Default: 1000'
    )
    parser.add_argument(
        '--delay', '-d', type=float, default=0.0,
        help='Delay between records (seconds). Default: 0.0'
    )
    parser.add_argument(
        '--report-interval', '-r', type=int, default=500,
        help='Records between reports. Default: 500'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Reduce output verbosity'
    )
    parser.add_argument(
        '--skip-checks', action='store_true',
        help='Skip prerequisite checks'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check prerequisites
    if not args.skip_checks:
        if not check_prerequisites():
            print("\n[FAIL] Cannot run simulation. Fix prerequisites first.")
            sys.exit(1)
    
    # Run simulation
    run_simulation(
        records=args.records,
        delay=args.delay,
        report_interval=args.report_interval,
        quiet=args.quiet
    )


if __name__ == "__main__":
    main()
