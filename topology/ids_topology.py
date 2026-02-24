"""
=============================================================
 IDS Topology - Storm Topology Definition
=============================================================
 This file defines the Apache Storm topology for the DNIDS.
 
 A topology is a DAG (Directed Acyclic Graph) that defines
 how data flows between components:

     KDD Spout → Preprocess Bolt → ML Bolt → Alert Bolt

 In a full Storm deployment (with streamparse), this would
 define the topology using the Storm Thrift API. For our
 Python implementation, this serves as the topology orchestrator.

 Usage:
   python topology/ids_topology.py [--records N] [--delay SECONDS]
=============================================================
"""

import os
import sys
import time
import argparse
import logging
import io

# Fix Windows console encoding
if sys.platform == 'win32' and not isinstance(sys.stdout, io.TextIOWrapper):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Import topology components
from topology.kdd_spout import KDDSpout
from topology.preprocess_bolt import PreprocessBolt
from topology.ml_bolt import MLBolt
from topology.alert_bolt import AlertBolt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [TOPOLOGY] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('IDSTopology')


class IDSTopology:
    """
    Defines and runs the DNIDS Storm topology.
    
    The topology connects four components in a pipeline:
    
    1. KDD Spout     → Reads CSV data line-by-line
    2. Preprocess Bolt → Scales and normalizes features
    3. ML Bolt        → Predicts Normal/Attack
    4. Alert Bolt     → Generates intrusion alerts
    
    In Storm terminology:
    - This topology runs in "local mode" (single process)
    - In a real Storm cluster, each component runs on separate workers
    - Storm handles data distribution, fault tolerance, and parallelism
    
    Attributes:
        name (str): Topology name.
        spout (KDDSpout): Data source component.
        preprocess (PreprocessBolt): Feature processing component.
        ml (MLBolt): ML prediction component.
        alert (AlertBolt): Alert generation component.
    """
    
    def __init__(self, name="dnids-intrusion-detection"):
        """
        Initialize the IDS topology.
        
        Args:
            name (str): Name of the topology.
        """
        self.name = name
        self.spout = None
        self.preprocess = None
        self.ml = None
        self.alert = None
        self._running = False
        
        logger.info(f"Topology '{name}' created")
    
    def build(self, csv_path=None, max_records=0, emit_delay=0.0,
              report_interval=5000, verbose=True):
        """
        Build the topology by creating and connecting components.
        
        This is where we define the DAG structure:
          Spout → Preprocess → ML → Alert
        
        Args:
            csv_path (str): Path to KDD dataset CSV.
            max_records (int): Maximum records to process.
            emit_delay (float): Delay between records (seconds).
            report_interval (int): Records between summary reports.
            verbose (bool): Verbose output for alerts.
        """
        logger.info("Building topology...")
        
        # Create components
        self.spout = KDDSpout(
            csv_path=csv_path,
            max_records=max_records,
            emit_delay=emit_delay
        )
        
        self.preprocess = PreprocessBolt()
        self.ml = MLBolt()
        self.alert = AlertBolt(
            report_interval=report_interval,
            verbose=verbose
        )
        
        logger.info("Topology components created:")
        logger.info("  [1] KDD Spout (source)")
        logger.info("  [2] Preprocess Bolt (feature engineering)")
        logger.info("  [3] ML Bolt (prediction)")
        logger.info("  [4] Alert Bolt (alerting)")
        logger.info("")
        logger.info("  Topology DAG:")
        logger.info("  KDD Spout --> Preprocess Bolt --> ML Bolt --> Alert Bolt")
        
        return self
    
    def run(self):
        """
        Run the topology (local mode).
        
        This simulates how Storm processes tuples:
        1. Spout emits a tuple
        2. Tuple flows through each bolt in sequence
        3. Each bolt processes and emits to the next
        4. Alert bolt is the terminal (no further output)
        
        In a real Storm cluster, this happens in parallel
        across multiple workers and machines.
        """
        print()
        print("=" * 60)
        print("  DISTRIBUTED NETWORK INTRUSION DETECTION SYSTEM")
        print("  " + "=" * 56)
        print(f"  Topology: {self.name}")
        print(f"  Mode: Local (simulating Storm cluster)")
        print(f"  Components: Spout -> Preprocess -> ML -> Alert")
        print("=" * 60)
        print()
        
        self._running = True
        start_time = time.time()
        processed = 0
        errors = 0
        
        try:
            # Initialize all components
            logger.info("Initializing components...")
            self.spout.open()
            self.preprocess.prepare()
            self.ml.prepare()
            self.alert.prepare()
            logger.info("All components initialized!\n")
            
            # Process tuples through the pipeline
            logger.info("Starting tuple processing...")
            print()
            
            while self._running and not self.spout.is_exhausted():
                # ── Stage 1: Spout emits raw data ──
                spout_output = self.spout.next_tuple()
                if spout_output is None:
                    break
                
                # ── Stage 2: Preprocess Bolt processes raw data ──
                preprocess_output = self.preprocess.process(spout_output)
                if preprocess_output is None:
                    errors += 1
                    continue
                
                # ── Stage 3: ML Bolt makes prediction ──
                ml_output = self.ml.process(preprocess_output)
                if ml_output is None:
                    errors += 1
                    continue
                
                # ── Stage 4: Alert Bolt generates alerts ──
                self.alert.process(ml_output)
                
                processed += 1
            
        except KeyboardInterrupt:
            print("\n\n[!] Topology interrupted by user (Ctrl+C)")
        except Exception as e:
            logger.error(f"Topology error: {e}")
            raise
        finally:
            self._running = False
            elapsed = time.time() - start_time
            
            # Finalize all components
            self.spout.close()
            self.alert.finalize()
            
            # Print topology summary
            print(f"\n{'=' * 60}")
            print(f"  Topology Summary")
            print(f"{'=' * 60}")
            print(f"  Records processed: {processed:,}")
            print(f"  Errors:            {errors}")
            print(f"  Processing time:   {elapsed:.1f}s")
            print(f"  Throughput:        {processed / max(0.001, elapsed):.0f} records/sec")
            print(f"{'=' * 60}")
    
    def stop(self):
        """Stop the running topology."""
        self._running = False
        logger.info("Topology stop requested")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='DNIDS - Distributed Network Intrusion Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process first 100 records (quick test)
  python topology/ids_topology.py --records 100

  # Process all records
  python topology/ids_topology.py

  # Slow mode (0.1s between records) for demo
  python topology/ids_topology.py --records 50 --delay 0.1

  # Quiet mode (less output)
  python topology/ids_topology.py --quiet
        """
    )
    
    parser.add_argument(
        '--records', '-n', type=int, default=0,
        help='Maximum records to process (0 = all). Default: 0'
    )
    parser.add_argument(
        '--delay', '-d', type=float, default=0.0,
        help='Delay between records in seconds. Default: 0.0'
    )
    parser.add_argument(
        '--dataset', type=str, default=None,
        help='Path to KDD dataset CSV file'
    )
    parser.add_argument(
        '--report-interval', '-r', type=int, default=5000,
        help='Records between summary reports. Default: 5000'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    # Build and run topology
    topology = IDSTopology()
    topology.build(
        csv_path=args.dataset,
        max_records=args.records,
        emit_delay=args.delay,
        report_interval=args.report_interval,
        verbose=not args.quiet
    )
    topology.run()


if __name__ == "__main__":
    main()
