"""
=============================================================
 Alert Bolt - Intrusion Alert System
=============================================================
 This bolt receives ML predictions and generates alerts
 for detected intrusions.

 For each incoming prediction:
 - If prediction = 1 (Attack) → Logs "INTRUSION DETECTED!" 🚨
 - If prediction = 0 (Normal) → Logs "Normal traffic" ✅

 It also:
 - Maintains running statistics
 - Calculates real-time accuracy
 - Generates periodic summary reports
 - Logs all alerts to a file

 Input Fields:  ['prediction', 'confidence', 'original_label', 'record_id']
 Output Fields: None (terminal bolt - no further output)

 This is the final bolt in the DNIDS topology pipeline.
=============================================================
"""

import os
import sys
import time
import logging
import io
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32' and not isinstance(sys.stdout, io.TextIOWrapper):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [ALERT] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('AlertBolt')


class AlertBolt:
    """
    Terminal bolt that generates intrusion alerts.
    
    This is the FINAL component in the DNIDS topology.
    It receives predictions from the ML Bolt and:
    1. Logs alerts for detected attacks
    2. Tracks running statistics
    3. Generates summary reports
    4. Optionally writes alerts to a log file
    
    Attributes:
        total_processed (int): Total records processed.
        attacks (int): Number of attacks detected.
        normals (int): Number of normal connections.
        correct (int): Number of correct predictions.
        alert_log_path (str): Path to alert log file.
        report_interval (int): Records between summary reports.
    """
    
    # Input field definitions (terminal bolt - no output)
    INPUT_FIELDS = ['prediction', 'confidence', 'original_label', 'record_id']
    
    # Attack type mapping (common KDD attack labels)
    ATTACK_TYPES = {
        'neptune.': 'DOS - Neptune (SYN Flood)',
        'smurf.': 'DOS - Smurf (ICMP Flood)',
        'back.': 'DOS - Back (Apache Attack)',
        'teardrop.': 'DOS - Teardrop (IP Fragment)',
        'pod.': 'DOS - Pod (Ping of Death)',
        'land.': 'DOS - Land (IP Spoofing)',
        'satan.': 'PROBE - Satan (Network Scan)',
        'ipsweep.': 'PROBE - IP Sweep',
        'portsweep.': 'PROBE - Port Sweep',
        'nmap.': 'PROBE - Nmap Scan',
        'warezclient.': 'R2L - Warezmaster Client',
        'guess_passwd.': 'R2L - Password Guessing',
        'warezmaster.': 'R2L - Warezmaster',
        'imap.': 'R2L - IMAP Attack',
        'ftp_write.': 'R2L - FTP Write',
        'multihop.': 'R2L - Multihop Attack',
        'phf.': 'R2L - PHF (CGI Attack)',
        'spy.': 'R2L - Spy',
        'buffer_overflow.': 'U2R - Buffer Overflow',
        'rootkit.': 'U2R - Rootkit',
        'loadmodule.': 'U2R - Load Module',
        'perl.': 'U2R - Perl Attack',
    }
    
    def __init__(self, alert_log_path=None, report_interval=1000, verbose=True):
        """
        Initialize the Alert Bolt.
        
        Args:
            alert_log_path (str): Path to write alert logs.
                                 If None, uses default logs/ directory.
            report_interval (int): How often to print summary reports.
            verbose (bool): If True, print individual alerts.
        """
        if alert_log_path is None:
            log_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', 'logs'
            )
            os.makedirs(log_dir, exist_ok=True)
            alert_log_path = os.path.join(log_dir, 'intrusion_alerts.log')
        
        self.alert_log_path = alert_log_path
        self.report_interval = report_interval
        self.verbose = verbose
        
        # Statistics
        self.total_processed = 0
        self.attacks = 0
        self.normals = 0
        self.correct = 0
        self.true_positives = 0    # Correctly detected attacks
        self.false_positives = 0   # Normal flagged as attack
        self.true_negatives = 0    # Correctly detected normal
        self.false_negatives = 0   # Attack missed (flagged as normal)
        
        # Timing
        self.start_time = None
        
        # Alert file
        self._alert_file = None
        self._initialized = False
        
        logger.info("Alert Bolt created")
    
    def prepare(self):
        """
        Initialize the alert bolt.
        Called once when the bolt starts.
        """
        self.start_time = time.time()
        
        # Open alert log file
        self._alert_file = open(self.alert_log_path, 'w')
        self._alert_file.write(
            f"{'=' * 70}\n"
            f" DNIDS Intrusion Alert Log\n"
            f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'=' * 70}\n\n"
        )
        
        self._initialized = True
        logger.info(f"Alert log: {self.alert_log_path}")
        logger.info("Alert Bolt ready!")
    
    def process(self, input_tuple):
        """
        Process a prediction result and generate alerts.
        
        This is the main processing method (Storm's 'execute' method).
        
        Args:
            input_tuple (dict): {
                'prediction': int (0 or 1),
                'confidence': float,
                'original_label': str,
                'record_id': int
            }
        """
        if not self._initialized:
            self.prepare()
        
        prediction = input_tuple['prediction']
        confidence = input_tuple['confidence']
        original_label = input_tuple['original_label'].strip()
        record_id = input_tuple['record_id']
        
        # Determine actual class
        actual_binary = 0 if original_label == 'normal.' else 1
        is_correct = prediction == actual_binary
        
        # Update statistics
        self.total_processed += 1
        if is_correct:
            self.correct += 1
        
        if prediction == 1:
            # ─── ATTACK DETECTED ───
            self.attacks += 1
            
            if actual_binary == 1:
                self.true_positives += 1
            else:
                self.false_positives += 1
            
            # Get attack type description
            attack_info = self.ATTACK_TYPES.get(
                original_label, f'Unknown ({original_label})'
            )
            
            # Generate alert message
            alert_msg = (
                f"[!!!] INTRUSION DETECTED! | Record #{record_id} | "
                f"Confidence: {confidence:.2f} | "
                f"Type: {attack_info}"
            )
            
            # Log alert
            if self.verbose and self.attacks <= 20:
                logger.warning(alert_msg)
            
            # Write to alert file
            if self._alert_file:
                timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                self._alert_file.write(
                    f"[{timestamp}] ALERT | "
                    f"Record: {record_id} | "
                    f"Confidence: {confidence:.4f} | "
                    f"Attack: {attack_info} | "
                    f"Correct: {is_correct}\n"
                )
        else:
            # ─── NORMAL TRAFFIC ───
            self.normals += 1
            
            if actual_binary == 0:
                self.true_negatives += 1
            else:
                self.false_negatives += 1
            
            if self.verbose and self.normals <= 5:
                logger.info(
                    f"[OK] Normal traffic | Record #{record_id} | "
                    f"Confidence: {confidence:.2f}"
                )
        
        # Print periodic summary report
        if self.total_processed % self.report_interval == 0:
            self._print_summary()
    
    def _print_summary(self):
        """Print a running summary report."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        accuracy = self.correct / max(1, self.total_processed) * 100
        rate = self.total_processed / max(0.001, elapsed)
        
        # Calculate precision and recall
        precision = (
            self.true_positives / max(1, self.true_positives + self.false_positives) * 100
        )
        recall = (
            self.true_positives / max(1, self.true_positives + self.false_negatives) * 100
        )
        
        report = f"""
{'-' * 60}
 DNIDS Running Statistics (Record #{self.total_processed:,})
{'-' * 60}
  Total Processed: {self.total_processed:>10,}
  Normal Traffic:  {self.normals:>10,} ({self.normals/max(1,self.total_processed)*100:>5.1f}%)
  Attacks Found:   {self.attacks:>10,} ({self.attacks/max(1,self.total_processed)*100:>5.1f}%)
{'-' * 60}
  Model Accuracy:  {accuracy:>10.2f}%
  Precision:       {precision:>10.2f}%
  Recall:          {recall:>10.2f}%
{'-' * 60}
  True Positives:  {self.true_positives:>10,}  (Attacks correctly detected)
  False Positives: {self.false_positives:>10,}  (Normal flagged as attack)
  True Negatives:  {self.true_negatives:>10,}  (Normal correctly identified)
  False Negatives: {self.false_negatives:>10,}  (Attacks missed)
{'-' * 60}
  Processing Rate: {rate:>10.0f} records/sec
  Elapsed Time:    {elapsed:>10.1f} seconds
{'-' * 60}"""
        print(report)
        
        # Also write to alert file
        if self._alert_file:
            self._alert_file.write(f"\n{report}\n\n")
            self._alert_file.flush()
    
    def finalize(self):
        """
        Print final report and close resources.
        Called when the topology shuts down.
        """
        print(f"\n{'=' * 60}")
        print(f" DNIDS - FINAL REPORT")
        print(f"{'=' * 60}")
        
        self._print_summary()
        
        # Close alert file
        if self._alert_file:
            self._alert_file.write(
                f"\n{'=' * 70}\n"
                f" Processing Complete\n"
                f" Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"{'=' * 70}\n"
            )
            self._alert_file.close()
            logger.info(f"Alert log saved: {self.alert_log_path}")
        
        print(f"\n [OK] DNIDS processing complete!")
        print(f" Alert log: {self.alert_log_path}")
    
    def get_stats(self):
        """Return alert bolt statistics."""
        accuracy = self.correct / max(1, self.total_processed) * 100
        return {
            'total_processed': self.total_processed,
            'attacks_detected': self.attacks,
            'normals': self.normals,
            'accuracy': f"{accuracy:.2f}%",
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'true_negatives': self.true_negatives,
            'false_negatives': self.false_negatives,
        }


# ─────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Alert Bolt - Standalone Test")
    print("=" * 60)
    
    bolt = AlertBolt(report_interval=3, verbose=True)
    bolt.prepare()
    
    # Test with sample predictions
    test_data = [
        {'prediction': 0, 'confidence': 0.95, 'original_label': 'normal.', 'record_id': 1},
        {'prediction': 1, 'confidence': 0.88, 'original_label': 'neptune.', 'record_id': 2},
        {'prediction': 1, 'confidence': 0.92, 'original_label': 'smurf.', 'record_id': 3},
        {'prediction': 0, 'confidence': 0.97, 'original_label': 'normal.', 'record_id': 4},
        {'prediction': 1, 'confidence': 0.75, 'original_label': 'normal.', 'record_id': 5},  # False positive
        {'prediction': 0, 'confidence': 0.60, 'original_label': 'satan.', 'record_id': 6},   # False negative
    ]
    
    for data in test_data:
        bolt.process(data)
    
    bolt.finalize()
    print(f"\n  [OK] Alert Bolt test complete!")
