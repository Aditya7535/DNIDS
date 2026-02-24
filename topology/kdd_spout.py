"""
=============================================================
 KDD Spout - Data Source for DNIDS Topology
=============================================================
 This spout reads the KDD Cup 99 dataset CSV file line-by-line
 and emits each row as a tuple into the Storm topology.

 In Apache Storm:
 - A Spout is the SOURCE of data streams
 - It continuously reads from an external source
 - Each emitted tuple flows through the topology

 This spout:
 1. Opens kddcup.data_10_percent.csv
 2. Reads one line at a time
 3. Parses the CSV fields (42 columns: 41 features + 1 label)
 4. Emits each record as a tuple to the Preprocess Bolt

 Output Fields: ['raw_data', 'record_id']
   - raw_data:  The raw CSV line (string)
   - record_id: Sequential record number (int)
=============================================================
"""

import os
import sys
import csv
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [SPOUT] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('KDDSpout')


class KDDSpout:
    """
    Spout that reads KDD Cup 99 dataset line-by-line.
    
    This is a simplified Python implementation that simulates
    Storm's spout behavior. In a full Storm deployment, this
    would extend streamparse.Spout.
    
    Attributes:
        csv_path (str): Path to the KDD dataset CSV file.
        record_id (int): Current record counter.
        file_handle: Open file handle for the CSV.
        reader: CSV reader object.
        max_records (int): Maximum records to process (0 = all).
        emit_delay (float): Delay between emits (seconds).
    """
    
    # Output field names (what this spout emits)
    OUTPUT_FIELDS = ['raw_data', 'record_id']
    
    def __init__(self, csv_path=None, max_records=0, emit_delay=0.0):
        """
        Initialize the KDD Spout.
        
        Args:
            csv_path (str): Path to kddcup.data_10_percent.csv.
                           If None, uses default dataset location.
            max_records (int): Max records to read. 0 = unlimited.
            emit_delay (float): Seconds to wait between emits.
        """
        if csv_path is None:
            # Default path relative to project root
            project_root = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..'
            )
            csv_path = os.path.join(
                project_root, 'dataset', 'kddcup.data_10_percent.csv'
            )
        
        self.csv_path = csv_path
        self.max_records = max_records
        self.emit_delay = emit_delay
        self.record_id = 0
        self.file_handle = None
        self._is_open = False
        self._exhausted = False
        
        logger.info(f"KDD Spout initialized")
        logger.info(f"  Dataset: {self.csv_path}")
        logger.info(f"  Max records: {'unlimited' if max_records == 0 else max_records}")
    
    def open(self):
        """
        Open the CSV file for reading.
        Called once when the topology starts.
        
        In Storm terminology, this is the 'open' method
        that initializes the spout.
        """
        if not os.path.exists(self.csv_path):
            logger.error(f"Dataset not found: {self.csv_path}")
            logger.error("Run 'python dataset/download_dataset.py' first!")
            raise FileNotFoundError(
                f"Dataset not found: {self.csv_path}\n"
                f"Run: python dataset/download_dataset.py"
            )
        
        self.file_handle = open(self.csv_path, 'r', newline='')
        self._is_open = True
        self._exhausted = False
        self.record_id = 0
        
        # Count total lines for progress reporting
        file_size = os.path.getsize(self.csv_path)
        logger.info(f"Opened dataset ({file_size / 1024 / 1024:.1f} MB)")
    
    def next_tuple(self):
        """
        Emit the next tuple from the dataset.
        
        This method is called repeatedly by Storm to get
        the next piece of data. It reads one CSV line and
        returns it as a tuple.
        
        Returns:
            dict: {'raw_data': str, 'record_id': int} or None if exhausted.
        """
        if not self._is_open:
            self.open()
        
        if self._exhausted:
            return None
        
        # Check max records limit
        if self.max_records > 0 and self.record_id >= self.max_records:
            logger.info(f"Reached max records limit: {self.max_records}")
            self._exhausted = True
            return None
        
        # Read next line
        line = self.file_handle.readline()
        
        if not line or line.strip() == '':
            logger.info(f"Dataset exhausted after {self.record_id} records")
            self._exhausted = True
            return None
        
        # Increment record counter
        self.record_id += 1
        raw_data = line.strip()
        
        # Log progress every 10000 records
        if self.record_id % 10000 == 0:
            logger.info(f"Emitted {self.record_id:,} records...")
        elif self.record_id <= 5:
            # Log first few records for debugging
            display = raw_data[:80] + "..." if len(raw_data) > 80 else raw_data
            logger.info(f"Emitting record #{self.record_id}: {display}")
        
        # Optional delay (useful for demo/visualization)
        if self.emit_delay > 0:
            time.sleep(self.emit_delay)
        
        # Return the tuple (in Storm, this would be self.emit([raw_data, self.record_id]))
        return {
            'raw_data': raw_data,
            'record_id': self.record_id
        }
    
    def close(self):
        """
        Close the file handle.
        Called when the topology shuts down.
        """
        if self.file_handle:
            self.file_handle.close()
            self._is_open = False
            logger.info(f"Spout closed. Total records emitted: {self.record_id:,}")
    
    def is_exhausted(self):
        """Check if the spout has finished reading all data."""
        return self._exhausted
    
    def get_stats(self):
        """Return spout statistics."""
        return {
            'records_emitted': self.record_id,
            'is_exhausted': self._exhausted,
            'dataset': self.csv_path
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


# ─────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  KDD Spout - Standalone Test")
    print("=" * 60)
    
    # Test with first 10 records
    spout = KDDSpout(max_records=10, emit_delay=0.1)
    
    with spout:
        while not spout.is_exhausted():
            result = spout.next_tuple()
            if result:
                fields = result['raw_data'].split(',')
                print(f"\n  Record #{result['record_id']}:")
                print(f"    Fields: {len(fields)}")
                print(f"    Protocol: {fields[1]}")
                print(f"    Service: {fields[2]}")
                print(f"    Label: {fields[-1]}")
    
    stats = spout.get_stats()
    print(f"\n  Final Stats: {stats}")
    print(f"\n  ✅ Spout test complete!")
