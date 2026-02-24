"""
=============================================================
 Preprocess Bolt - Feature Scaling & Normalization
=============================================================
 This bolt receives raw CSV data from the KDD Spout and:
 1. Parses the 42-column CSV row (41 features + 1 label)
 2. Encodes categorical features using pre-trained LabelEncoders
 3. Scales numerical features using pre-trained StandardScaler
 4. Extracts the original label for evaluation
 5. Emits the preprocessed feature vector

 In Apache Storm:
 - A Bolt is a PROCESSING unit
 - It receives tuples, processes them, and emits new tuples
 - This bolt transforms raw data into ML-ready features

 Input Fields:  ['raw_data', 'record_id']
 Output Fields: ['features', 'original_label', 'record_id']
   - features:       numpy array of 41 scaled features
   - original_label: original string label (e.g., 'normal.', 'smurf.')
   - record_id:      record identifier (passed through)
=============================================================
"""

import os
import sys
import numpy as np
import joblib
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [PREPROCESS] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('PreprocessBolt')

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dataset.download_dataset import get_feature_names, CATEGORICAL_FEATURES


class PreprocessBolt:
    """
    Bolt that preprocesses raw KDD data for ML prediction.
    
    This bolt:
    1. Parses raw CSV strings into individual fields
    2. Encodes categorical features (protocol_type, service, flag)
    3. Scales all 41 numerical features using StandardScaler
    4. Emits clean, normalized feature vectors
    
    Attributes:
        scaler (StandardScaler): Pre-fitted scaler for normalization.
        label_encoders (dict): Pre-fitted LabelEncoders for categoricals.
        feature_names (list): Names of the 41 features.
        records_processed (int): Counter for processed records.
        errors (int): Counter for processing errors.
    """
    
    # Input and output field definitions
    INPUT_FIELDS = ['raw_data', 'record_id']
    OUTPUT_FIELDS = ['features', 'original_label', 'record_id']
    
    # Indices of categorical features in the 41-feature array
    CATEGORICAL_INDICES = {
        'protocol_type': 1,  # Column index 1
        'service': 2,        # Column index 2
        'flag': 3            # Column index 3
    }
    
    def __init__(self, model_dir=None):
        """
        Initialize the Preprocess Bolt.
        
        Args:
            model_dir (str): Directory containing scaler.pkl and label_encoders.pkl.
                            If None, uses default model/ directory.
        """
        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', 'model'
            )
        
        self.model_dir = model_dir
        self.scaler = None
        self.label_encoders = None
        self.feature_names = get_feature_names()
        self.records_processed = 0
        self.errors = 0
        self._initialized = False
        
        logger.info("Preprocess Bolt created")
    
    def prepare(self):
        """
        Load pre-trained scaler and label encoders.
        Called once when the bolt starts (Storm's 'prepare' method).
        """
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        encoders_path = os.path.join(self.model_dir, 'label_encoders.pkl')
        
        # Load StandardScaler
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Scaler not found: {scaler_path}\n"
                f"Run: python model/train_model.py"
            )
        self.scaler = joblib.load(scaler_path)
        logger.info(f"Loaded scaler from {scaler_path}")
        
        # Load LabelEncoders
        if not os.path.exists(encoders_path):
            raise FileNotFoundError(
                f"Label encoders not found: {encoders_path}\n"
                f"Run: python model/train_model.py"
            )
        self.label_encoders = joblib.load(encoders_path)
        logger.info(f"Loaded {len(self.label_encoders)} label encoders")
        
        for name, le in self.label_encoders.items():
            logger.info(f"  {name}: {len(le.classes_)} classes")
        
        self._initialized = True
        logger.info("Preprocess Bolt ready!")
    
    def process(self, input_tuple):
        """
        Process a single input tuple from the KDD Spout.
        
        This is the main processing method (Storm's 'execute' method).
        
        Args:
            input_tuple (dict): {'raw_data': str, 'record_id': int}
        
        Returns:
            dict: {'features': np.array, 'original_label': str, 'record_id': int}
                  or None if processing fails.
        """
        if not self._initialized:
            self.prepare()
        
        try:
            raw_data = input_tuple['raw_data']
            record_id = input_tuple['record_id']
            
            # Step 1: Parse CSV line into fields
            fields = raw_data.strip().split(',')
            
            # Validate field count (should be 42: 41 features + 1 label)
            if len(fields) < 42:
                logger.warning(
                    f"Record #{record_id}: Expected 42 fields, got {len(fields)}. Skipping."
                )
                self.errors += 1
                return None
            
            # Step 2: Extract features (first 41 fields) and label (last field)
            feature_values = fields[:41]
            original_label = fields[41].strip().rstrip('.')  # Remove trailing dot
            original_label = fields[41].strip()  # Keep original format
            
            # Step 3: Encode categorical features
            processed_features = []
            for i, value in enumerate(feature_values):
                feature_name = self.feature_names[i]
                
                if feature_name in CATEGORICAL_FEATURES:
                    # Encode categorical value using LabelEncoder
                    le = self.label_encoders[feature_name]
                    try:
                        encoded_value = le.transform([value])[0]
                    except ValueError:
                        # Unknown category - use -1 or most common
                        logger.warning(
                            f"Record #{record_id}: Unknown {feature_name}='{value}'. Using 0."
                        )
                        encoded_value = 0
                    processed_features.append(float(encoded_value))
                else:
                    # Numerical feature - convert to float
                    try:
                        processed_features.append(float(value))
                    except ValueError:
                        logger.warning(
                            f"Record #{record_id}: Invalid number for {feature_name}='{value}'. Using 0.0."
                        )
                        processed_features.append(0.0)
            
            # Step 4: Scale features using StandardScaler
            features_array = np.array(processed_features).reshape(1, -1)
            scaled_features = self.scaler.transform(features_array)
            
            self.records_processed += 1
            
            # Log progress
            if self.records_processed <= 3:
                logger.info(
                    f"Record #{record_id}: Preprocessed 41 features "
                    f"(label: {original_label})"
                )
            elif self.records_processed % 10000 == 0:
                logger.info(f"Preprocessed {self.records_processed:,} records")
            
            # Step 5: Emit processed tuple
            return {
                'features': scaled_features[0],       # 1D array of 41 scaled features
                'original_label': original_label,      # Original label string
                'record_id': record_id                 # Pass through record ID
            }
        
        except Exception as e:
            self.errors += 1
            logger.error(f"Record #{input_tuple.get('record_id', '?')}: Error - {e}")
            return None
    
    def get_stats(self):
        """Return preprocessing statistics."""
        return {
            'records_processed': self.records_processed,
            'errors': self.errors,
            'error_rate': (
                f"{self.errors / max(1, self.records_processed + self.errors) * 100:.2f}%"
            )
        }


# ─────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Preprocess Bolt - Standalone Test")
    print("=" * 60)
    
    # Sample KDD record (raw CSV)
    sample_records = [
        "0,tcp,http,SF,181,5450,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,8,8,0.00,0.00,0.00,0.00,1.00,0.00,0.00,9,9,1.00,0.00,0.11,0.00,0.00,0.00,0.00,0.00,normal.",
        "0,tcp,private,S0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,123,6,1.00,1.00,0.00,0.00,0.05,0.07,0.00,255,26,0.10,0.05,0.00,0.00,1.00,1.00,0.00,0.00,neptune.",
    ]
    
    bolt = PreprocessBolt()
    bolt.prepare()
    
    for i, raw in enumerate(sample_records, 1):
        input_tuple = {'raw_data': raw, 'record_id': i}
        result = bolt.process(input_tuple)
        
        if result:
            print(f"\n  Record #{i}:")
            print(f"    Original label: {result['original_label']}")
            print(f"    Feature shape:  {result['features'].shape}")
            print(f"    Feature range:  [{result['features'].min():.3f}, {result['features'].max():.3f}]")
            print(f"    First 5 features: {result['features'][:5]}")
    
    print(f"\n  Stats: {bolt.get_stats()}")
    print(f"\n  ✅ Preprocess Bolt test complete!")
