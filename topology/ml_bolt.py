"""
=============================================================
 ML Bolt - Machine Learning Prediction
=============================================================
 This bolt receives preprocessed feature vectors and uses a
 pre-trained DecisionTreeClassifier to predict whether each
 network connection is Normal (0) or an Attack (1).

 In Apache Storm:
 - This bolt loads the trained model once (in 'prepare')
 - For each incoming tuple, it runs prediction
 - It emits the prediction result to the Alert Bolt

 Input Fields:  ['features', 'original_label', 'record_id']
 Output Fields: ['prediction', 'confidence', 'original_label', 'record_id']
   - prediction:     0 (Normal) or 1 (Attack)
   - confidence:     Prediction probability (0.0 to 1.0)
   - original_label: Original label for accuracy tracking
   - record_id:      Record identifier (passed through)
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
    format='%(asctime)s [ML] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('MLBolt')


class MLBolt:
    """
    Bolt that performs ML-based intrusion detection.
    
    This bolt:
    1. Loads a pre-trained DecisionTreeClassifier from .pkl
    2. Takes preprocessed 41-feature vectors as input
    3. Predicts: 0 = Normal, 1 = Attack
    4. Provides prediction confidence scores
    
    Attributes:
        model: Trained DecisionTreeClassifier.
        model_path (str): Path to the model .pkl file.
        predictions (int): Total predictions made.
        attacks_detected (int): Number of attacks detected.
        correct_predictions (int): Number of correct predictions.
    """
    
    # Input and output field definitions
    INPUT_FIELDS = ['features', 'original_label', 'record_id']
    OUTPUT_FIELDS = ['prediction', 'confidence', 'original_label', 'record_id']
    
    # Label mapping
    LABELS = {0: 'Normal', 1: 'Attack'}
    
    def __init__(self, model_path=None):
        """
        Initialize the ML Bolt.
        
        Args:
            model_path (str): Path to decision_tree_model.pkl.
                             If None, uses default model/ directory.
        """
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', 'model', 'decision_tree_model.pkl'
            )
        
        self.model_path = model_path
        self.model = None
        self._initialized = False
        
        # Statistics
        self.predictions = 0
        self.attacks_detected = 0
        self.normals_detected = 0
        self.correct_predictions = 0
        
        logger.info("ML Bolt created")
    
    def prepare(self):
        """
        Load the pre-trained model.
        Called once when the bolt starts (Storm's 'prepare' method).
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                f"Run: python model/train_model.py"
            )
        
        self.model = joblib.load(self.model_path)
        self._initialized = True
        
        logger.info(f"Loaded model from {self.model_path}")
        logger.info(f"  Model type: {type(self.model).__name__}")
        logger.info(f"  Tree depth: {self.model.get_depth()}")
        logger.info(f"  Num leaves: {self.model.get_n_leaves()}")
        logger.info("ML Bolt ready!")
    
    def process(self, input_tuple):
        """
        Process a single preprocessed feature vector and predict.
        
        This is the main processing method (Storm's 'execute' method).
        
        Args:
            input_tuple (dict): {
                'features': np.array (41 features),
                'original_label': str,
                'record_id': int
            }
        
        Returns:
            dict: {
                'prediction': int (0 or 1),
                'confidence': float (0.0-1.0),
                'original_label': str,
                'record_id': int
            }
        """
        if not self._initialized:
            self.prepare()
        
        try:
            features = input_tuple['features']
            original_label = input_tuple['original_label']
            record_id = input_tuple['record_id']
            
            # Reshape features for prediction (model expects 2D array)
            features_2d = features.reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(features_2d)[0]
            
            # Get prediction probabilities (confidence)
            probabilities = self.model.predict_proba(features_2d)[0]
            confidence = probabilities[prediction]  # Confidence of predicted class
            
            # Update statistics
            self.predictions += 1
            if prediction == 1:
                self.attacks_detected += 1
            else:
                self.normals_detected += 1
            
            # Check if prediction matches actual label
            actual_binary = 0 if original_label.strip() == 'normal.' else 1
            if prediction == actual_binary:
                self.correct_predictions += 1
            
            # Logging
            pred_label = self.LABELS[prediction]
            if self.predictions <= 5:
                logger.info(
                    f"Record #{record_id}: {pred_label} "
                    f"(confidence: {confidence:.2f}, actual: {original_label.strip()})"
                )
            elif self.predictions % 10000 == 0:
                accuracy = self.correct_predictions / self.predictions * 100
                logger.info(
                    f"Processed {self.predictions:,} records | "
                    f"Accuracy: {accuracy:.1f}% | "
                    f"Attacks: {self.attacks_detected:,}"
                )
            
            # Emit result
            return {
                'prediction': int(prediction),
                'confidence': float(confidence),
                'original_label': original_label,
                'record_id': record_id
            }
        
        except Exception as e:
            logger.error(f"Record #{input_tuple.get('record_id', '?')}: Prediction error - {e}")
            return None
    
    def get_stats(self):
        """Return ML bolt statistics."""
        accuracy = (
            self.correct_predictions / max(1, self.predictions) * 100
        )
        return {
            'total_predictions': self.predictions,
            'attacks_detected': self.attacks_detected,
            'normals_detected': self.normals_detected,
            'correct_predictions': self.correct_predictions,
            'accuracy': f"{accuracy:.2f}%",
            'attack_rate': f"{self.attacks_detected / max(1, self.predictions) * 100:.1f}%"
        }


# ─────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  ML Bolt - Standalone Test")
    print("=" * 60)
    
    # Create a dummy feature vector (41 features)
    dummy_features = np.random.randn(41)
    
    bolt = MLBolt()
    bolt.prepare()
    
    # Test prediction
    input_tuple = {
        'features': dummy_features,
        'original_label': 'normal.',
        'record_id': 1
    }
    
    result = bolt.process(input_tuple)
    if result:
        pred_label = MLBolt.LABELS[result['prediction']]
        print(f"\n  Prediction: {pred_label}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Original:   {result['original_label']}")
    
    print(f"\n  Stats: {bolt.get_stats()}")
    print(f"\n  ✅ ML Bolt test complete!")
