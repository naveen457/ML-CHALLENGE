import os
import pickle
from typing import Dict, List, Set, Tuple, Optional
import numpy as np

def compute_f05_single(true_set: Set[str], pred_set: Set[str]) -> float:
    """
    Computes F_0.5 score for a single Source 1 entity.
    Includes singleton handling as defined in the competition guidelines.
    """
    if len(true_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0
    if len(pred_set) == 0:
        return 0.0
        
    common = len(true_set.intersection(pred_set))
    if common == 0:
        return 0.0
        
    precision = float(common) / len(pred_set)
    recall = float(common) / len(true_set)
    denom = (0.25 * precision) + recall
    if denom == 0:
        return 0.0
    return (1.25 * precision * recall) / denom

def compute_macro_f05(ground_truth: Dict[str, Set[str]], predictions: Dict[str, Set[str]]) -> float:
    """Computes macro-average F_0.5 score across all Source 1 entities."""
    total_score = 0.0
    count = len(ground_truth)
    if count == 0:
        return 0.0
        
    for s1_id, true_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, set())
        total_score += compute_f05_single(true_matches, pred_matches)
        
    return total_score / count

class EntityMatchingModel:
    """
    Match scoring model for entity resolution pairs.
    Operates via trained GBDT model if available, or precision-calibrated scoring heuristic.
    """
    def __init__(self, model_path: Optional[str] = None, threshold: float = 0.65):
        self.threshold = threshold
        self.classifier = None
        self.feature_names = [
            'name_jaccard', 'name_overlap', 'name_ngram_jaccard',
            'name_ratio', 'token_sort_ratio', 'token_set_ratio',
            'exact_core_match', 'exact_full_match', 'concat_match',
            'prefix_match', 'len_diff', 'has_both_address',
            'addr_jaccard', 'addr_num_jaccard', 'addr_has_common_num',
            'addr_conflicting_num'
        ]

        # Auto-detect trained model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            default_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "matcher_gbdt.pkl"))
            if os.path.exists(default_path):
                self.load_model(default_path)

    def load_model(self, filepath: str):
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.classifier = data['classifier']
            self.feature_names = data['feature_names']
            self.threshold = data.get('threshold', self.threshold)
            print(f"Loaded trained GBDT model from: {filepath} (Threshold: {self.threshold:.2f})")
        except Exception as e:
            print(f"Warning: Failed to load model from {filepath}: {e}. Using calibrated heuristic.")
            self.classifier = None

    def score_batch(self, feat_dicts: List[Dict[str, float]]) -> np.ndarray:
        """
        Fast vectorized batch scoring. Passes full matrix to LightGBM in C++ for maximum throughput.
        """
        if not feat_dicts:
            return np.array([])
            
        if self.classifier is not None:
            # Build 2D numpy array in one go
            matrix = np.array([[fd.get(fn, 0.0) for fn in self.feature_names] for fd in feat_dicts], dtype=np.float32)
            probs = self.classifier.predict_proba(matrix)[:, 1]
            
            # Vectorized penalty for conflicting street numbers
            for i, fd in enumerate(feat_dicts):
                if fd.get('addr_conflicting_num', 0.0) > 0.5 and fd.get('exact_core_match', 0.0) < 0.5:
                    probs[i] *= 0.5
            return probs
            
        # Fallback heuristic
        return np.array([self.score_features(fd) for fd in feat_dicts])

    def score_features(self, feat_dict: Dict[str, float]) -> float:
        """
        Computes match probability/confidence score.
        If a trained ML model exists, uses model.predict_proba.
        Otherwise uses an optimized precision-weighted heuristic.
        """
        if self.classifier is not None:
            vec = np.array([[feat_dict.get(fn, 0.0) for fn in self.feature_names]])
            prob = float(self.classifier.predict_proba(vec)[0][1])
            # If addresses have conflicting building numbers, suppress score to protect precision
            if feat_dict.get('addr_conflicting_num', 0.0) > 0.5 and feat_dict.get('exact_core_match', 0.0) < 0.5:
                prob *= 0.5
            return prob
            
        # Calibrated Heuristic Mode
        # Immediate rejection on conflicting address numbers unless name is identical
        if feat_dict.get('addr_conflicting_num', 0.0) > 0.5 and feat_dict.get('exact_core_match', 0.0) < 0.5:
            return 0.10

        # Exact match or concatenation match (e.g. domain name vs spaced name)
        if feat_dict.get('concat_match', 0.0) > 0.5 or feat_dict.get('exact_core_match', 0.0) > 0.5:
            return 1.0

        # High-discrimination name similarity
        name_score = (
            0.30 * feat_dict['name_ratio'] +
            0.25 * feat_dict['token_sort_ratio'] +
            0.20 * feat_dict['name_overlap'] +
            0.15 * feat_dict['name_ngram_jaccard'] +
            0.10 * feat_dict['prefix_match']
        )
        
        has_addr = feat_dict.get('has_both_address', 0.0)
        if has_addr < 0.5:
            # Address missing in one or both records: require strong name match
            return name_score if name_score >= 0.65 else name_score * 0.5
            
        addr_score = (
            0.50 * feat_dict['addr_jaccard'] +
            0.50 * feat_dict['addr_has_common_num']
        )
        
        # If both address and numbers match strongly, boost score
        if feat_dict['addr_has_common_num'] > 0.5 and feat_dict['addr_jaccard'] > 0.25:
            return max(name_score, 0.50 * name_score + 0.50 * addr_score)
            
        combined = 0.70 * name_score + 0.30 * addr_score
        return combined

    def is_match(self, feat_dict: Dict[str, float]) -> bool:
        """Determines if a pair is predicted as a match."""
        return self.score_features(feat_dict) >= self.threshold
