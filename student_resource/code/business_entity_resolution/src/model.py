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
    Can operate via trained GBDT (LightGBM) or precision-calibrated scoring heuristic.
    """
    def __init__(self, threshold: float = 0.58):
        self.threshold = threshold
        self.classifier = None
        self.feature_names = [
            'name_jaccard', 'name_overlap', 'name_ngram_jaccard',
            'name_ratio', 'token_sort_ratio', 'token_set_ratio',
            'exact_core_match', 'exact_full_match', 'prefix_match',
            'has_address', 'addr_jaccard', 'addr_num_jaccard', 'addr_has_common_num'
        ]

    def score_features(self, feat_dict: Dict[str, float]) -> float:
        """
        Computes match probability/confidence score.
        If a trained ML model exists, uses model.predict_proba.
        Otherwise uses an optimized precision-weighted heuristic.
        """
        if self.classifier is not None:
            vec = np.array([[feat_dict.get(fn, 0.0) for fn in self.feature_names]])
            return float(self.classifier.predict_proba(vec)[0][1])
            
        # If exact match or concatenation match (e.g. domain name vs spaced name), high confidence
        if feat_dict.get('concat_match', 0.0) > 0.5 or feat_dict.get('exact_core_match', 0.0) > 0.5:
            return 1.0

        # Calibrated name similarity
        name_score = (
            0.30 * feat_dict['name_ratio'] +
            0.25 * feat_dict['token_sort_ratio'] +
            0.20 * feat_dict['name_overlap'] +
            0.15 * feat_dict['name_ngram_jaccard'] +
            0.10 * feat_dict['prefix_match']
        )
        
        has_addr = feat_dict.get('has_address', 0.0)
        if has_addr < 0.5:
            # Address missing in one or both records: rely cleanly on name similarity
            return name_score
            
        addr_score = (
            0.50 * feat_dict['addr_jaccard'] +
            0.50 * feat_dict['addr_has_common_num']
        )
        
        # If both address and numbers match strongly, boost match score even with slight name typo
        if feat_dict['addr_has_common_num'] > 0.5 and feat_dict['addr_jaccard'] > 0.25:
            return max(name_score, 0.50 * name_score + 0.50 * addr_score)
            
        # Standard weighted combination
        combined = 0.70 * name_score + 0.30 * addr_score
        return combined

    def is_match(self, feat_dict: Dict[str, float]) -> bool:
        """Determines if a pair is predicted as a match."""
        return self.score_features(feat_dict) >= self.threshold

    def save(self, filepath: str):
        """Saves model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> 'EntityMatchingModel':
        """Loads model from disk."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
