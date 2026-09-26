import os
import pickle
import json
import math
from typing import Dict, List, Set, Tuple, Optional
import numpy as np

def eval_tree_node(node: dict, row: List[float]) -> float:
    """Recursive evaluation of a single LightGBM decision tree node in pure Python."""
    if 'leaf_value' in node:
        return node['leaf_value']
    feat_idx = node['split_feature']
    val = row[feat_idx]
    if val <= node['threshold']:
        return eval_tree_node(node['left_child'], row)
    else:
        return eval_tree_node(node['right_child'], row)

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
    Operates via trained GBDT model if available, standalone pure JSON trees, or precision heuristic.
    """
    def __init__(self, model_path: Optional[str] = None, threshold: float = 0.75):
        self.threshold = threshold
        self.classifier = None
        self.json_trees = None
        self.feature_names = [
            'name_jaccard', 'name_overlap', 'name_ngram_jaccard',
            'name_ratio', 'token_sort_ratio', 'token_set_ratio',
            'max_name_sim', 'exact_core_match', 'exact_full_match',
            'concat_match', 'prefix_match', 'len_diff',
            'has_both_address', 'addr_jaccard', 'addr_num_jaccard',
            'addr_has_common_num', 'addr_conflicting_num', 'multi_tenant_conflict'
        ]

        # Auto-detect trained model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            default_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "matcher_gbdt.pkl"))
            if os.path.exists(default_path):
                self.load_model(default_path)
            else:
                json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "matcher_trees.json"))
                if os.path.exists(json_path):
                    self.load_model(json_path)

    def load_model(self, filepath: str):
        if filepath.endswith('.json'):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    tdata = json.load(f)
                self.json_trees = tdata['tree_info']
                self.feature_names = tdata.get('feature_names', self.feature_names)
                self.threshold = tdata.get('threshold', self.threshold)
                print(f"Loaded standalone JSON trees from: {filepath} (Threshold: {self.threshold:.2f})")
                return
            except Exception as je:
                print(f"Warning: Failed to load JSON trees from {filepath}: {je}")

        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.classifier = data['classifier']
            self.feature_names = data['feature_names']
            self.threshold = data.get('threshold', self.threshold)
            print(f"Loaded trained GBDT model from: {filepath} (Threshold: {self.threshold:.2f})")
        except Exception as e:
            # Fallback to pure JSON tree dump if lightgbm is not installed in current Python env
            trees_path = os.path.join(os.path.dirname(filepath), "matcher_trees.json")
            if os.path.exists(trees_path):
                try:
                    with open(trees_path, 'r', encoding='utf-8') as f:
                        tdata = json.load(f)
                    self.json_trees = tdata['tree_info']
                    self.feature_names = tdata.get('feature_names', self.feature_names)
                    self.threshold = tdata.get('threshold', self.threshold)
                    print(f"Loaded standalone JSON trees from: {trees_path} (Threshold: {self.threshold:.2f})")
                    return
                except Exception as je:
                    pass
            print(f"Warning: Failed to load model from {filepath}: {e}. Using calibrated heuristic.")
            self.classifier = None

    def score_batch(self, feat_dicts: List[Dict[str, float]]) -> np.ndarray:
        """
        Fast vectorized batch scoring. Passes full matrix to LightGBM in C++ for maximum throughput,
        or evaluates standalone decision trees in pure Python with zero dependencies.
        Enforces multi-tenant and street number consistency constraints.
        """
        if not feat_dicts:
            return np.array([])
            
        if self.classifier is not None:
            # Build 2D numpy array in one go
            matrix = np.array([[fd.get(fn, 0.0) for fn in self.feature_names] for fd in feat_dicts], dtype=np.float32)
            probs = self.classifier.predict_proba(matrix)[:, 1]
        elif self.json_trees is not None:
            # Standalone pure-Python tree evaluation (zero external dependencies)
            probs = []
            for fd in feat_dicts:
                row = [fd.get(fn, 0.0) for fn in self.feature_names]
                raw = sum(eval_tree_node(t['tree_structure'], row) for t in self.json_trees)
                p = 1.0 / (1.0 + math.exp(-raw))
                probs.append(p)
            probs = np.array(probs, dtype=np.float32)
        else:
            # Fallback heuristic
            probs = np.array([self.score_features(fd) for fd in feat_dicts], dtype=np.float32)
            
        # Vectorized multi-tenant & house-number consistency guards
        for i, fd in enumerate(feat_dicts):
            # 1. Multi-tenant building conflict: same address, unrelated businesses
            if fd.get('multi_tenant_conflict', 0.0) > 0.5:
                probs[i] = 0.0
            # 2. Conflicting street unit numbers on same street
            elif (fd.get('addr_conflicting_num', 0.0) > 0.5 and 
                  fd.get('exact_core_match', 0.0) < 0.5 and 
                  fd.get('concat_match', 0.0) < 0.5 and 
                  fd.get('max_name_sim', 0.0) < 0.85):
                probs[i] = 0.0
        return probs

    def score_features(self, feat_dict: Dict[str, float]) -> float:
        """
        Computes match probability/confidence score.
        If a trained ML model exists, uses model.predict_proba.
        Otherwise uses an optimized precision-weighted heuristic.
        """
        # Hard multi-tenant building constraint
        if feat_dict.get('multi_tenant_conflict', 0.0) > 0.5:
            return 0.0

        # Hard conflicting street numbers constraint
        if (feat_dict.get('addr_conflicting_num', 0.0) > 0.5 and 
            feat_dict.get('exact_core_match', 0.0) < 0.5 and 
            feat_dict.get('concat_match', 0.0) < 0.5 and 
            feat_dict.get('max_name_sim', 0.0) < 0.85):
            return 0.0

        if self.classifier is not None:
            vec = np.array([[feat_dict.get(fn, 0.0) for fn in self.feature_names]])
            prob = float(self.classifier.predict_proba(vec)[0][1])
            return prob
            
        # Calibrated Heuristic Mode
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
            return name_score if name_score >= 0.70 else name_score * 0.4
            
        addr_score = (
            0.50 * feat_dict['addr_jaccard'] +
            0.50 * feat_dict['addr_has_common_num']
        )
        
        # If both address and numbers match strongly, boost score
        if feat_dict['addr_has_common_num'] > 0.5 and feat_dict['addr_jaccard'] > 0.30:
            return max(name_score, 0.40 * name_score + 0.60 * addr_score)
            
        combined = 0.70 * name_score + 0.30 * addr_score
        return combined

    def is_match(self, feat_dict: Dict[str, float]) -> bool:
        """Determines if a pair is predicted as a match."""
        return self.score_features(feat_dict) >= self.threshold
