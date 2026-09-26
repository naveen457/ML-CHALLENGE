import os
import sys
import pickle
import time
import numpy as np
from typing import Dict, List, Set, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from preprocess import clean_business_name, clean_address, clean_country
from blocking import InvertedIndexBlocker
from features import compute_pairwise_features
from model import compute_macro_f05, compute_f05_single

try:
    import lightgbm as lgb
    USE_LIGHTGBM = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    USE_LIGHTGBM = False

def build_training_dataset(
    train_dir: str,
    num_s1_entities: int = 8000,
    max_negatives_per_entity: int = 3
):
    print("=" * 70)
    print(f"Building Training Dataset from: {train_dir}")
    print(f"Sampling {num_s1_entities:,} Source 1 entities for training & validation...")
    print("=" * 70)

    # 1. Load Ground Truth
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    ground_truth: Dict[str, Set[str]] = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            s1_id = p[0]
            matches = set(p[1].split(',')) if len(p) > 1 and p[1].strip() else set()
            ground_truth[s1_id] = matches
            if len(ground_truth) >= num_s1_entities:
                break

    print(f"Loaded ground truth for {len(ground_truth):,} entities.")

    # 2. Load Source 1 records
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s1_records: Dict[str, Tuple[str, str, str, str]] = {}
    with open(s1_path, 'r', encoding='utf-8', errors='replace') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            if len(p) >= 4 and p[0] in ground_truth:
                e_id, b_name, b_addr, b_country = p[0], p[1], p[2], p[3]
                core_name, full_name = clean_business_name(b_name)
                clean_addr = clean_address(b_addr)
                c_country = clean_country(b_country)
                s1_records[e_id] = (core_name, full_name, clean_addr, c_country)
                if len(s1_records) >= len(ground_truth):
                    break

    # 3. Needed target IDs: all true matches for these S1
    needed_target_ids: Set[str] = set()
    for matches in ground_truth.values():
        needed_target_ids.update(matches)

    print(f"Scanning Sources 2 & 3 for target candidate records ({len(needed_target_ids):,} IDs)...")

    # 4. Read Sources 2 & 3
    s23_records: Dict[str, Tuple[str, str, str, str]] = {}
    blocker = InvertedIndexBlocker(max_candidates_per_entity=10)

    for fname in ["train_source2.tsv", "train_source3.tsv"]:
        fpath = os.path.join(train_dir, fname)
        if not os.path.exists(fpath):
            continue
        print(f"  Reading {fname} ...")
        with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
            f.readline()
            for line in f:
                p = line.rstrip('\r\n').split('\t')
                if len(p) < 4:
                    continue
                e_id, b_name, b_addr, b_country = p[0], p[1], p[2], p[3]
                # Index if needed for positive match or index sample for hard negatives
                if e_id in needed_target_ids or len(s23_records) < 150000:
                    core_name, full_name = clean_business_name(b_name)
                    clean_addr = clean_address(b_addr)
                    c_country = clean_country(b_country)
                    s23_records[e_id] = (core_name, full_name, clean_addr, c_country)
                    blocker.add_candidate(e_id, core_name, full_name, clean_addr, c_country)

    blocker.prune_high_frequency_tokens()

    # 5. Extract Feature Matrix
    feature_names = [
        'name_jaccard', 'name_overlap', 'name_ngram_jaccard',
        'name_ratio', 'token_sort_ratio', 'token_set_ratio',
        'max_name_sim', 'exact_core_match', 'exact_full_match',
        'concat_match', 'prefix_match', 'len_diff',
        'has_both_address', 'addr_jaccard', 'addr_num_jaccard',
        'addr_has_common_num', 'addr_conflicting_num', 'multi_tenant_conflict'
    ]

    X: List[List[float]] = []
    y: List[int] = []

    print("Extracting positive and hard-negative training pairs...")
    pos_count = 0
    neg_count = 0

    for s1_id, (s1_core, s1_full, s1_addr, s1_country) in s1_records.items():
        true_matches = ground_truth.get(s1_id, set())

        # Positives
        for target_id in true_matches:
            if target_id in s23_records:
                t_core, t_full, t_addr, _ = s23_records[target_id]
                feats = compute_pairwise_features(s1_core, s1_full, s1_addr, t_core, t_full, t_addr)
                X.append([feats[fn] for fn in feature_names])
                y.append(1)
                pos_count += 1

        # Hard Negatives from Blocker
        candidates = blocker.query(s1_core, s1_full, s1_addr, s1_country)
        c_added = 0
        for cand_id, cand_core, cand_full, cand_addr in candidates:
            if cand_id not in true_matches:
                feats = compute_pairwise_features(s1_core, s1_full, s1_addr, cand_core, cand_full, cand_addr)
                X.append([feats[fn] for fn in feature_names])
                y.append(0)
                neg_count += 1
                c_added += 1
                if c_added >= max_negatives_per_entity:
                    break

    print(f"Generated {len(X):,} total pairs: {pos_count:,} positives, {neg_count:,} negatives.")
    return np.array(X), np.array(y), feature_names

def train_and_optimize_threshold(X: np.ndarray, y: np.ndarray, feature_names: List[str]):
    print("\nTraining Gradient Boosted Decision Tree Matcher...")
    t0 = time.time()

    # Train / Validation split (80 / 20)
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    split_pt = int(0.8 * len(X))

    train_idx = indices[:split_pt]
    val_idx = indices[split_pt:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    if USE_LIGHTGBM:
        clf = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
    else:
        clf = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.08,
            max_depth=6,
            random_state=42
        )

    clf.fit(X_train, y_train)
    print(f"Model trained in {time.time() - t0:.2f}s.")

    # Sweep thresholds on validation set to maximize F_0.5
    val_probs = clf.predict_proba(X_val)[:, 1]
    
    best_thresh = 0.65
    best_f05 = 0.0

    print("\nSweeping decision thresholds for Macro F_0.5 optimization:")
    for thresh in np.arange(0.50, 0.90, 0.05):
        val_preds = (val_probs >= thresh).astype(int)
        tp = np.sum((val_preds == 1) & (y_val == 1))
        fp = np.sum((val_preds == 1) & (y_val == 0))
        fn = np.sum((val_preds == 0) & (y_val == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = (0.25 * precision) + recall
        f05 = (1.25 * precision * recall) / denom if denom > 0 else 0.0

        print(f"  Threshold {thresh:.2f} -> Precision: {precision:.4f}, Recall: {recall:.4f}, Pair F_0.5: {f05:.4f}")
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh

    print(f"\nOptimal Decision Threshold: {best_thresh:.2f} (Pair F_0.5: {best_f05:.4f})")
    return clf, best_thresh

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "dataset", "train"))
    models_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "models"))
    os.makedirs(models_dir, exist_ok=True)

    X, y, feature_names = build_training_dataset(train_dir, num_s1_entities=15000, max_negatives_per_entity=4)
    clf, best_threshold = train_and_optimize_threshold(X, y, feature_names)

    model_payload = {
        'classifier': clf,
        'feature_names': feature_names,
        'threshold': best_threshold
    }

    save_path = os.path.join(models_dir, "matcher_gbdt.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(model_payload, f)

    print(f"\nTrained model and optimal threshold saved successfully to: {save_path}")

if __name__ == "__main__":
    main()
