import os
import sys
import time
from collections import defaultdict
from typing import Dict, Set, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from preprocess import clean_business_name, clean_address, clean_country
from blocking import InvertedIndexBlocker
from features import compute_pairwise_features
from model import EntityMatchingModel, compute_f05_single, compute_macro_f05

def run_local_evaluation(sample_size: int = 3000, max_candidates: int = 25):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "dataset", "train"))
    models_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "models"))

    print("=" * 70)
    print(f"FAST LOCAL VALIDATION HARNESS (Target: Macro F_0.5 >= 0.90)")
    print(f"Sample Size: {sample_size:,} Source 1 entities")
    print(f"Data Source: {train_dir}")
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
            if len(ground_truth) >= sample_size:
                break

    all_target_ids: Set[str] = set()
    for m in ground_truth.values():
        all_target_ids.update(m)

    # 2. Load S1 Records
    s1_records: Dict[str, Tuple[str, str, str, str]] = {}
    with open(os.path.join(train_dir, "train_source1.tsv"), 'r', encoding='utf-8', errors='replace') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            if p[0] in ground_truth:
                core_name, full_name = clean_business_name(p[1])
                addr = clean_address(p[2])
                country = clean_country(p[3])
                s1_records[p[0]] = (core_name, full_name, addr, country)
                if len(s1_records) >= len(ground_truth):
                    break

    # 3. Index Sources 2 & 3
    blocker = InvertedIndexBlocker(max_candidates_per_entity=max_candidates)
    indexed_records = 0
    t0 = time.time()

    for fname in ["train_source2.tsv", "train_source3.tsv"]:
        fpath = os.path.join(train_dir, fname)
        with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
            f.readline()
            for line in f:
                p = line.rstrip('\r\n').split('\t')
                if len(p) < 4:
                    continue
                # Index all true matches + background noise records
                if p[0] in all_target_ids or indexed_records < 80000:
                    core_name, full_name = clean_business_name(p[1])
                    addr = clean_address(p[2])
                    country = clean_country(p[3])
                    blocker.add_candidate(p[0], core_name, full_name, addr, country)
                    indexed_records += 1

    blocker.prune_high_frequency_tokens()
    print(f"Indexed {indexed_records:,} candidate records in {time.time() - t0:.1f}s.")

    # 4. Load Model
    model = EntityMatchingModel()
    print(f"Loaded Matcher (Threshold: {model.threshold:.2f})")

    # 5. Run Evaluation Pipeline
    predictions: Dict[str, Set[str]] = {}
    total_true_matches = sum(len(m) for m in ground_truth.values())
    retrieved_true_matches = 0
    total_predicted_matches = 0
    true_positives = 0

    eval_t0 = time.time()
    
    # Process in batches for maximum vector speed
    BATCH_SIZE = 1000
    items = list(s1_records.items())

    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i:i + BATCH_SIZE]
        all_feats = []
        slices = []

        for s1_id, (core_name, full_name, addr, country) in batch:
            cands = blocker.query(core_name, full_name, addr, country)
            cand_ids = {c[0] for c in cands}
            
            # Measure blocking recall
            true_set = ground_truth.get(s1_id, set())
            retrieved_true_matches += len(true_set.intersection(cand_ids))

            start_idx = len(all_feats)
            for c in cands:
                feats = compute_pairwise_features(core_name, full_name, addr, c[1], c[2], c[3])
                all_feats.append(feats)
            end_idx = len(all_feats)
            slices.append((s1_id, cands, start_idx, end_idx))

        scores = model.score_batch(all_feats)

        for s1_id, cands, start_idx, end_idx in slices:
            cand_scores = scores[start_idx:end_idx]
            scored = []
            for (c_id, _, _, _), sc in zip(cands, cand_scores):
                if sc >= model.threshold:
                    scored.append((sc, c_id))
            scored.sort(key=lambda x: x[0], reverse=True)
            pred_set = {cid for _, cid in scored}
            predictions[s1_id] = pred_set

            total_predicted_matches += len(pred_set)
            true_positives += len(ground_truth[s1_id].intersection(pred_set))

    eval_time = time.time() - eval_t0

    # Calculate Official Metrics
    macro_f05 = compute_macro_f05(ground_truth, predictions)
    blocking_recall = (retrieved_true_matches / total_true_matches * 100) if total_true_matches else 0.0
    pair_precision = (true_positives / total_predicted_matches * 100) if total_predicted_matches else 0.0
    pair_recall = (true_positives / total_true_matches * 100) if total_true_matches else 0.0

    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS ({eval_time:.1f}s):")
    print(f"  Candidate Blocker Recall:  {blocking_recall:.2f}%  (Upper bound ceiling)")
    print(f"  Model Pair Precision:      {pair_precision:.2f}%")
    print(f"  Model Pair Recall:         {pair_recall:.2f}%")
    print(f"  Total True Matches:        {total_true_matches:,}")
    print(f"  Total Predicted Matches:   {total_predicted_matches:,}")
    print("-" * 70)
    print(f"  >>> OFFICIAL MACRO F_0.5 SCORE: {macro_f05:.4f} <<<")
    print("=" * 70)

    if macro_f05 >= 0.90:
        print("SUCCESS! Target score >= 0.90 achieved! Ready for full test run.")
    else:
        print(f"Gap to 0.90 target: {0.90 - macro_f05:.4f}. Optimizing parameters...")

    return macro_f05

if __name__ == "__main__":
    run_local_evaluation()
