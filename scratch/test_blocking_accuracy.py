import os
import sys
from collections import defaultdict

sys.path.insert(0, r'c:\Users\navee\Documents\ML-Challenge\student_resource\code\business_entity_resolution\src')
from preprocess import clean_business_name, clean_address, clean_country
from blocking import InvertedIndexBlocker
from features import compute_pairwise_features
from model import EntityMatchingModel

train_dir = r'c:\Users\navee\Documents\ML-Challenge\student_resource\dataset\train'

print("Loading ground truth for 1,000 entities...")
gt = {}
with open(os.path.join(train_dir, "train_ground_truth.tsv"), 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        s1_id = p[0]
        matches = set(p[1].split(',')) if len(p) > 1 and p[1].strip() else set()
        if matches: # Only non-singletons
            gt[s1_id] = matches
        if len(gt) >= 1000:
            break

all_target_ids = set()
for m in gt.values():
    all_target_ids.update(m)

print(f"Targeting {len(gt)} S1 entities with {len(all_target_ids)} true matching records.")

# Read S1
s1_records = {}
with open(os.path.join(train_dir, "train_source1.tsv"), 'r', encoding='utf-8', errors='replace') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        if p[0] in gt:
            s1_records[p[0]] = (clean_business_name(p[1]), clean_address(p[2]), clean_country(p[3]))
            if len(s1_records) >= len(gt): break

# Read target S2 and S3 records and index them
blocker = InvertedIndexBlocker(max_candidates_per_entity=25)
target_records = {}

for fname in ["train_source2.tsv", "train_source3.tsv"]:
    fpath = os.path.join(train_dir, fname)
    with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            if p[0] in all_target_ids:
                (c_core, c_full) = clean_business_name(p[1])
                c_addr = clean_address(p[2])
                c_country = clean_country(p[3])
                target_records[p[0]] = (c_core, c_full, c_addr, c_country)
                blocker.add_candidate(p[0], c_core, c_full, c_addr, c_country)
            if len(target_records) >= len(all_target_ids):
                break

blocker.prune_high_frequency_tokens()

print(f"Indexed all {len(target_records)} target candidates. Now testing query recall...")

hits = 0
total = len(all_target_ids)
misses = []

for s1_id, (name_info, addr, country) in s1_records.items():
    core_name, full_name = name_info
    true_set = gt[s1_id]
    cands = blocker.query(core_name, full_name, addr, country)
    cand_ids = {c[0] for c in cands}
    for tid in true_set:
        if tid in target_records:
            if tid in cand_ids:
                hits += 1
            else:
                misses.append((s1_id, core_name, addr, country, tid, target_records[tid]))

print("=" * 60)
print(f"BLOCKER CONTROLLED RECALL: {hits}/{len(target_records)} = {hits/len(target_records)*100:.2f}%")
print("=" * 60)
print("Sample Misses (Why did blocker miss them?):")
for s1_id, s1_name, s1_addr, s1_country, tid, (t_core, t_full, t_addr, t_country) in misses[:5]:
    print(f"  S1: [{s1_name}] | [{s1_addr}] ({s1_country})")
    print(f"  True Match {tid}: [{t_core}] | [{t_addr}] ({t_country})")
    print("-" * 50)
