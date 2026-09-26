import os
import sys
from collections import defaultdict

sys.path.insert(0, r'c:\Users\navee\Documents\ML-Challenge\student_resource\code\business_entity_resolution\src')
from preprocess import clean_business_name, clean_address, clean_country
from features import compute_pairwise_features
from model import EntityMatchingModel

ADDR_STOPS = {
    'street', 'st', 'road', 'rd', 'avenue', 'ave', 'lane', 'ln', 'drive', 'dr',
    'court', 'ct', 'boulevard', 'blvd', 'unit', 'suite', 'ste', 'floor', 'fl',
    'apartment', 'apt', 'building', 'bldg', 'room', 'rm', 'block', 'shop',
    'near', 'opp', 'opposite', 'post', 'box', 'city', 'state', 'country',
    'north', 'south', 'east', 'west', 'ground', 'no', 'hno', 'plot'
}

def get_addr_keys(addr_str):
    if not addr_str: return []
    words = [w for w in addr_str.split() if len(w) >= 3 and w not in ADDR_STOPS and not w.isdigit()]
    nums = [w for w in addr_str.split() if w.isdigit()]
    keys = []
    for w in words[:3]:
        keys.append(f'loc_{w}')
    for n in nums[:2]:
        for w in words[:2]:
            keys.append(f'locnum_{w}_{n}')
    return keys

train_dir = r'c:\Users\navee\Documents\ML-Challenge\student_resource\dataset\train'

gt = {}
with open(os.path.join(train_dir, "train_ground_truth.tsv"), 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        s1_id = p[0]
        matches = set(p[1].split(',')) if len(p) > 1 and p[1].strip() else set()
        if matches:
            gt[s1_id] = matches
        if len(gt) >= 1000:
            break

all_target_ids = set()
for m in gt.values():
    all_target_ids.update(m)

# Read S1
s1_records = {}
with open(os.path.join(train_dir, "train_source1.tsv"), 'r', encoding='utf-8', errors='replace') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        if p[0] in gt:
            s1_records[p[0]] = (clean_business_name(p[1]), clean_address(p[2]), clean_country(p[3]))
            if len(s1_records) >= len(gt): break

# Dual-Key Blocker: Name + Address
index = defaultdict(lambda: defaultdict(list))
candidate_records = defaultdict(list)
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
                
                idx = len(candidate_records[c_country])
                candidate_records[c_country].append((p[0], c_core, c_full, c_addr))
                
                # Name keys
                for tok in set(c_core.split()):
                    if len(tok) >= 3:
                        index[c_country][tok].append(idx)
                # Address keys
                for a_key in get_addr_keys(c_addr):
                    index[c_country][a_key].append(idx)
            if len(target_records) >= len(all_target_ids):
                break

hits = 0
for s1_id, (name_info, addr, country) in s1_records.items():
    core_name, full_name = name_info
    true_set = gt[s1_id]
    
    # Query keys
    q_keys = [tok for tok in core_name.split() if len(tok) >= 3] + get_addr_keys(addr)
    cand_hits = defaultdict(int)
    for k in q_keys:
        for c_idx in index[country].get(k, []):
            cand_hits[c_idx] += 1
            
    top_cand_ids = {candidate_records[country][idx][0] for idx, _ in sorted(cand_hits.items(), key=lambda x: x[1], reverse=True)[:30]}
    
    for tid in true_set:
        if tid in top_cand_ids:
            hits += 1

print("=" * 60)
print(f"DUAL-KEY BLOCKER RECALL: {hits}/{len(target_records)} = {hits/len(target_records)*100:.2f}%")
print("=" * 60)
