import os
import sys
import csv
import argparse
import time
from typing import Dict, List, Set

# Add current directory to path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from preprocess import clean_business_name, clean_address, clean_country
from blocking import InvertedIndexBlocker
from features import compute_pairwise_features
from model import EntityMatchingModel

def parse_tsv_line(line: str) -> List[str]:
    return [col.strip() for col in line.rstrip('\r\n').split('\t')]

def resolve_data_dir(data_dir: str) -> str:
    """Auto-detects data directory regardless of whether the user runs from root, student_resource, or src."""
    if os.path.isdir(data_dir):
        return os.path.abspath(data_dir)
    
    candidates = [
        os.path.join("student_resource", data_dir),
        os.path.join(current_dir, "..", "..", data_dir),
        os.path.join(current_dir, "..", "..", "..", "student_resource", data_dir),
        os.path.join(current_dir, "..", data_dir)
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    return data_dir

def resolve_output_dir(output_dir: str, resolved_data_dir: str) -> str:
    """Resolves output directory relative to student_resource or current working directory."""
    if os.path.isabs(output_dir):
        return output_dir
    norm_out = output_dir.replace('\\', '/')
    if norm_out.startswith("student_resource/"):
        return os.path.abspath(output_dir)
    # If data_dir was found inside student_resource, default output to student_resource/output
    if "student_resource" in resolved_data_dir:
        sr_root = resolved_data_dir.split("student_resource")[0] + "student_resource"
        return os.path.abspath(os.path.join(sr_root, output_dir))
    return os.path.abspath(output_dir)

def run_pipeline(
    data_dir: str,
    output_dir: str,
    prefix: str = "test",
    limit: int = None,
    threshold: float = None,
    max_candidates: int = 25
):
    resolved_data = resolve_data_dir(data_dir)
    resolved_output = resolve_output_dir(output_dir, resolved_data)

    print("=" * 70)
    print(f"Starting Business Entity Resolution Pipeline")
    print(f"Data directory:   {resolved_data}")
    print(f"Output directory: {resolved_output}")
    print(f"Dataset prefix:   {prefix}")
    print(f"Entity limit:     {limit if limit else 'ALL (Full Run)'}")
    print("=" * 70)

    data_dir = resolved_data
    output_dir = resolved_output
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Initialize candidate blocker and matcher
    blocker = InvertedIndexBlocker(max_candidates_per_entity=max_candidates)
    model = EntityMatchingModel(threshold=threshold if threshold is not None else 0.75)
    print(f"Active Match Decision Threshold: {model.threshold:.2f}")

    # 2. Index Source 2 and Source 3
    source2_path = os.path.join(data_dir, f"{prefix}_source2.tsv")
    source3_path = os.path.join(data_dir, f"{prefix}_source3.tsv")

    try:
        import polars as pl
        has_polars = True
    except ImportError:
        has_polars = False

    for s_path in [source2_path, source3_path]:
        if not os.path.exists(s_path):
            print(f"Warning: File {s_path} not found. Skipping...")
            continue
            
        print(f"\nIndexing records from: {os.path.basename(s_path)} ...")
        t0 = time.time()
        count = 0

        if has_polars:
            # High-throughput columnar ingestion via Polars (C++/Rust multi-threaded)
            n_rows = (limit * 3) if limit else None
            df = pl.read_csv(s_path, separator='\t', truncate_ragged_lines=True, n_rows=n_rows)
            e_ids = df['entity_id'].to_list()
            b_names = df['business_name'].to_list()
            b_addrs = df['business_address'].to_list()
            b_countries = df['country'].to_list()
            del df # Free raw dataframe memory immediately

            for i in range(len(e_ids)):
                c_country = clean_country(b_countries[i])
                core_name, full_name = clean_business_name(b_names[i])
                clean_addr = clean_address(b_addrs[i])
                blocker.add_candidate(e_ids[i], core_name, full_name, clean_addr, c_country)
                count += 1
                if count % 500000 == 0:
                    print(f"  Indexed {count:,} records ({time.time() - t0:.1f}s)")
        else:
            # Streaming line-by-line fallback
            with open(s_path, 'r', encoding='utf-8', errors='replace') as f:
                header_line = f.readline()
                for line in f:
                    parts = line.rstrip('\r\n').split('\t')
                    if len(parts) < 4:
                        continue
                    e_id, b_name, b_addr, b_country = parts[0], parts[1], parts[2], parts[3]
                    c_country = clean_country(b_country)
                    core_name, full_name = clean_business_name(b_name)
                    clean_addr = clean_address(b_addr)
                    blocker.add_candidate(e_id, core_name, full_name, clean_addr, c_country)
                    count += 1
                    if count % 500000 == 0:
                        print(f"  Indexed {count:,} records ({time.time() - t0:.1f}s)")
                    if limit and count >= limit * 3:
                        break
        print(f"Done indexing {count:,} records in {time.time() - t0:.1f}s.")

    print("\nPruning high-frequency tokens from index...")
    blocker.prune_high_frequency_tokens()

    # 3. Process Source 1 out-of-core in chunks to maintain bounded RAM
    source1_path = os.path.join(data_dir, f"{prefix}_source1.tsv")
    if not os.path.exists(source1_path):
        raise FileNotFoundError(f"Source 1 file not found: {source1_path}")

    matching_tsv_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_tsv_path = os.path.join(output_dir, "candidate_pairs.tsv")

    print(f"\nProcessing Source 1 entities from: {os.path.basename(source1_path)} ...")
    t0 = time.time()
    s1_count = 0
    total_matches = 0
    total_candidates = 0
    CHUNK_SIZE = 5000

    with open(matching_tsv_path, 'w', encoding='utf-8', newline='') as match_f, \
         open(candidate_tsv_path, 'w', encoding='utf-8', newline='') as cand_f:

        # Write exact required headers
        match_f.write("source1_entity_id\tmatched_entity_ids\n")
        cand_f.write("source1_entity_id\tcandidate_entity_ids\n")

        def process_chunk(e_ids, b_names, b_addrs, b_countries):
            nonlocal total_candidates, total_matches, s1_count
            all_feats = []
            entity_cand_slices = []

            for i in range(len(e_ids)):
                e_id = e_ids[i]
                c_country = clean_country(b_countries[i])
                core_name, full_name = clean_business_name(b_names[i])
                clean_addr = clean_address(b_addrs[i])

                candidates = blocker.query(core_name, full_name, clean_addr, c_country)
                cand_ids = [c[0] for c in candidates]
                total_candidates += len(cand_ids)
                cand_f.write(f"{e_id}\t{','.join(cand_ids)}\n")

                start_idx = len(all_feats)
                for cand_id, cand_core, cand_full, cand_addr in candidates:
                    feats = compute_pairwise_features(
                        s1_name_core=core_name,
                        s1_name_full=full_name,
                        s1_addr=clean_addr,
                        s23_name_core=cand_core,
                        s23_name_full=cand_full,
                        s23_addr=cand_addr
                    )
                    all_feats.append(feats)
                end_idx = len(all_feats)
                entity_cand_slices.append((e_id, candidates, start_idx, end_idx))

            # Batch score all candidate pairs in one C++ matrix call
            all_scores = model.score_batch(all_feats)

            for e_id, candidates, start_idx, end_idx in entity_cand_slices:
                cand_scores = all_scores[start_idx:end_idx]
                scored = []
                for (cand_id, _, _, _), sc in zip(candidates, cand_scores):
                    if sc >= model.threshold:
                        scored.append((sc, cand_id))
                scored.sort(key=lambda x: x[0], reverse=True)
                matched_ids = [cid for _, cid in scored]
                total_matches += len(matched_ids)
                match_f.write(f"{e_id}\t{','.join(matched_ids)}\n")
                s1_count += 1
                if s1_count % 10000 == 0:
                    print(f"  Processed {s1_count:,} S1 entities ({time.time() - t0:.1f}s) | Matches found: {total_matches:,}")

            # Flush immediately to disk to prevent buffered RAM buildup
            match_f.flush()
            cand_f.flush()

        # Out-of-core chunk streaming using Polars collect_batches
        if has_polars:
            s1_batches = pl.scan_csv(source1_path, separator='\t').collect_batches(chunk_size=CHUNK_SIZE)
            for batch in s1_batches:
                b_eids = batch['entity_id'].to_list()
                b_names = batch['business_name'].to_list()
                b_addrs = batch['business_address'].to_list()
                b_countries = batch['country'].to_list()
                del batch # Free batch memory immediately

                if limit and s1_count + len(b_eids) > limit:
                    cut = limit - s1_count
                    b_eids, b_names, b_addrs, b_countries = b_eids[:cut], b_names[:cut], b_addrs[:cut], b_countries[:cut]

                process_chunk(b_eids, b_names, b_addrs, b_countries)

                if limit and s1_count >= limit:
                    break
        else:
            # Pandas / standard line chunk fallback
            import pandas as pd
            for chunk in pd.read_csv(source1_path, sep='\t', chunksize=CHUNK_SIZE, dtype=str, keep_default_na=False):
                b_eids = chunk['entity_id'].tolist()
                b_names = chunk['business_name'].tolist()
                b_addrs = chunk['business_address'].tolist()
                b_countries = chunk['country'].tolist()
                del chunk

                if limit and s1_count + len(b_eids) > limit:
                    cut = limit - s1_count
                    b_eids, b_names, b_addrs, b_countries = b_eids[:cut], b_names[:cut], b_addrs[:cut], b_countries[:cut]

                process_chunk(b_eids, b_names, b_addrs, b_countries)

                if limit and s1_count >= limit:
                    break

    print("\n" + "=" * 70)
    print(f"Pipeline completed successfully in {time.time() - t0:.1f}s!")
    print(f"Total Source 1 entities processed: {s1_count:,}")
    print(f"Total candidate pairs generated:   {total_candidates:,}")
    print(f"Total matches predicted:           {total_matches:,}")
    print(f"Matching results saved to:         {matching_tsv_path}")
    print(f"Candidate pairs saved to:          {candidate_tsv_path}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument("--data-dir", type=str, default="dataset/test", help="Path to input TSV datasets")
    parser.add_argument("--output-dir", type=str, default="output", help="Path to write output TSVs")
    parser.add_argument("--prefix", type=str, default="test", choices=["test", "train"], help="Dataset file prefix")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of S1 entities to process")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold for matching (default: loaded from model)")
    parser.add_argument("--max-candidates", type=int, default=25, help="Maximum candidates to retrieve per entity")
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        prefix=args.prefix,
        limit=args.limit,
        threshold=args.threshold,
        max_candidates=args.max_candidates
    )
