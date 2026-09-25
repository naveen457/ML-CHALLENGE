# Business Entity Resolution Pipeline

This repository contains the end-to-end runnable pipeline for the Amazon ML Challenge: Business Entity Resolution.

## Pipeline Architecture
1. **Preprocessing (`src/preprocess.py`)**: Normalizes business names, strips corporate legal suffixes (`Inc`, `Corp`, `LLC`, `Pvt Ltd`, `SARL`, `SASU`), standardizes addresses, and parses country codes.
2. **Candidate Generation / Blocking (`src/blocking.py`)**: Partitions entities by country to constrain comparisons, builds a high-throughput Inverted Index on core name n-grams and tokens, and retrieves the top-$K$ candidate entities from Sources 2 & 3 for each Source 1 reference entity.
3. **Pairwise Feature Engineering (`src/features.py`)**: Computes character n-gram similarities, token Jaccard scores, RapidFuzz string distances, and address component/number overlaps.
4. **Scoring & Model (`src/model.py`)**: Precision-weighted decision scoring optimized specifically for Macro $F_{0.5}$ and accurate singleton preservation.
5. **Pipeline Runner (`src/main.py`)**: Coordinates candidate generation and matching inference, outputting `output/candidate_pairs.tsv` and `output/matching_results.tsv`.

## Setup & Requirements
```bash
pip install -r requirements.txt
```

## Running the Pipeline

To run the pipeline on the test dataset and generate submission outputs:
```bash
python src/main.py --data-dir ../../dataset/test --output-dir ../../output --prefix test
```

### Fast Test / Dry-Run (e.g. 5,000 entities):
```bash
python src/main.py --data-dir ../../dataset/test --output-dir ../../output --prefix test --limit 5000
```

## Output Validation
Validate the generated outputs against the official competition requirements:
```bash
python ../../utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../dataset/test
```
