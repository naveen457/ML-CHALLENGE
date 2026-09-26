import re
import unicodedata
from typing import Dict, Any, Set, List

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

DIGIT_REGEX = re.compile(r'\b\d+\b')

def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from text."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def jaccard_similarity(set_a: Set[Any], set_b: Set[Any]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return float(intersection) / union if union > 0 else 0.0

def token_overlap_ratio(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Compute token overlap ratio relative to the shorter string."""
    set_a, set_b = set(tokens_a), set(tokens_b)
    if not set_a or not set_b:
        return 0.0
    common = len(set_a.intersection(set_b))
    return float(common) / min(len(set_a), len(set_b))

def extract_numbers(text: str) -> Set[str]:
    """Extract numeric tokens (building numbers, postal codes)."""
    return set(DIGIT_REGEX.findall(text))

def compute_pairwise_features(
    s1_name_core: str,
    s1_name_full: str,
    s1_addr: str,
    s23_name_core: str,
    s23_name_full: str,
    s23_addr: str
) -> Dict[str, float]:
    """
    Computes a comprehensive, high-discrimination feature vector for (S1, S2/S3) pair.
    """
    s1_tokens = s1_name_core.split()
    s23_tokens = s23_name_core.split()
    
    s1_token_set = set(s1_tokens)
    s23_token_set = set(s23_tokens)
    
    name_jaccard = jaccard_similarity(s1_token_set, s23_token_set)
    name_overlap = token_overlap_ratio(s1_tokens, s23_tokens)
    
    # Character 3-gram similarity
    s1_ngrams = get_char_ngrams(s1_name_core, 3)
    s23_ngrams = get_char_ngrams(s23_name_core, 3)
    name_ngram_jaccard = jaccard_similarity(s1_ngrams, s23_ngrams)
    
    # Exact and concatenation matches
    exact_core_match = 1.0 if s1_name_core and s1_name_core == s23_name_core else 0.0
    exact_full_match = 1.0 if s1_name_full and s1_name_full == s23_name_full else 0.0
    
    s1_compact = s1_name_core.replace(' ', '')
    s23_compact = s23_name_core.replace(' ', '')
    concat_match = 1.0 if (s1_compact and s23_compact and s1_compact == s23_compact) else 0.0
    
    prefix_match = 1.0 if (s1_name_core and s23_name_core and 
                          (s1_name_core.startswith(s23_name_core) or s23_name_core.startswith(s1_name_core))) else 0.0

    len_diff = abs(len(s1_name_core) - len(s23_name_core)) / max(len(s1_name_core), len(s23_name_core), 1)

    # String distance ratios
    if HAS_RAPIDFUZZ:
        ratio = fuzz.ratio(s1_name_core, s23_name_core) / 100.0
        token_sort_ratio = fuzz.token_sort_ratio(s1_name_full, s23_name_full) / 100.0
        token_set_ratio = fuzz.token_set_ratio(s1_name_full, s23_name_full) / 100.0
    else:
        import difflib
        ratio = difflib.SequenceMatcher(None, s1_name_core, s23_name_core).ratio()
        s1_sorted = " ".join(sorted(s1_name_full.split()))
        s23_sorted = " ".join(sorted(s23_name_full.split()))
        token_sort_ratio = difflib.SequenceMatcher(None, s1_sorted, s23_sorted).ratio()
        # token set overlap approximation
        common_tokens = " ".join(sorted(set(s1_name_full.split()).intersection(set(s23_name_full.split()))))
        token_set_ratio = max(ratio, difflib.SequenceMatcher(None, s1_sorted, common_tokens).ratio() if common_tokens else 0.0)

    # Address features
    has_both_address = 1.0 if (s1_addr and s23_addr) else 0.0
    s1_addr_tokens = set(s1_addr.split())
    s23_addr_tokens = set(s23_addr.split())
    addr_jaccard = jaccard_similarity(s1_addr_tokens, s23_addr_tokens)
    
    # Numerical consistency (Crucial Precision Filter)
    s1_nums = extract_numbers(s1_addr)
    s23_nums = extract_numbers(s23_addr)
    addr_num_jaccard = jaccard_similarity(s1_nums, s23_nums)
    
    addr_has_common_num = 1.0 if (s1_nums and s23_nums and len(s1_nums.intersection(s23_nums)) > 0) else 0.0
    # Conflicting numbers: both have numbers, but NONE match (strong negative indicator)
    addr_conflicting_num = 1.0 if (s1_nums and s23_nums and len(s1_nums.intersection(s23_nums)) == 0) else 0.0

    # Best overall name similarity
    max_name_sim = max(ratio, token_sort_ratio, token_set_ratio, name_overlap)

    # Multi-tenant / Shared Building Conflict:
    # Addresses are identical or highly overlapping, but names have near-zero similarity
    multi_tenant_conflict = 1.0 if (
        has_both_address > 0.5 and 
        addr_jaccard >= 0.35 and 
        max_name_sim < 0.35 and 
        concat_match < 0.5 and 
        exact_core_match < 0.5
    ) else 0.0

    return {
        'name_jaccard': name_jaccard,
        'name_overlap': name_overlap,
        'name_ngram_jaccard': name_ngram_jaccard,
        'name_ratio': ratio,
        'token_sort_ratio': token_sort_ratio,
        'token_set_ratio': token_set_ratio,
        'max_name_sim': max_name_sim,
        'exact_core_match': exact_core_match,
        'exact_full_match': exact_full_match,
        'concat_match': concat_match,
        'prefix_match': prefix_match,
        'len_diff': len_diff,
        'has_both_address': has_both_address,
        'addr_jaccard': addr_jaccard,
        'addr_num_jaccard': addr_num_jaccard,
        'addr_has_common_num': addr_has_common_num,
        'addr_conflicting_num': addr_conflicting_num,
        'multi_tenant_conflict': multi_tenant_conflict
    }
