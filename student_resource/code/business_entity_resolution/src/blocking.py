import array
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Iterator
import re

# Stop words to ignore during inverted index candidate generation
STOP_WORDS = {
    'the', 'and', 'for', 'of', 'in', 'on', 'at', 'to', 'a', 'an', 'by',
    'road', 'rd', 'street', 'st', 'avenue', 'ave', 'lane', 'ln', 'block',
    'floor', 'near', 'opp', 'opposite', 'building', 'bldg', 'complex',
    'sector', 'phase', 'nagar', 'city', 'post', 'state'
}

ADDR_STOPS = {
    'street', 'st', 'road', 'rd', 'avenue', 'ave', 'lane', 'ln', 'drive', 'dr',
    'court', 'ct', 'boulevard', 'blvd', 'unit', 'suite', 'ste', 'floor', 'fl',
    'apartment', 'apt', 'building', 'bldg', 'room', 'rm', 'block', 'shop',
    'near', 'opp', 'opposite', 'post', 'box', 'city', 'state', 'country',
    'north', 'south', 'east', 'west', 'ground', 'no', 'hno', 'plot',
    'uttar', 'pradesh', 'maharashtra', 'delhi', 'karnataka', 'tamil', 'nadu'
}

def get_address_blocking_keys(addr_str: str) -> List[str]:
    """Fast extraction of PIN codes and Locality+Number composite keys with typo tolerance."""
    if not addr_str:
        return []
    tokens = addr_str.split()
    keys = []
    loc_tokens = []
    first_num = None
    for tok in tokens:
        if tok.isdigit():
            if first_num is None:
                first_num = tok
            if 5 <= len(tok) <= 6:
                keys.append(f"pin_{tok}")
        elif len(tok) >= 4 and tok not in ADDR_STOPS and len(loc_tokens) < 2:
            loc_tokens.append(tok)
            
    if first_num and loc_tokens:
        for loc in loc_tokens:
            keys.append(f"loc_{loc}_{first_num}")
    return keys

class InvertedIndexBlocker:
    """
    High-Throughput Dual-Key (Name + Address Composite) candidate generator.
    Uses compact 32-bit integer arrays (array.array('I')) for out-of-core memory efficiency.
    """
    def __init__(self, max_candidates_per_entity: int = 25, max_token_frequency: int = 5000):
        self.max_candidates = max_candidates_per_entity
        self.max_token_freq = max_token_frequency
        # index: country -> token -> compact 32-bit uint array (4 bytes vs 36 bytes in Python list)
        self.country_indices: Dict[str, Dict[str, array.array]] = defaultdict(lambda: defaultdict(lambda: array.array('I')))
        # metadata store: country -> list of record tuples: (entity_id, core_name, full_name, addr)
        self.candidate_records: Dict[str, List[Tuple[str, str, str, str]]] = defaultdict(list)

    def add_candidate(self, entity_id: str, core_name: str, full_name: str, address: str, country: str):
        """Index a candidate record from Source 2 or Source 3."""
        c_records = self.candidate_records[country]
        idx = len(c_records)
        c_records.append((entity_id, core_name, full_name, address))
        
        # 1. Core name tokens
        tokens = set(core_name.split())
        for tok in tokens:
            if len(tok) >= 3 and tok not in STOP_WORDS:
                self.country_indices[country][tok].append(idx)

        # 2. Compound / Domain prefix
        compact = "".join(core_name.split())
        if len(compact) >= 5:
            self.country_indices[country][compact[:5]].append(idx)
                
        # 3. High-precision address keys (PIN / Locality-Number)
        for a_key in get_address_blocking_keys(address):
            self.country_indices[country][a_key].append(idx)

    def prune_high_frequency_tokens(self):
        """Prune inverted lists that are too common to avoid combinatorial explosion."""
        for country, index in self.country_indices.items():
            to_remove = [tok for tok, lst in index.items() if len(lst) > self.max_token_freq]
            for tok in to_remove:
                del index[tok]

    def query(self, core_name: str, full_name: str, address: str, country: str) -> List[Tuple[str, str, str, str]]:
        """
        Queries top candidates across Name and Address composite signals.
        """
        if country not in self.country_indices:
            return []
            
        index = self.country_indices[country]
        c_records = self.candidate_records[country]
        
        # Collect query keys: Name tokens + Address keys
        tokens = [tok for tok in core_name.split() if len(tok) >= 3 and tok not in STOP_WORDS]
        if not tokens:
            tokens = [tok for tok in full_name.split() if len(tok) >= 3 and tok not in STOP_WORDS]

        compact = "".join(core_name.split())
        if len(compact) >= 5:
            tokens.append(compact[:5])

        tokens.extend(get_address_blocking_keys(address))
            
        hit_counts: Dict[int, int] = Counter()
        for tok in tokens:
            if tok in index:
                for cand_idx in index[tok]:
                    hit_counts[cand_idx] += 1
                    
        if not hit_counts:
            return []
            
        # Top-K candidate indices by hit frequency
        top_indices = [idx for idx, _ in hit_counts.most_common(self.max_candidates)]
        return [c_records[idx] for idx in top_indices]
