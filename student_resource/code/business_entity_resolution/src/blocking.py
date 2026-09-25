import sys
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Iterator
import re

# Stop words / high-frequency terms to ignore during inverted index candidate generation
STOP_WORDS = {
    'the', 'and', 'for', 'of', 'in', 'on', 'at', 'to', 'a', 'an', 'by',
    'road', 'rd', 'street', 'st', 'avenue', 'ave', 'lane', 'ln', 'block',
    'floor', 'near', 'opp', 'opposite', 'building', 'bldg', 'complex',
    'sector', 'phase', 'nagar', 'city', 'post', 'state'
}

class InvertedIndexBlocker:
    """
    Candidate generator using Inverted Index on core name tokens and address signals,
    partitioned strictly by Country.
    """
    def __init__(self, max_candidates_per_entity: int = 15, max_token_frequency: int = 5000):
        self.max_candidates = max_candidates_per_entity
        self.max_token_freq = max_token_frequency
        # index: country -> token -> list of candidate record indices
        self.country_indices: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        # metadata store: country -> list of record tuples: (entity_id, core_name, full_name, addr)
        self.candidate_records: Dict[str, List[Tuple[str, str, str, str]]] = defaultdict(list)

    def add_candidate(self, entity_id: str, core_name: str, full_name: str, address: str, country: str):
        """Index a candidate record (from Source 2 or Source 3)."""
        c_records = self.candidate_records[country]
        idx = len(c_records)
        c_records.append((entity_id, core_name, full_name, address))
        
        # Tokenize core name
        tokens = set(core_name.split())
        for tok in tokens:
            if len(tok) >= 3 and tok not in STOP_WORDS:
                self.country_indices[country][tok].append(idx)

        # If name is a concatenated single token (like a domain or compound name), index prefix
        if len(tokens) == 1:
            tok = next(iter(tokens))
            if len(tok) >= 6:
                self.country_indices[country][tok[:5]].append(idx)
                
        # Index distinctive address numbers (e.g. street numbers, PIN codes)
        if address:
            addr_nums = [n for n in address.split() if n.isdigit() and len(n) >= 2]
            for num in addr_nums[:2]:
                self.country_indices[country][f"num_{num}"].append(idx)

    def prune_high_frequency_tokens(self):
        """Prune inverted lists that are too long to prevent combinatorial explosion."""
        for country, index in self.country_indices.items():
            to_remove = [tok for tok, lst in index.items() if len(lst) > self.max_token_freq]
            for tok in to_remove:
                del index[tok]

    def query(self, core_name: str, full_name: str, address: str, country: str) -> List[Tuple[str, str, str, str]]:
        """
        Queries top candidate records for a Source 1 entity.
        Returns list of (candidate_entity_id, candidate_core_name, candidate_full_name, candidate_addr)
        """
        if country not in self.country_indices:
            return []
            
        index = self.country_indices[country]
        c_records = self.candidate_records[country]
        
        tokens = [tok for tok in core_name.split() if len(tok) >= 3 and tok not in STOP_WORDS]
        if not tokens:
            tokens = [tok for tok in full_name.split() if len(tok) >= 3 and tok not in STOP_WORDS]

        # Check for compound / domain prefix
        for tok in tokens:
            if len(tok) >= 6:
                tokens.append(tok[:5])

        # Add address numbers
        if address:
            addr_nums = [n for n in address.split() if n.isdigit() and len(n) >= 2]
            for num in addr_nums[:2]:
                tokens.append(f"num_{num}")
            
        hit_counts: Dict[int, int] = Counter()
        for tok in tokens:
            if tok in index:
                for cand_idx in index[tok]:
                    hit_counts[cand_idx] += 1
                    
        if not hit_counts:
            return []
            
        # Top-K candidate indices by token hit frequency
        top_indices = [idx for idx, _ in hit_counts.most_common(self.max_candidates)]
        return [c_records[idx] for idx in top_indices]
