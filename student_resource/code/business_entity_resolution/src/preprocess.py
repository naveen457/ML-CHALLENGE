import re
import unicodedata
from typing import Tuple

# Common legal business suffixes across US, India, France, and transliterated Indic
LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'llp', 'pc', 'co', 'company', 'gmbh', 'sa', 'sarl',
    'sasu', 'sas', 'eurl', 'trust', 'group', 'services', 'solutions',
    'holdings', 'enterprises', 'associates', 'industries', 'management',
    # Common transliterated variations
    'praivet', 'praiveta', 'limiteda', 'kampani', 'kaumpani', 'vyapar', 'traders'
}

# Regex to clean protocols and domain extensions while keeping domain names
URL_CLEAN_REGEX = re.compile(r'https?://|www\.', re.IGNORECASE)
DOMAIN_EXT_REGEX = re.compile(r'\.(?:com|org|net|in|fr|co|io|edu|gov|biz|info)\b', re.IGNORECASE)
# Regex for non-alphanumeric characters (keeping spaces and unicode word characters)
PUNCT_REGEX = re.compile(r'[^\w\s]', re.UNICODE)
# Multi-space collapse
SPACE_REGEX = re.compile(r'\s+')

# Pure Python Brahmic script transliterator mapping (covers Devanagari, Bengali, Gurmukhi, Gujarati, Tamil, Telugu, Kannada, Malayalam)
INDIC_OFFSET_MAP = {
    0x01: 'n', 0x02: 'n', 0x03: 'h',
    0x05: 'a', 0x06: 'a', 0x07: 'i', 0x08: 'i', 0x09: 'u', 0x0a: 'u',
    0x0f: 'e', 0x10: 'ai', 0x13: 'o', 0x14: 'au',
    0x15: 'k', 0x16: 'kh', 0x17: 'g', 0x18: 'gh', 0x19: 'ng',
    0x1a: 'ch', 0x1b: 'chh', 0x1c: 'j', 0x1d: 'jh', 0x1e: 'ny',
    0x1f: 't', 0x20: 'th', 0x21: 'd', 0x22: 'dh', 0x23: 'n',
    0x24: 't', 0x25: 'th', 0x26: 'd', 0x27: 'dh', 0x28: 'n',
    0x2a: 'p', 0x2b: 'ph', 0x2c: 'b', 0x2d: 'bh', 0x2e: 'm',
    0x2f: 'y', 0x30: 'r', 0x32: 'l', 0x33: 'l', 0x35: 'v',
    0x36: 'sh', 0x37: 'sh', 0x38: 's', 0x39: 'h',
    0x3e: 'a', 0x3f: 'i', 0x40: 'i', 0x41: 'u', 0x42: 'u',
    0x47: 'e', 0x48: 'ai', 0x4b: 'o', 0x4c: 'au', 0x4d: ''
}
INDIC_BASES = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B80, 0x0C00, 0x0C80, 0x0D00]

def transliterate_indic_to_latin(text: str) -> str:
    """Fast, local transliteration of Indic scripts (Devanagari, Kannada, etc.) to Latin."""
    res = []
    for ch in text:
        cp = ord(ch)
        mapped = False
        for base in INDIC_BASES:
            if base <= cp < base + 0x80:
                off = cp - base
                if off in INDIC_OFFSET_MAP:
                    res.append(INDIC_OFFSET_MAP[off])
                    mapped = True
                    break
        if not mapped:
            res.append(ch)
    return ''.join(res)

def normalize_text(text: str) -> str:
    """Basic unicode normalization, transliteration, and lowercasing."""
    if not isinstance(text, str) or not text.strip():
        return ""
    # Transliterate Indic scripts to Latin characters
    text = transliterate_indic_to_latin(text)
    # NFKD normalization to separate accents from base Latin characters (e.g. French accents: é -> e)
    text = unicodedata.normalize('NFKD', text)
    # Strip protocols and domain extensions (leaving domain name words)
    text = URL_CLEAN_REGEX.sub(' ', text)
    text = DOMAIN_EXT_REGEX.sub(' ', text)
    # Lowercase
    text = text.lower()
    # Strip non-alphanumeric punctuation
    text = PUNCT_REGEX.sub(' ', text)
    # Collapse whitespace
    text = SPACE_REGEX.sub(' ', text).strip()
    return text

def clean_business_name(name: str) -> Tuple[str, str]:
    """
    Cleans business name and separates the core name tokens from legal suffixes.
    Returns:
        (core_name, cleaned_full_name)
    """
    normalized = normalize_text(name)
    if not normalized:
        return "", ""
    
    tokens = normalized.split()
    core_tokens = [tok for tok in tokens if tok not in LEGAL_SUFFIXES]
    
    # If all tokens were legal suffixes, keep the original tokens
    if not core_tokens:
        core_tokens = tokens
        
    core_name = " ".join(core_tokens)
    return core_name, normalized

def clean_address(address: str) -> str:
    """Normalizes address string, transliterates non-Latin scripts, and removes noise."""
    normalized = normalize_text(address)
    return normalized

def clean_country(country: str) -> str:
    """Normalizes country string."""
    if not isinstance(country, str):
        return "UNKNOWN"
    c = country.strip().upper()
    return c if c else "UNKNOWN"
