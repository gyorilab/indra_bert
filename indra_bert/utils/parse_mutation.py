"""
Mutation parsing utilities for converting mutation text to INDRA-compatible format.

This module provides functions to parse various mutation mention formats
and convert them to INDRA's MutCondition format.
"""

import re
from typing import Dict, Optional, List, Union


def parse_mutation_text(mutation_text: str) -> Optional[Dict[str, str]]:
    """
    Parse mutation text to extract position, residue_from, and residue_to.
    
    Supports various mutation formats:
    - Single letter mutations: "L858R", "K77R", "V600E"
    - HGVS format: "p.L858R", "p.K77R" 
    - Deletions: "L858del", "F508del", "delF508"
    - Insertions: "ins", "insL", "ins858"
    - Frameshifts: "fs", "frameshift"
    - Stop codons: "X", "*", "Ter", "STOP"
    
    Args:
        mutation_text: Raw mutation text (e.g., "L858R", "p.L858R", "F508del")
        
    Returns:
        Dictionary with 'position', 'residue_from', 'residue_to' keys,
        or None if parsing fails
    """
    if not mutation_text or not isinstance(mutation_text, str):
        return None
    
    mutation_text = mutation_text.strip()
    
    # Pattern 1: HGVS format with p. prefix (p.L858R, p.K77R)
    pattern1 = r'^p\.([A-Z])(\d+)([A-Z*X])$'
    match = re.match(pattern1, mutation_text)
    if match:
        return {
            'position': match.group(2),
            'residue_from': match.group(1),
            'residue_to': _normalize_residue_to(match.group(3))
        }
    
    # Pattern 2: Single letter mutations without p. prefix (L858R, K77R, V600E)
    pattern2 = r'^([A-Z])(\d+)([A-Z*X])$'
    match = re.match(pattern2, mutation_text)
    if match:
        return {
            'position': match.group(2),
            'residue_from': match.group(1),
            'residue_to': _normalize_residue_to(match.group(3))
        }
    
    # Pattern 3: Deletions with del suffix (F508del, L858del)
    pattern3 = r'^([A-Z])(\d+)del$'
    match = re.match(pattern3, mutation_text)
    if match:
        return {
            'position': match.group(2),
            'residue_from': match.group(1),
            'residue_to': None  # Deletion
        }
    
    # Pattern 4: Deletions with del prefix (delF508)
    pattern4 = r'^del([A-Z])(\d+)$'
    match = re.match(pattern4, mutation_text)
    if match:
        return {
            'position': match.group(2),
            'residue_from': match.group(1),
            'residue_to': None  # Deletion
        }
    
    # Pattern 5: Insertions (ins, insL, ins858L)
    pattern5 = r'^ins(\d*)([A-Z]?)$'
    match = re.match(pattern5, mutation_text)
    if match:
        position = match.group(1) if match.group(1) else "unknown"
        residue_to = match.group(2) if match.group(2) else "unknown"
        return {
            'position': position,
            'residue_from': None,  # Unknown for insertions
            'residue_to': residue_to
        }
    
    # Pattern 6: Frameshifts (fs, frameshift, fs123)
    pattern6 = r'^(fs|frameshift)(\d*)$'
    match = re.match(pattern6, mutation_text, re.IGNORECASE)
    if match:
        position = match.group(2) if match.group(2) else "unknown"
        return {
            'position': position,
            'residue_from': None,  # Unknown for frameshifts
            'residue_to': 'fs'  # Special marker for frameshift
        }
    
    # Pattern 7: Stop codons (X, *, Ter, STOP)
    pattern7 = r'^([A-Z])(\d+)(X|\*|Ter|STOP)$'
    match = re.match(pattern7, mutation_text, re.IGNORECASE)
    if match:
        return {
            'position': match.group(2),
            'residue_from': match.group(1),
            'residue_to': '*'  # Normalized stop codon
        }
    
    # Pattern 8: Complex deletions (del123-456)
    pattern8 = r'^del(\d+)-(\d+)$'
    match = re.match(pattern8, mutation_text)
    if match:
        # For range deletions, use the start position
        return {
            'position': f"{match.group(1)}-{match.group(2)}",  # Range format
            'residue_from': None,  # Unknown for range deletions
            'residue_to': None  # Deletion
        }
    
    # If no pattern matches, return None
    return None


def _normalize_residue_to(residue: str) -> Optional[str]:
    """
    Normalize residue_to values to standard format.
    
    Args:
        residue: Raw residue string
        
    Returns:
        Normalized residue string or None
    """
    if not residue:
        return None
    
    residue = residue.upper()
    
    # Normalize stop codon representations
    if residue in ['*', 'X', 'TER', 'STOP']:
        return '*'
    
    # Return single letter amino acid code
    if len(residue) == 1 and residue.isalpha():
        return residue
    
    return residue


def parse_mutations_list(mutations: List[Dict[str, Union[int, str]]]) -> List[Dict[str, str]]:
    """
    Parse a list of mutation dictionaries with 'text' field.
    
    Args:
        mutations: List of mutation dicts with 'text' field
        (e.g., [{"start": 100, "end": 105, "text": "L858R"}])
        
    Returns:
        List of parsed mutation dicts with INDRA-compatible format
    """
    parsed_mutations = []
    
    for mutation in mutations:
        mutation_text = mutation.get('text', '')
        parsed = parse_mutation_text(mutation_text)
        
        if parsed:
            # Add original position info if available
            parsed['original_start'] = mutation.get('start')
            parsed['original_end'] = mutation.get('end')
            parsed['original_text'] = mutation_text
            parsed_mutations.append(parsed)
        else:
            # If parsing fails, create a fallback entry
            parsed_mutations.append({
                'position': 'unknown',
                'residue_from': None,
                'residue_to': None,
                'original_start': mutation.get('start'),
                'original_end': mutation.get('end'),
                'original_text': mutation_text,
                'parse_error': True
            })
    
    return parsed_mutations


def convert_to_indra_mutations(mutations: List[Dict[str, Union[int, str]]]) -> List[Dict[str, str]]:
    """
    Convert mutation list to INDRA-compatible format.
    
    This is the main function to be called from indra_bert.py
    
    Args:
        mutations: List of mutation dicts from agent mutation detector
        
    Returns:
        List of mutation dicts in INDRA MutCondition format
    """
    return parse_mutations_list(mutations)


# Test cases for validation
if __name__ == "__main__":
    test_cases = [
        "L858R",      # Single letter mutation
        "p.L858R",    # HGVS format
        "F508del",    # Deletion
        "delF508",    # Deletion with del prefix
        "K77*",       # Stop codon
        "p.V600E",    # HGVS with different mutation
        "ins",        # Insertion
        "fs",         # Frameshift
        "del123-456", # Range deletion
    ]
    
    print("Testing mutation parsing:")
    for test_case in test_cases:
        result = parse_mutation_text(test_case)
        print(f"{test_case:12} -> {result}")
