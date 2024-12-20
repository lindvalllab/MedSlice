import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
import re

def find_substr_match(note, substr, start_pos=0):
    """
    Finds the match object of substr in note, allowing for variable spacing.

    Parameters:
    - note (str): The large string where the search is performed.
    - substr (str): The substring to find within note.
    - start_pos (int): The position in note to start the search from.

    Returns:
    - re.Match: The match object if found; otherwise, None.
    """
    # Step 1: Escape substr to handle special regex characters
    escaped_substr = re.escape(substr)
    
    # Step 2: Replace escaped spaces with a regex pattern to match one or more whitespace characters
    pattern = escaped_substr.replace(r'\ ', r'\s+')
    
    # Step 3: Compile the regex pattern for efficiency, starting the search from start_pos
    # The regex should search from start_pos, so we slice the note
    # However, to get the correct index in the original note, we need to account for the offset
    sliced_note = note[start_pos:]
    regex = re.compile(pattern)
    
    # Step 4: Search for the pattern in the sliced_note
    match = regex.search(sliced_note)
    
    # Step 5: If a match is found, adjust the start and end indices relative to the original note
    if match:
        adjusted_start = start_pos + match.start()
        adjusted_end = start_pos + match.end()
        return match, adjusted_start, adjusted_end
    else:
        return None, -1, -1

def fuzzy_matching_per_row(row, section, fuzz_threshold: int = 80):
    """
    Extracts character indices from a clinical note that correspond to a section defined by 
    start and end strings. Fuzzy matching is applied to find the closest matches within the note.

    :param row: A Pandas Series representing a row from a DataFrame. Must have:
                 - 'note' column for the full note text.
                 - start_col for the start string.
                 - end_col for the end string.
    :param section: The section name (e.g., 'rch', 'ap').
    :param end_col: The column name in 'row' containing the end string.
    :param fuzz_threshold: The minimum partial ratio score (0-100) needed to consider a fuzzy match valid.
    :return: A tuple (start_index, end_index) indicating the character-level indices in the note text 
             that encompass the extracted section. Returns (NaN, NaN) if no valid extraction is found.
    """
    note = row['note']
    start_candidate = row[f"{section}_start_pred"]
    end_candidate = row[f"{section}_end_pred"]

    if pd.isna(start_candidate) or pd.isna(end_candidate):
        return np.nan, np.nan

    # Generate overlapping sequences of 5 words for fuzzy matching
    note_words = note.split()
    overlapping_sequences = [' '.join(note_words[i:i+5]) for i in range(len(note_words) - 4)]

    # Fuzzy match the start candidate
    start_match_result = process.extractOne(start_candidate, overlapping_sequences, scorer=fuzz.partial_ratio, score_cutoff=fuzz_threshold)
    if start_match_result:
        best_start_match = start_match_result[0]
    else:
        return np.nan, np.nan

    # Fuzzy match the end candidate
    end_match_result = process.extractOne(end_candidate, overlapping_sequences, scorer=fuzz.partial_ratio, score_cutoff=fuzz_threshold)
    if end_match_result:
        best_end_match = end_match_result[0]
    else:
        return np.nan, np.nan

    # Find the exact substring matches in the note text
    start_found, start_index, start_index_end = find_substr_match(note, best_start_match)
    if start_found:
        end_found, end_index, end_index_end = find_substr_match(note, best_end_match, start_index)
    else:
        return np.nan, np.nan

    # Validate the indices
    if start_index == -1 or end_index == -1 or start_index >= end_index:
        return np.nan, np.nan

    return start_index, end_index_end

def get_pred_indexes_section(df, section, fuzz_threshold: int = 80):
    """
    Apply fuzzy matching to assign predicted start and end strings for a section.

    :param df: DataFrame with 'note' and prediction columns.
    :param section: The section name (e.g., 'RCH', 'AP').
    :param fuzz_threshold: Minimum score for fuzzy matching.
    :return: DataFrame with new columns '{section}_start_pred' and '{section}_end_pred'.
    """
    assert f"{section}_start_pred" in df.columns, f"Column '{section}_start_pred' not found in df.columns."
    assert f"{section}_end_pred" in df.columns, f"Column '{section}_end_pred' not found in df.columns."
    df[f'{section}_start_pred'], df[f'{section}_end_pred'] = zip(*df.apply(lambda row: fuzzy_matching_per_row(row,
                                                                                                              section,
                                                                                                              fuzz_threshold), axis=1))
    return df

def get_pred_indexes(df, fuzz_threshold: int = 80):
    """
    Apply fuzzy matching to assign predicted start and end strings for both RCH and AP sections.

    :param df: DataFrame with 'note' and prediction columns.
    :param fuzz_threshold: Minimum score for fuzzy matching.
    :return: DataFrame with new columns '{section}_start_pred' and '{section}_end_pred' for both section.
    """
    df = get_pred_indexes_section(df, "RCH", fuzz_threshold)
    df = get_pred_indexes_section(df, "AP", fuzz_threshold)
    return df