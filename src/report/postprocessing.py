import pandas as pd

def extract_section_from_indices(df, text_col: str, start_col: str, end_col: str):
    """
    Extract a substring from a text column using float-based start and end indices.

    :param df: pandas.DataFrame
        The input DataFrame containing the text column and start/end index columns.
    :param text_col: str
        The name of the column containing strings from which substrings will be extracted.
    :param start_col: str
        The name of the column containing float start indices.
    :param end_col: str
        The name of the column containing float end indices.
    :return: pandas.Series
        A Series containing extracted substrings.
        If start or end is pd.NA, the extracted value will be pd.NA.
    """
    # Check if all columns exist in the DataFrame
    assert text_col in df.columns, f"Column '{text_col}' not found in the DataFrame."
    assert start_col in df.columns, f"Column '{start_col}' not found in the DataFrame."
    assert end_col in df.columns, f"Column '{end_col}' not found in the DataFrame."

    def extract_section(row):
        text = row[text_col]
        start = row[start_col]
        end = row[end_col]

        # Return pd.NA if start or end is missing
        if pd.isna(start) or pd.isna(end) or pd.isna(text):
            return pd.NA

        # Ensure indices are integers and extract the substring
        start_int = int(start)
        end_int = int(end)
        return text[start_int:end_int]

    # Apply the extraction logic row-wise and return the result as a Series
    return df.apply(extract_section, axis=1)


def postprocessing(df):
    """
    Add two new columns 'RCH_pred' and 'AP_pred' to the DataFrame.
    
    'RCH_pred' extracts substrings from 'note_text' using 'RCH_start_pred' and 'RCH_end_pred'.
    'AP_pred' extracts substrings from 'note_text' using 'AP_start_pred' and 'AP_end_pred'.
    
    :param df: pandas.DataFrame
        Input DataFrame containing the following columns:
        - 'note_text': Full text where substrings are extracted.
        - 'RCH_start_pred' and 'RCH_end_pred': Start and end indices for RCH section.
        - 'AP_start_pred' and 'AP_end_pred': Start and end indices for AP section.
    :return: pandas.DataFrame
        The original DataFrame with two new columns:
        - 'RCH_pred': Extracted substrings for RCH.
        - 'AP_pred': Extracted substrings for AP.
    """

    # Add the 'RCH_pred' column
    df['RCH_pred'] = extract_section_from_indices(df, text_col='note_text',
                                                  start_col='RCH_start_pred',
                                                  end_col='RCH_end_pred')
    
    # Add the 'AP_pred' column
    df['AP_pred'] = extract_section_from_indices(df, text_col='note_text',
                                                 start_col='AP_start_pred',
                                                 end_col='AP_end_pred')
    return df

