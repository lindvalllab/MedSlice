import pandas as pd
import numpy as np

METRICS = ['EM', 'Precision', 'Recall', 'F1_Score']

def compute_metrics(df, start_col, end_col, start_pred_col, end_pred_col):
    """
    Compute evaluation metrics (EM, Precision, Recall, F1 Score) for predictions 
    compared to ground truth spans.

    :param df: DataFrame containing ground truth and predicted start/end indices.
    :param start_col: Column name for ground truth start indices.
    :param end_col: Column name for ground truth end indices.
    :param start_pred_col: Column name for predicted start indices.
    :param end_pred_col: Column name for predicted end indices.
    :return: DataFrame with metrics (EM, Precision, Recall, F1 Score) for each row.
    """
    def calc_metrics(row):
        gt_start, gt_end = row[start_col], row[end_col]
        pred_start, pred_end = row[start_pred_col], row[end_pred_col]
        
        # If gt is empty, then pred being empty is a success
        if pd.isna(gt_start) and pd.isna(gt_end) and pd.isna(pred_start) and pd.isna(pred_end):
            return pd.Series([1, 1, 1, 1], index=METRICS)
        # If a pred is missing this row fails
        if pd.isna(pred_start) or pd.isna(pred_end):
            return pd.Series([0, 0, 0, 0], index=METRICS)
        
        # Calculate metrics
        em = int(gt_start == pred_start and gt_end == pred_end)
        intersection = max(0, min(gt_end, pred_end) - max(gt_start, pred_start) + 1)
        pred_len = pred_end - pred_start + 1
        gt_len = gt_end - gt_start + 1
        precision = intersection / pred_len if pred_len > 0 else 0
        recall = intersection / gt_len if gt_len > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return pd.Series([em, precision, recall, f1_score], index=METRICS)

    # Compute metrics for each row
    result_df = df[[start_col, end_col, start_pred_col, end_pred_col]].copy()
    result_df[METRICS] = df.apply(calc_metrics, axis=1)
    return result_df


def scorer_section(df, section):
    """
    Compute metrics for a specific section (e.g., "HPI_Interval_Hx" or "A&P").

    :param df: DataFrame containing ground truth and predicted indices for the section.
    :param section: Section name used to determine column names for ground truth 
                    and predictions (e.g., "{section}_start_gt").
    :return: DataFrame with metrics for the section.
    """
    start_col = f"{section}_start_gt"
    end_col = f"{section}_end_gt"
    start_pred_col = f"{section}_start_pred"
    end_pred_col = f"{section}_end_pred"

    # Ensure required columns exist
    assert f"{section}_start_gt" in df.columns, f"Column '{section}_start_gt' not found in df.columns."
    assert f"{section}_end_gt" in df.columns, f"Column '{section}_end_gt' not found in df.columns."
    assert f"{section}_start_pred" in df.columns, f"Column '{section}_start_pred' not found in df.columns."
    assert f"{section}_end_pred" in df.columns, f"Column '{section}_end_pred' not found in df.columns."

    return compute_metrics(df, start_col, end_col, start_pred_col, end_pred_col)


def scorer_row(df):
    """
    Compute metrics for all sections and add them as new columns to the DataFrame.

    :param df: DataFrame containing ground truth and predicted indices for all sections.
    :return: DataFrame with added columns for metrics (EM, Precision, Recall, F1 Score) 
             for each section (e.g., "RCH_EM", "AP_F1_Score").
    """
    rch_scores = scorer_section(df, "RCH")
    ap_scores = scorer_section(df, "AP")
    
    for metric in METRICS:
        df[f'RCH_{metric}'] = rch_scores[metric]
        df[f'AP_{metric}'] = ap_scores[metric]
    return df

def scorer(df):
    """
    Compute average metrics for all sections and prints them.

    :param df: DataFrame containing ground truth and predicted indices for all sections..
    """
    scored_df = scorer_row(df)
    metrics_data = {metric: [] for metric in METRICS}
    
    # Collect metrics for RCH and AP
    for metric in METRICS:
        metrics_data[metric].append(scored_df[f'RCH_{metric}'].mean())
        metrics_data[metric].append(scored_df[f'AP_{metric}'].mean())

    # Create a DataFrame
    result_df = pd.DataFrame(
        metrics_data, 
        index=['RCH', 'AP']
    ).transpose()

    return result_df