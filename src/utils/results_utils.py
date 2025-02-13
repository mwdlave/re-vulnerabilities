import pandas as pd
import numpy as np
from src.utils.file_utils import load_pickle_from_gpu



def compute_region_stats(rankings_list):
    """
    Computes the mean and standard deviation of variation scores across multiple datasets.
    Ensures that all rankings are correctly aligned by region.

    Parameters:
        rankings_list (list of pd.Series): A list where each element is a Pandas Series 
                                           with regions as the index and variation scores as values.

    Returns:
        pd.DataFrame: A DataFrame containing region, mean_score, and std_var.
    """

    # Ensure all rankings contain the same regions by reindexing them with the full region set
    all_regions = sorted(set().union(*[r.index for r in rankings_list]))  # Get unique regions sorted
    rankings_aligned = [r.reindex(all_regions) for r in rankings_list]  # Reindex all rankings

    # Convert list of reindexed Series into a DataFrame
    df = pd.DataFrame(rankings_aligned).T  # Transpose to have regions as rows

    # Compute mean and standard deviation across datasets
    df_stats = pd.DataFrame({
        'region': df.index,
        'mean_score': df.mean(axis=1),
        'std_var': df.std(axis=1)
    }).reset_index(drop=True)

    return df_stats.sort_values('mean_score', ascending=False)


def get_results(file_name, dfs, threshold_logit, mean_b=None, std_b=None):
    """
    Load results from a file and compute various metrics.

    Parameters:
        file_name (str): The file name to load results from.
        dfs (pd.DataFrame): The DataFrame containing the data.
        threshold_logit (float): The logit threshold to determine correctness.
    
    Returns:
        pd.Series: The mean variation score per region.
        pd.Series: The mean absolute variation score per region.
        float: The mean standard deviation of variation scores.
        float: The accuracy of the model.
        float: The mean majority metric.
    """

    results_0 = load_pickle_from_gpu(file_name)
    results_0 = [item.item() for sublist in results_0 for item in sublist]

    dfs['log_diff_0'] = np.array(results_0)

    if mean_b is None:
        mean_b = dfs['log_diff_0'].mean()
    if std_b is None:
        std_b = dfs['log_diff_0'].std()

    # Determine correctness based on logit threshold
    dfs['correct_0'] = dfs.apply(
        lambda row: (row['toxicity'] > 0.5) == (row['log_diff_0'] > threshold_logit), axis=1
    )

    accuracy = dfs['correct_0'].mean()

    # z score normalization
    dfs['log_diff_0_normalized'] = (dfs['log_diff_0'] - mean_b) / std_b

    grouped = dfs.groupby('prompt')['log_diff_0_normalized'].std()
    mean_std_dev = grouped.mean()

    # Compute majority metric per prompt group
    majority_counts = []
    for _, group in dfs.groupby('prompt'):
        above_threshold = (group['log_diff_0'] > threshold_logit).sum()
        below_threshold = len(group) - above_threshold
        majority_counts.append(max(above_threshold, below_threshold))  # Get the higher count

    mean_majority = np.mean(majority_counts)

    dfs['variation_score'] = dfs['log_diff_0_normalized'] - dfs.groupby('prompt')['log_diff_0_normalized'].transform('mean')
    dfs['variation_score_abs'] = dfs['variation_score'].abs()
    return  dfs.groupby('Sub_Region')['variation_score'].mean().sort_values(ascending=False), \
            dfs.groupby('Sub_Region')['variation_score_abs'].mean().sort_values(ascending=False),\
            mean_std_dev,\
            accuracy,\
            mean_majority,\
            mean_b,\
            std_b