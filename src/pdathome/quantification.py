import numpy as np

from pdathome.constants import gait_constants as gc, mappings as mp

def extract_features(df, l_groupby, l_metrics, l_aggregates, l_quantiles=[]):
    """
    Extracts features from a DataFrame by aggregating specified metrics.

    Args:
    - df (pd.DataFrame): DataFrame containing the data.
    - l_groupby (list): List of columns to group by.
    - l_metrics (list): List of metrics to aggregate.
    - l_aggregates (list): List of aggregate functions to apply (e.g., median).
    - l_quantiles (list): List of quantiles to compute (default is empty).

    Returns:
    - pd.DataFrame: Aggregated DataFrame with metrics and quantiles.
    """
    # Initialize an empty list to store aggregated DataFrames
    agg_dfs = []

    for metric in l_metrics:
        # Aggregate using specified functions
        df_agg = df.groupby(l_groupby)[metric].agg(l_aggregates).reset_index()
        df_agg.columns = [*l_groupby, *(f"{metric}_{agg}" for agg in l_aggregates)]
        
        # Compute quantiles if specified
        if l_quantiles:
            quantiles = df.groupby(l_groupby)[metric].quantile(l_quantiles).unstack()
            quantiles.columns = [f"{metric}_quantile_{int(q * 100)}" for q in l_quantiles]
            df_agg = df_agg.merge(quantiles.reset_index(), on=l_groupby, how='left')

        agg_dfs.append(df_agg)

    # Merge all aggregated DataFrames
    final_df = agg_dfs[0]
    for agg_df in agg_dfs[1:]:
        final_df = final_df.merge(agg_df, on=l_groupby, how='left')

    return final_df


def initialize_d_quant(df_agg_side, l_measures):
    """Initialize the result dictionary for both overall and segment categories."""
    d_quant = {}
    for med_stage in df_agg_side[gc.columns.PRE_OR_POST].unique():
        d_quant[med_stage] = {}
        for affected_side in [gc.descriptives.MOST_AFFECTED_SIDE, gc.descriptives.LEAST_AFFECTED_SIDE]:
            d_quant[med_stage][affected_side] = {
                'seconds': {'overall': 0},
                'values': {measure: {'overall': 0} for measure in l_measures}
            }
    return d_quant


def populate_overall_aggregates(df, df_agg_side, d_quant, l_measures):
    """Populates the overall aggregated metrics and total time."""
    for med_stage in df_agg_side[gc.columns.PRE_OR_POST].unique():
        for affected_side in [gc.descriptives.MOST_AFFECTED_SIDE, gc.descriptives.LEAST_AFFECTED_SIDE]:
            # Compute overall time (seconds)
            overall_seconds = df.loc[(df[gc.columns.PRE_OR_POST] == med_stage) &
                                     (df[gc.columns.SIDE] == affected_side)].shape[0] / gc.parameters.DOWNSAMPLED_FREQUENCY
            d_quant[med_stage][affected_side]['seconds']['overall'] = overall_seconds

            # Populate aggregated measures
            for measure in l_measures:
                d_quant[med_stage][affected_side]['values'][measure]['overall'] = \
                    df_agg_side.loc[(df_agg_side[gc.columns.PRE_OR_POST] == med_stage) &
                                    (df_agg_side[gc.columns.SIDE] == affected_side), measure].values[0]
                

def populate_segment_aggregates(df, df_agg_segments, d_quant, l_measures):
    """Populates the segment-specific aggregated metrics."""
    for med_stage in df_agg_segments[gc.columns.PRE_OR_POST].unique():
        df_med_stage = df_agg_segments.loc[df_agg_segments[gc.columns.PRE_OR_POST] == med_stage]
        for affected_side in df_med_stage[gc.columns.SIDE].unique():
            df_aff = df_med_stage.loc[df_med_stage[gc.columns.SIDE] == affected_side]
            for segment_category in df_aff[gc.columns.TRUE_GAIT_SEGMENT_CAT].unique():
                segment_seconds = df.loc[
                    (df[gc.columns.PRE_OR_POST] == med_stage) &
                    (df[gc.columns.SIDE] == affected_side) &
                    (df[gc.columns.TRUE_GAIT_SEGMENT_CAT] == segment_category)
                ].shape[0] / gc.parameters.DOWNSAMPLED_FREQUENCY
                d_quant[med_stage][affected_side]['seconds'][segment_category] = segment_seconds
                
                # Populate each measure for the segment
                for measure in l_measures:
                    d_quant[med_stage][affected_side]['values'][measure][segment_category] = \
                        df_aff.loc[df_aff[gc.columns.TRUE_GAIT_SEGMENT_CAT] == segment_category, measure].values[0]

    # Compute non-gait time
    compute_non_gait_time(d_quant)


def compute_non_gait_time(d_quant):
    """Calculates and adds non-gait time to the dictionary."""
    for med_stage in d_quant.keys():
        for affected_side in d_quant[med_stage].keys():
            gait_seconds = np.sum([d_quant[med_stage][affected_side]['seconds'][seg] 
                                   for seg in mp.segment_map.keys() if seg in d_quant[med_stage][affected_side]['seconds']])
            overall_seconds = d_quant[med_stage][affected_side]['seconds']['overall']
            d_quant[med_stage][affected_side]['seconds']['non_gait'] = overall_seconds - gait_seconds


def compute_aggregations(df):

    l_metrics = ['range_of_motion', 'peak_velocity']
    l_aggregates = ['median']
    l_quantiles = [0.95]

    l_measures = [
        'range_of_motion_median', 'range_of_motion_quantile_95',
        'peak_velocity_median', 'peak_velocity_quantile_95'
    ]

    l_groupby = [gc.columns.PRE_OR_POST]
    l_groupby_side = l_groupby + [gc.columns.SIDE]
    l_groupby_segments = l_groupby_side + [gc.columns.TRUE_GAIT_SEGMENT_CAT]

    # Extract features aggregated by side and segment
    df_agg_side = extract_features(df, l_groupby_side, l_metrics, l_aggregates, l_quantiles)
    df_agg_segments = extract_features(df, l_groupby_segments, l_metrics, l_aggregates, l_quantiles)

    # Initialize the result dictionary
    d_quant = initialize_d_quant(df_agg_side, l_measures)

    # Populate overall metrics by side
    populate_overall_aggregates(df, df_agg_side, d_quant, l_measures)

    # Populate segment-specific metrics
    populate_segment_aggregates(df, df_agg_segments, d_quant, l_measures)

    return d_quant


def bootstrap_samples(data, stat, num_samples=5000):
    """
    Perform bootstrapping on the given data to compute point estimates.
    
    Args:
    - data (array): Array of values to bootstrap.
    - stat (str): Statistic to compute ('median' or '95').
    - num_samples (int): Number of bootstrapped samples (default: 5000).
    
    Returns:
    - bootstrap_stat (array): Bootstrapped statistics based on the provided stat.
    """
    bootstraps = np.random.choice(data, size=(num_samples, len(data)), replace=True)
    if stat == 'median':
        return np.median(bootstraps, axis=1)
    elif stat == '95':
        return np.percentile(bootstraps, 95, axis=1)
    else:
        raise ValueError("Unsupported stat, choose either 'median' or '95'.")