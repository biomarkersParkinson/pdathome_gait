import numpy as np
import os
import pandas as pd

from pdathome.constants import global_constants as gc

from paradigma.segmenting import tabulate_windows

def tabulate_windows_pdathome(subject, df, colnames, window_length_s, window_step_length_s, sampling_frequency):
    windowed_data = []
    
    if subject in gc.participant_ids.PD_IDS:
        df_grouped = df.groupby(gc.columns.PRE_OR_POST, sort=False)
        order = ['pre', 'post']

        for label in order:
            if label in df_grouped.groups:  # Ensure the label exists in the groups
                group = df_grouped.get_group(label)
                windows = tabulate_windows(
                    df=group,
                    columns=colnames,
                    window_length_s=window_length_s,
                    window_step_length_s=window_step_length_s,
                    fs=sampling_frequency
                )
                if len(windows) > 0:  # Skip if no windows are created
                    windowed_data.append(windows)

    else:
        windows = tabulate_windows(
            df=df,
            columns=colnames,
            window_length_s=window_length_s,
            window_step_length_s=window_step_length_s,
            fs=sampling_frequency
        )
        if len(windows) > 0:  # Skip if no windows are created
            windowed_data.append(windows)

    if len(windowed_data) > 0:
        windowed_data = np.concatenate(windowed_data, axis=0)
    else:
        raise ValueError("No windows were created from the given data.")
    
    return windowed_data

def merge_timestamps_and_predictions(df_ts, df_pred, time_colname, pred_proba_colname, window_length_s, fs):

    # 1. Expand each window into individual timestamps
    expanded_data = []
    for _, row in df_pred.iterrows():
        start_time = row[time_colname]
        proba = row[pred_proba_colname]
        timestamps = np.arange(start_time, start_time + window_length_s, 1/fs)
        expanded_data.extend(zip(timestamps, [proba] * len(timestamps)))

    expanded_df = pd.DataFrame(expanded_data, columns=[time_colname, pred_proba_colname])

    # Step 2: Round timestamps to avoid floating-point inaccuracies
    expanded_df[time_colname] = expanded_df[time_colname].round(2)
    df_ts[time_colname] = df_ts[time_colname].round(2)

    # Step 3: Aggregate by unique timestamps and calculate the mean probability
    expanded_df = expanded_df.groupby(time_colname, as_index=False)[pred_proba_colname].mean()

    df_ts = pd.merge(left=df_ts, right=expanded_df, how='right', on=time_colname)

    return df_ts


def save_to_pickle(df, path, filename):
    """
    Saves a DataFrame to a pickle file, creating directories if they don't exist.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame to save.
    path (str): The directory path where the file will be saved.
    filename (str): The name of the pickle file.
    """
    # Ensure the directory exists, create if it doesn't
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Save the DataFrame to the specified pickle file
    file_path = os.path.join(path, filename)
    df.to_pickle(file_path)


def key_value_list_to_dict(key_value_list):
    d = {}
    for item in key_value_list:
        key, value = item.split(':')
        d[key] = float(value)

    return d