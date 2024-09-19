import datetime
import numpy as np
import os
import pandas as pd

from collections import Counter
from scipy.interpolate import CubicSpline

from paradigma.feature_extraction import extract_temporal_domain_features, extract_spectral_domain_features
from paradigma.gait_analysis_config import GaitFeatureExtractionConfig
from paradigma.imu_preprocessing import butterworth_filter
from paradigma.preprocessing_config import IMUPreprocessingConfig
from paradigma.windowing import tabulate_windows

from pdathome.constants import columns, descriptives, parameters, participant_ids, paths
from pdathome.load import load_stage_start_end


def preprocess_gait(subject, side, path_input=paths.PATH_DATAFRAMES, path_output=paths.PATH_GAIT_FEATURES):
    print(f"Time {datetime.datetime.now()} - {subject} {side} - Processing ...")
    df = pd.read_pickle(os.path.join(path_input, f'{subject}_{side}.pkl'))

    config = IMUPreprocessingConfig()
    config.acceleration_units = 'g'

    # Extract relevant columns for accelerometer data
    accel_cols = list(config.d_channels_accelerometer.keys())

    # Change to correct units [g]
    df[accel_cols] = df[accel_cols] / 9.81 if config.acceleration_units == 'm/s^2' else df[accel_cols]

    # Extract the accelerometer data as a 2D array
    accel_data = df[accel_cols].values

    # Define filtering passbands
    passbands = ['hp', 'lp'] 
    filtered_data = {}

    # Apply Butterworth filter for each passband and result type
    for result, passband in zip(['filt', 'grav'], passbands):
        filtered_data[result] = butterworth_filter(
            sensor_data=accel_data,
            order=config.filter_order,
            cutoff_frequency=config.lower_cutoff_frequency,
            passband=passband,
            sampling_frequency=parameters.DOWNSAMPLED_FREQUENCY
        )

    # Create DataFrames from filtered data
    filtered_dfs = {f'{result}_{col}': pd.Series(data[:, i]) for i, col in enumerate(accel_cols) for result, data in filtered_data.items()}

    # Combine filtered columns into DataFrame
    filtered_df = pd.DataFrame(filtered_dfs)

    # Drop original accelerometer columns and append filtered results
    df = df.drop(columns=accel_cols).join(filtered_df).rename(columns={col: col.replace('filt_', '') for col in filtered_df.columns})

    config = GaitFeatureExtractionConfig()

    config.l_data_point_level_cols += [columns.TIME, columns.FREE_LIVING_LABEL]
    l_ts_cols = [columns.TIME, columns.WINDOW_NR, columns.FREE_LIVING_LABEL]
    l_export_cols = [columns.TIME, columns.WINDOW_NR, columns.ACTIVITY_LABEL_MAJORITY_VOTING, columns.GAIT_MAJORITY_VOTING] + list(config.d_channels_values.keys())

    if subject in participant_ids.L_PD_IDS:
        config.l_data_point_level_cols += [columns.PRE_OR_POST, columns.ARM_LABEL]
        l_ts_cols += [columns.PRE_OR_POST, columns.ARM_LABEL]
        l_export_cols += [columns.PRE_OR_POST, columns.ARM_LABEL_MAJORITY_VOTING]
    if subject in participant_ids.L_TREMOR_IDS:
        config.l_data_point_level_cols += [columns.TREMOR_LABEL]
        l_ts_cols += [columns.TREMOR_LABEL]


    df_windowed = tabulate_windows(
            df=df,
            time_column_name=columns.TIME,
            data_point_level_cols=config.l_data_point_level_cols,
            window_length_s=config.window_length_s,
            window_step_size_s=config.window_step_size_s,
            sampling_frequency=parameters.DOWNSAMPLED_FREQUENCY
    )
    
    # store windows with timestamps for later use
    df_windowed[l_ts_cols].to_pickle(os.path.join(paths.PATH_GAIT_FEATURES, f'{subject}_{side}_ts.pkl'))

    # Determine most prevalent activity
    df_windowed[columns.ACTIVITY_LABEL_MAJORITY_VOTING] = df_windowed[columns.FREE_LIVING_LABEL].apply(lambda x: pd.Series(x).mode()[0])

    # Determine if the majority of the window is walking
    df_windowed[columns.GAIT_MAJORITY_VOTING] = df_windowed[columns.FREE_LIVING_LABEL].apply(lambda x: x.count('Walking') >= len(x)/2)

    if subject in participant_ids.L_PD_IDS:
        df_windowed[columns.PRE_OR_POST] = df_windowed[columns.PRE_OR_POST].str[0]
        df_windowed[columns.ARM_LABEL_MAJORITY_VOTING] = df_windowed[columns.ARM_LABEL].apply(lambda x: arm_label_majority_voting(config, x))

    df_windowed = df_windowed.drop(columns=[x for x in l_ts_cols if x not in [columns.WINDOW_NR, columns.PRE_OR_POST]])

    # compute statistics of the temporal domain signals
    df_windowed = extract_temporal_domain_features(
        config=config,
        df_windowed=df_windowed,
        l_gravity_stats=['mean', 'std']
    )

    # transform the signals from the temporal domain to the spectral domain using the fast fourier transform
    # and extract spectral features
    df_windowed = extract_spectral_domain_features(
        config=config,
        df_windowed=df_windowed,
        sensor=config.sensor,
        l_sensor_colnames=config.l_accelerometer_cols
    )

    df_windowed[l_export_cols].to_pickle(os.path.join(path_output, f'{subject}_{side}.pkl'))


def determine_wrist_pos(subject, watch_side, d_participant_distribution):
    if watch_side == descriptives.MOST_AFFECTED_SIDE and subject in d_participant_distribution[descriptives.MOST_AFFECTED_SIDE]['right']:
        return descriptives.descriptives.RIGHT_WRIST
    elif watch_side == descriptives.MOST_AFFECTED_SIDE and subject in d_participant_distribution[descriptives.MOST_AFFECTED_SIDE]['left']:
        return descriptives.descriptives.LEFT_WRIST
    elif watch_side == descriptives.LEAST_AFFECTED_SIDE and subject in d_participant_distribution[descriptives.LEAST_AFFECTED_SIDE]['right']:
        return descriptives.descriptives.RIGHT_WRIST
    else:
        return descriptives.descriptives.LEFT_WRIST
    

def sync_video_annotations(df_annotations_part_1, df_annotations_part_2, peakstart, peakend):
    syncstart = df_annotations_part_1.loc[(df_annotations_part_1['tier']=='General protocol structure') & (df_annotations_part_1['code']==1), 'start_s'].values[0]
    df_annotations_part_1['start_s'] = df_annotations_part_1['start_s'] - syncstart
    df_annotations_part_1['end_s'] = df_annotations_part_1['end_s'] - syncstart

    syncend = df_annotations_part_2.loc[(df_annotations_part_2['tier']=='General protocol structure') & (df_annotations_part_2['code']==9), 'start_s'].values[0]
    df_annotations_part_2['start_s'] = df_annotations_part_2['start_s'] - syncend
    df_annotations_part_2['end_s'] = df_annotations_part_2['end_s'] - syncend

    phys_end = peakend - peakstart
    df_annotations_part_2['start_s'] = df_annotations_part_2['start_s'] + phys_end
    df_annotations_part_2['end_s'] = df_annotations_part_2['end_s'] + phys_end

    df_annotations = pd.concat([df_annotations_part_1, df_annotations_part_2]).reset_index(drop=True)
    df_annotations = df_annotations.drop(columns=['duration_s'])
    df_annotations = df_annotations.rename(columns={'start_s': 'start', 'end_s': 'end'})

    return df_annotations


def preprocess_video_annotations(df, d_labels, d_tiers):  
    for tier in d_labels:
        if tier == 'Arm':
            df.loc[df['tier']=='Left arm', 'label'] = df.loc[df['tier']=='Left arm', 'code'].map(d_labels[tier])
            df.loc[df['tier']=='Right arm', 'label'] = df.loc[df['tier']=='Right arm', 'code'].map(d_labels[tier])
        else:
            df.loc[df['tier']==tier, 'label'] = df.loc[df['tier']==tier, 'code'].map(d_labels[tier])

    if 'start' in df.columns:
        for moment in ['start', 'end']:
            df[moment+'_s'] = df[moment].copy()

    df['tier'] = df['tier'].map(d_tiers)

    df = df.drop(columns=['start', 'end'])

    # remove nan tier, which is used for setting up and synchronizing the devices
    df = df.loc[~df['tier'].isna()] 

    return df


def preprocess_sensor_data(df_sensors):
    df_sensors[columns.TIME+'_s'] = df_sensors[columns.TIME].copy()
    df_sensors[columns.TIME+'_dt'] = df_sensors[columns.TIME+'_s'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    df_sensors = df_sensors.resample(str(1/parameters.SAMPLING_FREQUENCY)+'s', on=columns.TIME+'_dt').first().reset_index()
    df_sensors[columns.TIME] = df_sensors[columns.TIME+'_dt'].apply(lambda x: x.timestamp())
    df_sensors = df_sensors.drop(columns=[columns.TIME+'_dt', columns.TIME+'_s'])
    df_sensors[columns.TIME] = df_sensors[columns.TIME] - df_sensors[columns.TIME].min()    

    for col in columns.L_ACCEL_COLS:
        cs = CubicSpline(
            df_sensors.loc[
                df_sensors[col].isna()==False,
                columns.TIME
                ],
             df_sensors.loc[
                df_sensors[col].isna()==False,
                col
                ]
        )

        df_sensors.loc[
            df_sensors[col].isna()==True,
            col
            ] = cs(
            df_sensors.loc[
                df_sensors[col].isna()==True,
                columns.TIME
                ]
            )
        
    downsample_rate = parameters.DOWNSAMPLED_FREQUENCY/parameters.SAMPLING_FREQUENCY

    df_sensors = df_sensors[0::int(1/downsample_rate)]

    return df_sensors


def attach_label_to_signal(df_sensors, df_labels, label_column, tier_column, start_column, end_column, l_tiers=None):
    if l_tiers == None:
        l_tiers = df_labels[tier_column].unique()
    for tier in l_tiers:
        for _, row in df_labels.loc[df_labels[tier_column]==tier].iterrows():
            df_sensors.loc[(df_sensors[columns.TIME] >= row[start_column]) & (df_sensors[columns.TIME] < row[end_column]), '{}_label'.format(tier)] = row[label_column]

    df_sensors = df_sensors.reset_index(drop=True)
    return df_sensors


def determine_med_stage(df, subject, watch_side, path_annotations):
    prestart, preend, poststart, postend = load_stage_start_end(path_annotations, subject)

    if subject == 'hbv051' and watch_side == descriptives.MOST_AFFECTED_SIDE:
        df = df.loc[df[columns.TIME]>=5491.138] 

    df['pre_or_post'] = df.apply(lambda x: 'pre' if x[columns.TIME] >= prestart and x[columns.TIME] <= preend else 
                                        'post' if x[columns.TIME] >= poststart and x[columns.TIME] <= postend else
                                        np.nan, axis=1)
    
    return df


def rotate_axes(df, subject, wrist):
    for sensor in ['accelerometer', 'gyroscope']:
        df[f'{sensor}_x_new'] = df[f'{sensor}_y'].copy()
        df[f'{sensor}_y'] = df[f'{sensor}_x'].copy()
        df = df.drop(columns=[f'{sensor}_x'])
        df = df.rename(columns={f'{sensor}_x_new': f'{sensor}_x'})

    if wrist == descriptives.LEFT_WRIST and subject in participant_ids.L_L_NORMAL:
        df[columns.ACCELEROMETER_Y] *= -1
    elif wrist == descriptives.LEFT_WRIST and subject not in participant_ids.L_L_NORMAL:
        df[columns.ACCELEROMETER_X] *= -1
        df[columns.GYROSCOPE_X] *= -1
        df[columns.GYROSCOPE_Y] *= -1
    elif wrist == descriptives.RIGHT_WRIST and subject in participant_ids.L_R_NORMAL:
        df[columns.ACCELEROMETER_X] *= -1
        df[columns.ACCELEROMETER_Y] *= -1
        df[columns.GYROSCOPE_Z] *= -1
        df[columns.GYROSCOPE_Y] *= -1
    elif wrist == descriptives.RIGHT_WRIST and subject not in participant_ids.L_R_NORMAL:
        df[columns.GYROSCOPE_X] *= -1
        df[columns.GYROSCOPE_Z] *= -1

    return df


def arm_label_majority_voting(config, arm_label):
    non_nan_count = sum(~pd.isna(arm_label))
    if non_nan_count > config.window_length_s * parameters.DOWNSAMPLED_FREQUENCY / 2:
        return Counter(arm_label).most_common(1)[0][0]
    return np.nan