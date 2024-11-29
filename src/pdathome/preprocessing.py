import datetime
import json
import numpy as np
import os
import pandas as pd

from collections import Counter
from scipy.interpolate import CubicSpline

from paradigma.gait.feature_extraction import extract_temporal_domain_features, extract_spectral_domain_features, \
    pca_transform_gyroscope, compute_angle, remove_moving_average_angle, extract_angle_features
from paradigma.gait.gait_analysis_config import GaitFeatureExtractionConfig, ArmActivityFeatureExtractionConfig
from paradigma.imu_preprocessing import butterworth_filter
from paradigma.preprocessing_config import IMUPreprocessingConfig
from paradigma.segmenting import tabulate_windows, create_segments, discard_segments, categorize_segments

from pdathome.constants import global_constants as gc, mappings as mp
from pdathome.load import load_stage_start_end, load_sensor_data, load_video_annotations
from pdathome.utils import save_to_pickle

def compute_mode(data):
    """Computes the mode for 1D data using np.unique."""
    values, counts = np.unique(data, return_counts=True)
    max_count_index = np.argmax(counts)
    return values[max_count_index], counts[max_count_index]

def is_majority(data, target="Walking"):
    """Checks if 'target' occurs more than half the time in 1D data."""
    values, counts = np.unique(data, return_counts=True)
    target_count = counts[values == target].sum() if target in values else 0
    return target_count > (len(data) / 2)


def prepare_data(subject):
    print(f"Time {datetime.datetime.now()} - {subject} - Starting preparing data ...")
    with open(os.path.join(gc.paths.PATH_CLINICAL_DATA, 'distribution_participants.json'), 'r') as f:
        d_participant_distribution = json.load(f)

    l_time_cols = [gc.columns.TIME]
    l_sensor_cols = gc.columns.L_ACCELEROMETER + gc.columns.L_GYROSCOPE
    l_other_cols = [gc.columns.FREE_LIVING_LABEL]

    if subject in gc.participant_ids.L_PD_IDS:
        file_sensor_data = 'phys_cur_PD_merged.mat'
        path_annotations = os.path.join(gc.paths.PATH_ANNOTATIONS, 'pd')
        l_other_cols += [gc.columns.ARM_LABEL, gc.columns.PRE_OR_POST]
    else:
        file_sensor_data = 'phys_cur_HC_merged.mat'
        path_annotations = os.path.join(gc.paths.PATH_ANNOTATIONS, 'controls')

    if subject in gc.participant_ids.L_TREMOR_IDS:
        l_other_cols.append(gc.columns.TREMOR_LABEL)

    l_cols_to_export = l_time_cols + l_sensor_cols + l_other_cols

    for side in [gc.descriptives.MOST_AFFECTED_SIDE, gc.descriptives.LEAST_AFFECTED_SIDE]:        

        ## loading_annotations
        wrist_pos = determine_wrist_pos(subject, side, d_participant_distribution)
        df_sensors, peakstart, peakend = load_sensor_data(gc.paths.PATH_SENSOR_DATA, file_sensor_data, 'phys', subject, wrist_pos)

        if subject in gc.participant_ids.L_PD_IDS:
            df_annotations_part_1, df_annotations_part_2 = load_video_annotations(path_annotations, subject)
            df_annotations = sync_video_annotations(df_annotations_part_1, df_annotations_part_2, peakstart, peakend)
        else:
            df_annotations = load_video_annotations(path_annotations, subject)

        df_annotations = preprocess_video_annotations(df=df_annotations, code_label_map=mp.tiers_labels_map, d_tier_rename=mp.tiers_rename)
        
        ## merging
        df_sensors[gc.columns.SIDE] = side

        df_sensors = preprocess_sensor_data(df_sensors)

        df_sensors = attach_label_to_signal(df_sensors, df_annotations, 'label', 'tier', 'start_s', 'end_s')

        if subject in gc.participant_ids.L_PD_IDS:
            if wrist_pos == gc.descriptives.RIGHT_WRIST:
                df_sensors[gc.columns.ARM_LABEL] = df_sensors['right_arm_label']
            else:
                df_sensors[gc.columns.ARM_LABEL] = df_sensors['left_arm_label']

            df_sensors[gc.columns.ARM_LABEL] = df_sensors[gc.columns.ARM_LABEL].fillna('non_gait')

        df_sensors = determine_med_stage(df=df_sensors, subject=subject,
                                         watch_side=side,
                                         path_annotations=path_annotations)
        
        df_sensors = rotate_axes(df=df_sensors, subject=subject, wrist=wrist_pos)

        df_sensors = df_sensors.loc[df_sensors[gc.columns.FREE_LIVING_LABEL].notna()]
        df_sensors = df_sensors.loc[df_sensors[gc.columns.FREE_LIVING_LABEL]!='Unknown']
        df_sensors = df_sensors.loc[df_sensors[gc.columns.PRE_OR_POST].notna()]
        df_sensors = df_sensors.reset_index(drop=True)

        l_drop_cols = ['clinical_tests_label']
        if subject in gc.participant_ids.L_PD_IDS:
            l_drop_cols += ['med_and_motor_status_label', 'left_arm_label', 'right_arm_label']

        df_sensors = df_sensors.drop(columns=l_drop_cols)

        # temporarily store as pickle until tsdf issue is resolved
        save_to_pickle(
            df=df_sensors[l_cols_to_export],
            path=gc.paths.PATH_PREPARED_DATA,
            filename=f'{subject}_{side}.pkl'
        )
    print(f"Time {datetime.datetime.now()} - {subject} - Finished preparing data.")


def preprocess_gait_detection(subject):
    print(f"Time {datetime.datetime.now()} - {subject} - Starting preprocessing gait detection ...")
    for side in [gc.descriptives.MOST_AFFECTED_SIDE, gc.descriptives.LEAST_AFFECTED_SIDE]:
        df = pd.read_pickle(os.path.join(gc.paths.PATH_PREPARED_DATA, f'{subject}_{side}.pkl'))

        imu_config = IMUPreprocessingConfig()
        gait_config = GaitFeatureExtractionConfig()
        imu_config.acceleration_units = 'g'

        # Extract relevant gc.columns for accelerometer data
        accel_cols = imu_config.l_accelerometer_cols

        # Change to correct units [g]
        df[accel_cols] = df[accel_cols] / 9.81 if imu_config.acceleration_units == 'm/s^2' else df[accel_cols]

        # Extract accelerometer data
        accel_data = df[imu_config.l_accelerometer_cols].values

        filter_configs = {
            "hp": {"result_columns": imu_config.l_accelerometer_cols, "replace_original": True},
            "lp": {"result_columns": [f'{col}_grav' for col in imu_config.l_accelerometer_cols], "replace_original": False},
        }

        # Apply filters in a loop
        for passband, filter_config in filter_configs.items():
            filtered_data = butterworth_filter(
                data=accel_data,
                order=imu_config.filter_order,
                cutoff_frequency=imu_config.lower_cutoff_frequency,
                passband=passband,
                sampling_frequency=imu_config.sampling_frequency,
            )

            # Replace or add new columns based on configuration
            df[filter_config["result_columns"]] = filtered_data

        windowed_data = []

        l_windowed_cols = [
            gc.columns.TIME, gc.columns.FREE_LIVING_LABEL
            ] + gait_config.l_accelerometer_cols + gait_config.l_gravity_cols
        
        if subject in gc.participant_ids.L_PD_IDS:
            l_windowed_cols += [gc.columns.ARM_LABEL]

            df_grouped = df.groupby(gc.columns.PRE_OR_POST, sort=False)
            order = ['pre', 'post']

            for label in order:
                if label in df_grouped.groups:  # Ensure the label exists in the groups
                    group = df_grouped.get_group(label)
                    windows = tabulate_windows(
                        config=gait_config,
                        df=group,
                        columns=l_windowed_cols
                    )
                    if len(windows) > 0:  # Skip if no windows are created
                        windowed_data.append(windows)

        else:
            windows = tabulate_windows(
                config=gait_config,
                df=df,
                columns=l_windowed_cols
            )
            if len(windows) > 0:  # Skip if no windows are created
                windowed_data.append(windows)

        if len(windowed_data) > 0:
            windowed_data = np.concatenate(windowed_data, axis=0)
        else:
            raise ValueError("No windows were created from the given data.")

        df_features = pd.DataFrame()

        df_features[gc.columns.TIME] = sorted(windowed_data[:, 0, l_windowed_cols.index(gc.columns.TIME)])

        if subject in gc.participant_ids.L_PD_IDS:
            df_features = pd.merge(left=df_features, right=df[[gc.columns.TIME, gc.columns.PRE_OR_POST]], how='left', on=gc.columns.TIME) 

        # Calulate the mode of the labels
        windowed_labels = windowed_data[:, :, l_windowed_cols.index(gc.columns.FREE_LIVING_LABEL)]
        modes_and_counts = np.apply_along_axis(lambda x: compute_mode(x), axis=1, arr=windowed_labels)
        modes, counts = zip(*modes_and_counts)

        df_features[gc.columns.ACTIVITY_LABEL_MAJORITY_VOTING] = modes
        df_features[gc.columns.GAIT_MAJORITY_VOTING] = [is_majority(window) for window in windowed_labels]

        if subject in gc.participant_ids.L_PD_IDS:
            windowed_labels = windowed_data[:, :, l_windowed_cols.index(gc.columns.ARM_LABEL)]
            modes_and_counts = np.apply_along_axis(lambda x: compute_mode(x), axis=1, arr=windowed_labels)
            modes, counts = zip(*modes_and_counts)

            df_features[gc.columns.ARM_LABEL_MAJORITY_VOTING] = modes
            df_features[gc.columns.NO_OTHER_ARM_ACTIVITY_MAJORITY_VOTING] = [is_majority(window, target="Gait without other behaviours or other positions") for window in windowed_labels]

        # compute statistics of the temporal domain signals
        accel_indices = [l_windowed_cols.index(x) for x in gait_config.l_accelerometer_cols]
        grav_indices = [l_windowed_cols.index(x) for x in gait_config.l_gravity_cols]

        accel_windowed = np.asarray(windowed_data[:, :, np.min(accel_indices):np.max(accel_indices) + 1], dtype=float)
        grav_windowed = np.asarray(windowed_data[:, :, np.min(grav_indices):np.max(grav_indices) + 1], dtype=float)

        df_temporal_features = extract_temporal_domain_features(
            config=gait_config,
            windowed_acc=accel_windowed,
            windowed_grav=grav_windowed,
            l_grav_stats=['mean', 'std']
        )

        df_features = pd.concat([df_features, df_temporal_features], axis=1)

        # transform the signals from the temporal domain to the spectral domain using the fast fourier transform
        # and extract spectral features
        df_spectral_features = extract_spectral_domain_features(
            config=gait_config,
            sensor=gait_config.sensor,
            windowed_data=accel_windowed,
        )

        df_features = pd.concat([df_features, df_spectral_features], axis=1)
        
        file_path = os.path.join(gc.paths.PATH_GAIT_FEATURES, f'{subject}_{side}.pkl')
        df_features.to_pickle(file_path)

    print(f"Time {datetime.datetime.now()} - {subject} - Finished preprocessing gait detection.")


def preprocess_filtering_gait(subject):
    print(f"Time {datetime.datetime.now()} - {subject} - Starting preprocessing filtering gait ...")
    for side in [gc.descriptives.MOST_AFFECTED_SIDE, gc.descriptives.LEAST_AFFECTED_SIDE]:

        # load timestamps
        df_ts = pd.read_pickle(os.path.join(gc.paths.PATH_PREPARED_DATA, f'{subject}_{side}.pkl'))
        df_ts['time'] = df_ts['time'].round(2)

        # load gait features
        df_features = pd.read_pickle(os.path.join(gc.paths.PATH_GAIT_FEATURES, f'{subject}_{side}.pkl'))

        # Load gait predictions
        df_pred = pd.read_pickle(os.path.join(gc.paths.PATH_GAIT_PREDICTIONS, gc.classifiers.GAIT_DETECTION_CLASSIFIER_SELECTED, f'{subject}_{side}.pkl'))

        # Load classification threshold
        with open(os.path.join(gc.paths.PATH_THRESHOLDS, 'gait', f'{gc.classifiers.GAIT_DETECTION_CLASSIFIER_SELECTED}.txt'), 'r') as f:
            threshold = float(f.read())

        # Determine gait prediction per timestamp
        l_cols_features = ['time']
        df_predictions = pd.concat([df_features[l_cols_features], df_pred], axis=1)
        
        imu_config = IMUPreprocessingConfig()
        gait_config = GaitFeatureExtractionConfig()
        arm_activity_config = ArmActivityFeatureExtractionConfig()

        # Step 1: Expand each window into individual timestamps
        expanded_data = []
        for _, row in df_predictions.iterrows():
            start_time = row['time']
            proba = row['pred_gait_proba']
            timestamps = np.arange(start_time, start_time + gait_config.window_length_s, 1/gc.parameters.DOWNSAMPLED_FREQUENCY)
            expanded_data.extend(zip(timestamps, [proba] * len(timestamps)))

        # Create a new DataFrame with expanded timestamps
        expanded_df = pd.DataFrame(expanded_data, columns=['time', 'pred_gait_proba'])

        # Step 2: Round timestamps to avoid floating-point inaccuracies
        expanded_df['time'] = expanded_df['time'].round(2)

        # Step 3: Aggregate by unique timestamps and calculate the mean probability
        expanded_df = expanded_df.groupby('time', as_index=False)['pred_gait_proba'].mean()

        df_ts = pd.merge(left=df_ts, right=expanded_df, how='left', on='time')

        imu_config.acceleration_units = 'g'
        arm_activity_config.list_value_cols += [gc.columns.TIME, gc.columns.FREE_LIVING_LABEL]

        # Extract relevant gc.columns for accelerometer data
        accel_cols = imu_config.l_accelerometer_cols

        # Change to correct units [g]
        df_ts[accel_cols] = df_ts[accel_cols] / 9.81 if imu_config.acceleration_units == 'm/s^2' else df_ts[accel_cols]

        # Extract accelerometer data
        accel_data = df_ts[imu_config.l_accelerometer_cols].values

        filter_configs = {
            "hp": {"result_columns": imu_config.l_accelerometer_cols, "replace_original": True},
            "lp": {"result_columns": [f'{col}_grav' for col in imu_config.l_accelerometer_cols], "replace_original": False},
        }

        # Apply filters in a loop
        for passband, filter_config in filter_configs.items():
            filtered_data = butterworth_filter(
                data=accel_data,
                order=imu_config.filter_order,
                cutoff_frequency=imu_config.lower_cutoff_frequency,
                passband=passband,
                sampling_frequency=imu_config.sampling_frequency,
            )

            # Replace or add new columns based on configuration
            df_ts[filter_config["result_columns"]] = filtered_data

        # Process free living label and remove nans
        df_ts = df_ts.dropna(subset=gc.columns.L_GYROSCOPE)
            
        # Apply threshold and filter data
        df_ts[gc.columns.PRED_GAIT] = (df_ts[gc.columns.PRED_GAIT_PROBA] >= threshold).astype(int)

        # Perform principal component analysis on the gyroscope signals to obtain the angular velocity in the
        # direction of the swing of the arm 
        df_ts[gc.columns.VELOCITY] = pca_transform_gyroscope(
            config=arm_activity_config,
            df=df_ts,
        )

        # Integrate the angular velocity to obtain an estimation of the angle
        df_ts[gc.columns.ANGLE] = compute_angle(
            config=arm_activity_config,
            df=df_ts,
        )

        # Remove the moving average from the angle to account for possible drift caused by the integration
        # of noise in the angular velocity
        df_ts[gc.columns.ANGLE] = remove_moving_average_angle(
            config=arm_activity_config,
            df=df_ts,
        )
        
        # Filter unobserved data
        if subject in gc.participant_ids.L_PD_IDS:
            df_ts = df_ts[df_ts[gc.columns.ARM_LABEL] != 'Cannot assess']
        
        # Use only predicted gait for the subsequent steps
        df_ts = df_ts[df_ts[gc.columns.PRED_GAIT] == 1].reset_index(drop=True)

        # Group consecutive timestamps into segments with new segments starting after a pre-specified gap
        df_ts[gc.columns.SEGMENT_NR] = create_segments(
            config=arm_activity_config,
            df=df_ts
        )

        # Remove any segments that do not adhere to predetermined criteria
        df_ts = discard_segments(
            config=arm_activity_config,
            df=df_ts
        )

        # Create windows of fixed length and step size from the time series
        windowed_data = []

        l_windowed_cols = [
            gc.columns.TIME, gc.columns.FREE_LIVING_LABEL, gc.columns.ANGLE, gc.columns.VELOCITY
            ] + arm_activity_config.l_accelerometer_cols + arm_activity_config.l_gravity_cols + arm_activity_config.l_gyroscope_cols
        
        if subject in gc.participant_ids.L_PD_IDS:
            l_windowed_cols += [gc.columns.ARM_LABEL]

        df_grouped = df_ts.groupby(gc.columns.SEGMENT_NR, sort=False)

        for _, group in df_grouped:
            windows = tabulate_windows(
                config=arm_activity_config,
                df=group,
                columns=l_windowed_cols
            )
            if len(windows) > 0:  # Skip if no windows are created
                windowed_data.append(windows)

        if len(windowed_data) > 0:
            windowed_data = np.concatenate(windowed_data, axis=0)
        else:
            raise ValueError("No windows were created from the given data.")

        df_features = pd.DataFrame()

        df_features[gc.columns.TIME] = sorted(windowed_data[:, 0, l_windowed_cols.index(gc.columns.TIME)])

        if subject in gc.participant_ids.L_PD_IDS:
            df_features = pd.merge(left=df_features, right=df_ts[[gc.columns.TIME, gc.columns.PRE_OR_POST]], how='left', on=gc.columns.TIME) 

        # Calulate the mode of the labels
        windowed_labels = windowed_data[:, :, l_windowed_cols.index(gc.columns.FREE_LIVING_LABEL)]
        modes_and_counts = np.apply_along_axis(lambda x: compute_mode(x), axis=1, arr=windowed_labels)
        modes, counts = zip(*modes_and_counts)

        df_features[gc.columns.ACTIVITY_LABEL_MAJORITY_VOTING] = modes
        df_features[gc.columns.GAIT_MAJORITY_VOTING] = [is_majority(window) for window in windowed_labels]

        if subject in gc.participant_ids.L_PD_IDS:
            windowed_labels = windowed_data[:, :, l_windowed_cols.index(gc.columns.ARM_LABEL)]
            modes_and_counts = np.apply_along_axis(lambda x: compute_mode(x), axis=1, arr=windowed_labels)
            modes, counts = zip(*modes_and_counts)

            df_features[gc.columns.ARM_LABEL_MAJORITY_VOTING] = modes
            df_features[gc.columns.NO_OTHER_ARM_ACTIVITY_MAJORITY_VOTING] = [is_majority(window, target="Gait without other behaviours or other positions") for window in windowed_labels]

        # compute statistics of the temporal domain signals
        accel_indices = [l_windowed_cols.index(x) for x in gait_config.l_accelerometer_cols]
        grav_indices = [l_windowed_cols.index(x) for x in gait_config.l_gravity_cols]
        gyro_indices = [l_windowed_cols.index(x) for x in gait_config.l_gyroscope_cols]
        idx_angle = l_windowed_cols.index(gc.columns.ANGLE)
        idx_velocity = l_windowed_cols.index(gc.columns.VELOCITY)

        accel_windowed = np.asarray(windowed_data[:, :, np.min(accel_indices):np.max(accel_indices) + 1], dtype=float)
        grav_windowed = np.asarray(windowed_data[:, :, np.min(grav_indices):np.max(grav_indices) + 1], dtype=float)
        gyro_windowed = np.asarray(windowed_data[:, :, np.min(gyro_indices):np.max(gyro_indices) + 1], dtype=float)
        angle_windowed = np.asarray(windowed_data[:, :, idx_angle], dtype=float)
        velocity_windowed = np.asarray(windowed_data[:, :, idx_velocity], dtype=float)

        # angle features
        df_features_angle = extract_angle_features(arm_activity_config, angle_windowed, velocity_windowed)
        df_features = pd.concat([df_features, df_features_angle], axis=1)

        # compute statistics of the temporal domain accelerometer signals
        df_temporal_features = extract_temporal_domain_features(arm_activity_config, accel_windowed, grav_windowed, l_grav_stats=['mean', 'std'])
        df_features = pd.concat([df_features, df_temporal_features], axis=1)

        # transform the accelerometer and gyroscope signals from the temporal domain to the spectral domain
        # using the fast fourier transform and extract spectral features
        for sensor_name, windowed_sensor in zip(['accelerometer', 'gyroscope'], [accel_windowed, gyro_windowed]):
            df_spectral_features = extract_spectral_domain_features(arm_activity_config, sensor_name, windowed_sensor)
            df_features = pd.concat([df_features, df_spectral_features], axis=1)

        file_path = os.path.join(gc.paths.PATH_ARM_ACTIVITY_FEATURES, f'{subject}_{side}.pkl')
        df_features.to_pickle(file_path)

    print(f"Time {datetime.datetime.now()} - {subject} - Finished preprocessing filtering gait.")


def determine_wrist_pos(subject, watch_side, d_participant_distribution):
    if watch_side == gc.descriptives.MOST_AFFECTED_SIDE and subject in d_participant_distribution['most_affected']['right']:
        return gc.descriptives.RIGHT_WRIST
    elif watch_side == gc.descriptives.MOST_AFFECTED_SIDE and subject in d_participant_distribution['most_affected']['left']:
        return gc.descriptives.LEFT_WRIST
    elif watch_side == gc.descriptives.LEAST_AFFECTED_SIDE and subject in d_participant_distribution['least_affected']['right']:
        return gc.descriptives.RIGHT_WRIST
    else:
        return gc.descriptives.LEFT_WRIST
    

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


def preprocess_video_annotations(df, code_label_map, d_tier_rename):  
    for tier in code_label_map:
        if tier == 'Arm':
            df.loc[df['tier']=='Left arm', 'label'] = df.loc[df['tier']=='Left arm', 'code'].map(code_label_map[tier])
            df.loc[df['tier']=='Right arm', 'label'] = df.loc[df['tier']=='Right arm', 'code'].map(code_label_map[tier])
        else:
            df.loc[df['tier']==tier, 'label'] = df.loc[df['tier']==tier, 'code'].map(code_label_map[tier])

    if 'start' in df.columns:
        for moment in ['start', 'end']:
            df[moment+'_s'] = df[moment].copy()

    df['tier'] = df['tier'].map(d_tier_rename)

    df = df.drop(columns=['start', 'end'])

    # remove nan tier, which is used for setting up and synchronizing the devices
    df = df.loc[~df['tier'].isna()] 


    return df


def preprocess_sensor_data(df_sensors):
    df_sensors[gc.columns.TIME+'_s'] = df_sensors[gc.columns.TIME].copy()
    df_sensors[gc.columns.TIME+'_dt'] = df_sensors[gc.columns.TIME+'_s'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    df_sensors = df_sensors.resample(str(1/gc.parameters.SAMPLING_FREQUENCY)+'s', on=gc.columns.TIME+'_dt').first().reset_index()
    df_sensors[gc.columns.TIME] = df_sensors[gc.columns.TIME+'_dt'].apply(lambda x: x.timestamp())
    df_sensors = df_sensors.drop(columns=[gc.columns.TIME+'_dt', gc.columns.TIME+'_s'])
    df_sensors[gc.columns.TIME] = df_sensors[gc.columns.TIME] - df_sensors[gc.columns.TIME].min()    

    for col in gc.columns.L_ACCELEROMETER:
        cs = CubicSpline(
            df_sensors.loc[
                df_sensors[col].isna()==False,
                gc.columns.TIME
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
                gc.columns.TIME
                ]
            )
        
    downsample_rate = gc.parameters.DOWNSAMPLED_FREQUENCY/gc.parameters.SAMPLING_FREQUENCY

    df_sensors = df_sensors[0::int(1/downsample_rate)]

    return df_sensors


def attach_label_to_signal(df_sensors, df_labels, label_column, tier_column, start_column, end_column, l_tiers=None):
    if l_tiers == None:
        l_tiers = df_labels[tier_column].unique()
    for tier in l_tiers:
        for _, row in df_labels.loc[df_labels[tier_column]==tier].iterrows():
            df_sensors.loc[(df_sensors[gc.columns.TIME] >= row[start_column]) & (df_sensors[gc.columns.TIME] < row[end_column]), '{}_label'.format(tier)] = row[label_column]

    df_sensors = df_sensors.reset_index(drop=True)
    return df_sensors


def determine_med_stage(df, subject, watch_side, path_annotations):
    prestart, preend, poststart, postend = load_stage_start_end(path_annotations, subject)

    if subject == 'hbv051' and watch_side == gc.descriptives.MOST_AFFECTED_SIDE:
        df = df.loc[df[gc.columns.TIME]>=5491.138] 

    df[gc.columns.PRE_OR_POST] = df.apply(lambda x: 'pre' if x[gc.columns.TIME] >= prestart and x[gc.columns.TIME] <= preend else 
                                        'post' if x[gc.columns.TIME] >= poststart and x[gc.columns.TIME] <= postend else
                                        np.nan, axis=1)
    
    return df


def rotate_axes(df, subject, wrist):
    for sensor in ['accelerometer', 'gyroscope']:
        df[f'{sensor}_x_new'] = df[f'{sensor}_y'].copy()
        df[f'{sensor}_y'] = df[f'{sensor}_x'].copy()
        df = df.drop(columns=[f'{sensor}_x'])
        df = df.rename(columns={f'{sensor}_x_new': f'{sensor}_x'})

    if wrist == gc.descriptives.LEFT_WRIST and subject in gc.participant_ids.L_L_NORMAL:
        df[gc.columns.ACCELEROMETER_Y] *= -1
    elif wrist == gc.descriptives.LEFT_WRIST and subject not in gc.participant_ids.L_L_NORMAL:
        df[gc.columns.ACCELEROMETER_X] *= -1
        df[gc.columns.GYROSCOPE_X] *= -1
        df[gc.columns.GYROSCOPE_Y] *= -1
    elif wrist == gc.descriptives.RIGHT_WRIST and subject in gc.participant_ids.L_R_NORMAL:
        df[gc.columns.ACCELEROMETER_X] *= -1
        df[gc.columns.ACCELEROMETER_Y] *= -1
        df[gc.columns.GYROSCOPE_Z] *= -1
        df[gc.columns.GYROSCOPE_Y] *= -1
    elif wrist == gc.descriptives.RIGHT_WRIST and subject not in gc.participant_ids.L_R_NORMAL:
        df[gc.columns.GYROSCOPE_X] *= -1
        df[gc.columns.GYROSCOPE_Z] *= -1

    return df


def arm_label_majority_voting(config, arm_label):
    non_nan_count = sum(~pd.isna(arm_label))
    if non_nan_count > config.window_length_s * gc.parameters.DOWNSAMPLED_FREQUENCY / 2:
        return Counter(arm_label).most_common(1)[0][0]
    return np.nan


def add_segment_category(config, df, activity_colname, segment_nr_colname,
                         segment_cat_colname, activity_value):
    # Create segments based on video-annotations of gait
    segments = create_segments(
        config=config,
        df=df
    )

    # Assign segment numbers to the raw data
    df.loc[df[activity_colname] == activity_value, segment_nr_colname] = segments

    # Non-gait raw data is assigned a segment number of -1
    df[segment_nr_colname] = df[segment_nr_colname].fillna(-1)

    # Map categories to segments of video-annotated gait
    segments_cat = categorize_segments(
        df=df.loc[(df[activity_colname] == activity_value) & (df[segment_nr_colname] != -1)],
        segment_nr_colname=segment_nr_colname,
        sampling_frequency=gc.parameters.DOWNSAMPLED_FREQUENCY
    )

    # Assign segment categories to the raw data
    df.loc[(df[activity_colname] == activity_value) & (df[segment_nr_colname] != -1), segment_cat_colname] = segments_cat

    # Non-gait raw data is assigned a segment category of -1
    df[segment_cat_colname] = df[segment_cat_colname].fillna(-1)

    # Map segment categories to segments of video-annotated gait
    df[segment_cat_colname] = df[segment_cat_colname].apply(
            lambda x: mp.segment_map[x]
        )
    
    return df