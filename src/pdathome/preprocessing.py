import datetime
import numpy as np
import os
import pandas as pd

from collections import Counter
from scipy.interpolate import interp1d

from paradigma.pipelines.gait_pipeline import extract_temporal_domain_features, extract_spectral_domain_features
from paradigma.pipelines.gait_pipeline import merge_predictions_with_timestamps
from paradigma.preprocessing import butterworth_filter
from paradigma.config import IMUConfig, GaitFeatureExtractionConfig, ArmActivityFeatureExtractionConfig
from paradigma.segmenting import create_segments, discard_segments, categorize_segments, WindowedDataExtractor

from pdathome.constants import global_constants as gc, mappings as mp
from pdathome.load import load_stage_start_end, load_sensor_data, load_video_annotations
from pdathome.utils import tabulate_windows_pdathome

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


def prepare_data(subject, wrist):
    """Process a single side (most/least affected) for a given subject."""
    time_cols = [gc.columns.TIME]
    sensor_cols = gc.columns.ACCELEROMETER_COLS + gc.columns.GYROSCOPE_COLS
    other_cols = [gc.columns.FREE_LIVING_LABEL]

    path_annotations = os.path.join(gc.paths.PATH_ANNOTATIONS, 'pd' if subject in gc.participant_ids.PD_IDS else 'controls')
    file_sensor_data = 'phys_cur_PD_merged.mat' if subject in gc.participant_ids.PD_IDS else 'phys_cur_HC_merged.mat'

    if subject in gc.participant_ids.PD_IDS:
        other_cols += [gc.columns.ARM_LABEL, gc.columns.PRE_OR_POST]

    if subject in gc.participant_ids.TREMOR_IDS:
        other_cols.append(gc.columns.TREMOR_LABEL)

    cols_to_export = time_cols + sensor_cols + other_cols

    ## loading_annotations
    df_sensors, peakstart, peakend = load_sensor_data(gc.paths.PATH_SENSOR_DATA, file_sensor_data, 'phys', subject, wrist)

    if subject in gc.participant_ids.PD_IDS:
        df_annotations_part_1, df_annotations_part_2 = load_video_annotations(path_annotations, subject)
        df_annotations = sync_video_annotations(df_annotations_part_1, df_annotations_part_2, peakstart, peakend)
    else:
        df_annotations = load_video_annotations(path_annotations, subject)

    df_annotations = preprocess_video_annotations(df=df_annotations, code_label_map=mp.tiers_labels_map, d_tier_rename=mp.tiers_rename)
    
    ## merging
    df_sensors = preprocess_sensor_data(df_sensors)

    df_sensors = attach_label_to_signal(df_sensors, df_annotations, 'label', 'tier', 'start', 'end')

    if subject in gc.participant_ids.PD_IDS:
        arm_label = 'right_arm_label' if wrist == gc.descriptives.RIGHT_WRIST else 'left_arm_label'
        df_sensors[gc.columns.ARM_LABEL] = df_sensors[arm_label].fillna('non_gait')

    df_sensors = determine_med_stage(df=df_sensors, subject=subject,
                                     path_annotations=path_annotations)
    
    df_sensors = rotate_axes(df=df_sensors, subject=subject, wrist=wrist)

    # Filter and clean data
    df_sensors = df_sensors.loc[
        df_sensors[gc.columns.FREE_LIVING_LABEL].notna() &
        (df_sensors[gc.columns.FREE_LIVING_LABEL] != 'Unknown') &
        df_sensors[gc.columns.PRE_OR_POST].notna()
    ].reset_index(drop=True)

    drop_cols = ['clinical_tests_label']
    if subject in gc.participant_ids.PD_IDS:
        drop_cols += ['med_and_motor_status_label', 'left_arm_label', 'right_arm_label']

    df_sensors = df_sensors.drop(columns=drop_cols)

    return df_sensors[cols_to_export]

# def preprocess_step(subject, affected_side, step):

#     if not step in ['gait', 'arm_activity']:
#         raise ValueError(f"Invalid step: {step}. Must be one of ['gait', 'arm_activity']")
    
#     df = pd.read_parquet(os.path.join(gc.paths.PATH_PREPARED_DATA, f'{subject}_{affected_side}.parquet'))

#     imu_config = IMUConfig()
#     imu_config.acceleration_units = 'g'

#     if step == 'gait':
#         feature_config = GaitFeatureExtractionConfig()
#     else:
#         gait_feature_config = GaitFeatureExtractionConfig()
#         feature_config = ArmActivityFeatureExtractionConfig()

#         # Merge timestamps into predictions
#         df_pred = pd.read_parquet(os.path.join(gc.paths.PATH_GAIT_PREDICTIONS, gc.classifiers.GAIT_DETECTION_CLASSIFIER_SELECTED, f'{subject}_{affected_side}.parquet'))
#         df = merge_timestamps_and_predictions(
#             df_ts=df,
#             df_pred=df_pred,
#             time_colname=gc.columns.TIME,
#             pred_proba_colname=gc.columns.PRED_GAIT_PROBA,
#             window_length_s=gait_feature_config.window_length_s,
#             fs=feature_config.sampling_frequency,
#         )

#     # Extract relevant gc.columns for accelerometer data
#     accel_cols = imu_config.accelerometer_cols

#     # Change to correct units [g]
#     df[accel_cols] = df[accel_cols] / 9.81 if imu_config.acceleration_units == 'm/s^2' else df[accel_cols]

#     # Extract accelerometer data
#     accel_data = df[imu_config.accelerometer_cols].values

#     filter_configs = {
#         "hp": {"result_columns": imu_config.accelerometer_cols, "replace_original": True},
#         "lp": {"result_columns": [f'{col}_grav' for col in imu_config.accelerometer_cols], "replace_original": False},
#     }

#     # Apply filters in a loop
#     for passband, filter_config in filter_configs.items():
#         filtered_data = butterworth_filter(
#             data=accel_data,
#             order=imu_config.filter_order,
#             cutoff_frequency=imu_config.lower_cutoff_frequency,
#             passband=passband,
#             sampling_frequency=imu_config.sampling_frequency,
#         )

#         # Replace or add new columns based on configuration
#         df[filter_config["result_columns"]] = filtered_data

#     # Process free living label and remove nans
#     df = df.dropna(subset=gc.columns.GYROSCOPE_COLS)
        
#     if step == 'arm_activity':
#         # Load classification threshold
#         with open(os.path.join(gc.paths.PATH_THRESHOLDS, 'gait', f'{gc.classifiers.GAIT_DETECTION_CLASSIFIER_SELECTED}.txt'), 'r') as f:
#             threshold = float(f.read())

#         # Apply threshold and filter data
#         df[gc.columns.PRED_GAIT] = (df[gc.columns.PRED_GAIT_PROBA] >= threshold).astype(int)

#         # Use only predicted gait for the subsequent steps
#         df = df[df[gc.columns.PRED_GAIT] == 1].reset_index(drop=True)
    
#         # Filter unobserved data
#         if subject in gc.participant_ids.PD_IDS:
#             df = df[df[gc.columns.ARM_LABEL] != 'Cannot assess']
    
#     # Group consecutive timestamps into segments with new segments starting after a pre-specified gap
#     df[gc.columns.SEGMENT_NR] = create_segments(
#         time_array=df[gc.columns.TIME].values,
#         max_segment_gap_s=feature_config.max_segment_gap_s
#     )

#     # Remove any segments that do not adhere to predetermined criteria
#     df = discard_segments(
#         df=df,
#         segment_nr_colname=gc.columns.SEGMENT_NR,
#         min_segment_length_s=feature_config.min_segment_length_s,
#         sampling_frequency=feature_config.sampling_frequency,
#         format='timestamps'
#     )

#     windowed_cols = [
#         gc.columns.TIME, gc.columns.FREE_LIVING_LABEL
#         ] + feature_config.accelerometer_cols + feature_config.gravity_cols
    
#     if step == 'arm_activity':
#         windowed_cols += feature_config.gyroscope_cols
    
#     if subject in gc.participant_ids.PD_IDS:
#         windowed_cols += [gc.columns.ARM_LABEL]

#     windowed_data = tabulate_windows_pdathome(
#         subject=subject,
#         df=df,
#         colnames=windowed_cols,
#         window_length_s=feature_config.window_length_s,
#         window_step_length_s=feature_config.window_step_length_s,
#         sampling_frequency=feature_config.sampling_frequency
#     )


def preprocess_gait_detection(subject):
    print(f"Time {datetime.datetime.now()} - {subject} - Starting preprocessing gait detection ...")
    for side in [gc.descriptives.MOST_AFFECTED_SIDE, gc.descriptives.LEAST_AFFECTED_SIDE]:
        df = pd.read_parquet(os.path.join(gc.paths.PATH_PREPARED_DATA, f'{subject}_{side}.parquet'))

        imu_config = IMUConfig()
        gait_config = GaitFeatureExtractionConfig()
        imu_config.acceleration_units = 'g'

        # Extract relevant gc.columns for accelerometer data
        accel_cols = imu_config.accelerometer_cols

        # Change to correct units [g]
        df[accel_cols] = df[accel_cols] / 9.81 if imu_config.acceleration_units == 'm/s^2' else df[accel_cols]

        # Extract accelerometer data
        accel_data = df[imu_config.accelerometer_cols].values

        filter_configs = {
            "hp": {"result_columns": imu_config.accelerometer_cols, "replace_original": True},
            "lp": {"result_columns": [f'{col}_grav' for col in imu_config.accelerometer_cols], "replace_original": False},
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

        windowed_cols = [
            gc.columns.TIME, gc.columns.FREE_LIVING_LABEL
            ] + gait_config.accelerometer_cols + gait_config.gravity_cols
        
        if subject in gc.participant_ids.PD_IDS:
            windowed_cols += [gc.columns.ARM_LABEL]

        windowed_data = tabulate_windows_pdathome(
            subject=subject,
            df=df,
            colnames=windowed_cols,
            window_length_s=gait_config.window_length_s,
            window_step_length_s=gait_config.window_step_length_s,
            sampling_frequency=gait_config.sampling_frequency
        )

        extractor = WindowedDataExtractor(windowed_cols)

        idx_time = extractor.get_index(gc.columns.TIME)
        idx_activity_label = extractor.get_index(gc.columns.FREE_LIVING_LABEL)
        idx_acc = extractor.get_slice(gait_config.accelerometer_cols)
        idx_grav = extractor.get_slice(gait_config.gravity_cols)

        df_features = pd.DataFrame()

        df_features[gc.columns.TIME] = sorted(windowed_data[:, 0, idx_time])

        if subject in gc.participant_ids.PD_IDS:
            df_features = pd.merge(left=df_features, right=df[[gc.columns.TIME, gc.columns.PRE_OR_POST]], how='left', on=gc.columns.TIME) 

        # Calulate the mode of the labels
        windowed_labels = windowed_data[:, :, idx_activity_label]
        modes_and_counts = np.apply_along_axis(lambda x: compute_mode(x), axis=1, arr=windowed_labels)
        modes, _ = zip(*modes_and_counts)

        df_features[gc.columns.ACTIVITY_LABEL_MAJORITY_VOTING] = modes
        df_features[gc.columns.GAIT_MAJORITY_VOTING] = [is_majority(window) for window in windowed_labels]

        if subject in gc.participant_ids.PD_IDS:
            idx_arm_label = extractor.get_index(gc.columns.ARM_LABEL)
            windowed_labels = windowed_data[:, :, idx_arm_label]
            modes_and_counts = np.apply_along_axis(lambda x: compute_mode(x), axis=1, arr=windowed_labels)
            modes, _ = zip(*modes_and_counts)

            df_features[gc.columns.ARM_LABEL_MAJORITY_VOTING] = modes
            df_features[gc.columns.NO_OTHER_ARM_ACTIVITY_MAJORITY_VOTING] = [is_majority(window, target="Gait without other behaviours or other positions") for window in windowed_labels]

        # compute statistics of the temporal domain signals
        accel_windowed = np.asarray(windowed_data[:, :, idx_acc], dtype=float)
        grav_windowed = np.asarray(windowed_data[:, :, idx_grav], dtype=float)

        df_temporal_features = extract_temporal_domain_features(
            config=gait_config,
            windowed_acc=accel_windowed,
            windowed_grav=grav_windowed,
            grav_stats=['mean', 'std']
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
        
        file_path = os.path.join(gc.paths.PATH_GAIT_FEATURES, f'{subject}_{side}.parquet')
        df_features.to_parquet(file_path)

    print(f"Time {datetime.datetime.now()} - {subject} - Finished preprocessing gait detection.")


def preprocess_filtering_gait(subject):
    print(f"Time {datetime.datetime.now()} - {subject} - Starting preprocessing filtering gait ...")
    for side in [gc.descriptives.MOST_AFFECTED_SIDE, gc.descriptives.LEAST_AFFECTED_SIDE]:

        # load timestamps
        df_ts = pd.read_parquet(os.path.join(gc.paths.PATH_PREPARED_DATA, f'{subject}_{side}.parquet'))
        df_ts[gc.columns.TIME] = df_ts[gc.columns.TIME].round(2)

        # Load gait predictions
        df_pred = pd.read_parquet(os.path.join(gc.paths.PATH_GAIT_PREDICTIONS, gc.classifiers.GAIT_DETECTION_CLASSIFIER_SELECTED, f'{subject}_{side}.parquet'))

        # Load classification threshold
        with open(os.path.join(gc.paths.PATH_THRESHOLDS, 'gait', f'{gc.classifiers.GAIT_DETECTION_CLASSIFIER_SELECTED}.txt'), 'r') as f:
            threshold = float(f.read())
        
        imu_config = IMUConfig()
        gait_config = GaitFeatureExtractionConfig()
        arm_activity_config = ArmActivityFeatureExtractionConfig()

        # MERGE TIMESTAMPS AND PREDICTIONS
        df_ts = merge_predictions_with_timestamps(
            df_ts=df_ts, 
            df_predictions=df_pred, 
            pred_proba_colname=gc.columns.PRED_GAIT_PROBA, 
            window_length_s=gait_config.window_length_s, 
            fs=gait_config.sampling_frequency
        )

        # # Step 1: Expand each window into individual timestamps
        # expanded_data = []
        # for _, row in df_pred.iterrows():
        #     start_time = row[gc.columns.TIME]
        #     proba = row[gc.columns.PRED_GAIT_PROBA]
        #     timestamps = np.arange(start_time, start_time + gait_config.window_length_s, 1/gc.parameters.DOWNSAMPLED_FREQUENCY)
        #     expanded_data.extend(zip(timestamps, [proba] * len(timestamps)))

        # # Create a new DataFrame with expanded timestamps
        # expanded_df = pd.DataFrame(expanded_data, columns=[gc.columns.TIME, gc.columns.PRED_GAIT_PROBA])

        # # Step 2: Round timestamps to avoid floating-point inaccuracies
        # expanded_df[gc.columns.TIME] = expanded_df[gc.columns.TIME].round(2)
        # df_ts[gc.columns.TIME] = df_ts[gc.columns.TIME].round(2)

        # # Step 3: Aggregate by unique timestamps and calculate the mean probability
        # expanded_df = expanded_df.groupby(gc.columns.TIME, as_index=False)[gc.columns.PRED_GAIT_PROBA].mean()

        # df_ts = pd.merge(left=df_ts, right=expanded_df, how='left', on=gc.columns.TIME)

        imu_config.acceleration_units = 'g'

        # Extract relevant gc.columns for accelerometer data
        accel_cols = imu_config.accelerometer_cols

        # Change to correct units [g]
        df_ts[accel_cols] = df_ts[accel_cols] / 9.81 if imu_config.acceleration_units == 'm/s^2' else df_ts[accel_cols]

        # Extract accelerometer data
        accel_data = df_ts[imu_config.accelerometer_cols].values

        filter_configs = {
            "hp": {"result_columns": imu_config.accelerometer_cols, "replace_original": True},
            "lp": {"result_columns": [f'{col}_grav' for col in imu_config.accelerometer_cols], "replace_original": False},
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
        df_ts = df_ts.dropna(subset=gc.columns.GYROSCOPE_COLS)
            
        # Apply threshold and filter data
        df_ts[gc.columns.PRED_GAIT] = (df_ts[gc.columns.PRED_GAIT_PROBA] >= threshold).astype(int)
        
        # Filter unobserved data
        if subject in gc.participant_ids.PD_IDS:
            df_ts = df_ts[df_ts[gc.columns.ARM_LABEL] != 'Cannot assess']
        
        # Use only predicted gait for the subsequent steps
        df_ts = df_ts[df_ts[gc.columns.PRED_GAIT] == 1].reset_index(drop=True)

        # Group consecutive timestamps into segments with new segments starting after a pre-specified gap
        df_ts[gc.columns.SEGMENT_NR] = create_segments(
            time_array=df_ts[gc.columns.TIME].values,
            max_segment_gap_s=arm_activity_config.max_segment_gap_s
        )

        # Remove any segments that do not adhere to predetermined criteria
        df_ts = discard_segments(
            df=df_ts,
            segment_nr_colname=gc.columns.SEGMENT_NR,
            min_segment_length_s=arm_activity_config.min_segment_length_s,
            fs=arm_activity_config.sampling_frequency,
            format='timestamps'
        )

        # Create windows of fixed length and step size from the time series
        windowed_data = []

        windowed_cols = [
            gc.columns.TIME, gc.columns.FREE_LIVING_LABEL
        ] + arm_activity_config.accelerometer_cols + arm_activity_config.gravity_cols + arm_activity_config.gyroscope_cols
        
        if subject in gc.participant_ids.PD_IDS:
            windowed_cols += [gc.columns.ARM_LABEL]

        windowed_data = tabulate_windows_pdathome(
            subject=subject,
            df=df_ts,
            colnames=windowed_cols,
            window_length_s=arm_activity_config.window_length_s,
            window_step_length_s=arm_activity_config.window_step_length_s,
            sampling_frequency=arm_activity_config.sampling_frequency
        )
        
        extractor = WindowedDataExtractor(windowed_cols)

        idx_time = extractor.get_index(gc.columns.TIME)
        idx_activity_label = extractor.get_index(gc.columns.FREE_LIVING_LABEL)
        idx_acc = extractor.get_slice(gait_config.accelerometer_cols)
        idx_grav = extractor.get_slice(gait_config.gravity_cols)
        idx_gyro = extractor.get_slice(gait_config.gyroscope_cols)

        df_features = pd.DataFrame()

        df_features[gc.columns.TIME] = sorted(windowed_data[:, 0, idx_time])

        if subject in gc.participant_ids.PD_IDS:
            df_features = pd.merge(left=df_features, right=df_ts[[gc.columns.TIME, gc.columns.PRE_OR_POST]], how='left', on=gc.columns.TIME) 

        # Calulate the mode of the labels
        windowed_labels = windowed_data[:, :, idx_activity_label]
        modes_and_counts = np.apply_along_axis(lambda x: compute_mode(x), axis=1, arr=windowed_labels)
        modes, _ = zip(*modes_and_counts)

        df_features[gc.columns.ACTIVITY_LABEL_MAJORITY_VOTING] = modes
        df_features[gc.columns.GAIT_MAJORITY_VOTING] = [is_majority(window) for window in windowed_labels]

        if subject in gc.participant_ids.PD_IDS:
            idx_arm_label = extractor.get_index(gc.columns.ARM_LABEL)
            windowed_labels = windowed_data[:, :, idx_arm_label]
            modes_and_counts = np.apply_along_axis(lambda x: compute_mode(x), axis=1, arr=windowed_labels)
            modes, _ = zip(*modes_and_counts)

            df_features[gc.columns.ARM_LABEL_MAJORITY_VOTING] = modes
            df_features[gc.columns.NO_OTHER_ARM_ACTIVITY_MAJORITY_VOTING] = [is_majority(window, target="Gait without other behaviours or other positions") for window in windowed_labels]

        # compute statistics of the temporal domain signals
        accel_windowed = np.asarray(windowed_data[:, :, idx_acc], dtype=float)
        grav_windowed = np.asarray(windowed_data[:, :, idx_grav], dtype=float)
        gyro_windowed = np.asarray(windowed_data[:, :, idx_gyro], dtype=float)

        # compute statistics of the temporal domain accelerometer signals
        arm_activity_config.sensor = 'accelerometer'
        df_temporal_features = extract_temporal_domain_features(arm_activity_config, accel_windowed, grav_windowed, grav_stats=['mean', 'std'])
        df_features = pd.concat([df_features, df_temporal_features], axis=1)

        # transform the accelerometer and gyroscope signals from the temporal domain to the spectral domain
        # using the fast fourier transform and extract spectral features
        for sensor_name, windowed_sensor in zip(['accelerometer', 'gyroscope'], [accel_windowed, gyro_windowed]):
            df_spectral_features = extract_spectral_domain_features(arm_activity_config, sensor_name, windowed_sensor)
            df_features = pd.concat([df_features, df_spectral_features], axis=1)

        file_path = os.path.join(gc.paths.PATH_ARM_ACTIVITY_FEATURES, f'{subject}_{side}.parquet')
        df_features.to_parquet(file_path)


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

    protocol_tier = 'General protocol structure'
    start_code = 1
    end_code = 9

    def validate_input(df, tier_value, code_value):
        required_columns = {'tier', 'code', 'start', 'end'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"DataFrame is missing required columns: {required_columns - set(df.columns)}")
        if df.loc[(df['tier'] == tier_value) & (df['code'] == code_value)].empty:
            raise ValueError(f"No rows found for tier '{tier_value}' and code '{code_value}'.")

    def synchronize_annotations(df, sync_code, tier_value):
        sync_time = df.loc[(df['tier'] == tier_value) & (df['code'] == sync_code), 'start'].values[0]
        df['start'] -= sync_time
        df['end'] -= sync_time
        return df
    
    def adjust_annotations(df, duration):
        df['start'] += duration
        df['end'] += duration
        return df
    
    # Validate inputs
    validate_input(df_annotations_part_1, protocol_tier, start_code)
    validate_input(df_annotations_part_2, protocol_tier, end_code)

    if not isinstance(peakstart, (int, float)) or not isinstance(peakend, (int, float)):
        raise ValueError("Both 'peakstart' and 'peakend' must be numerical values.")

    # Synchronize and adjust annotations
    df_annotations_part_1 = synchronize_annotations(df_annotations_part_1, start_code, protocol_tier)
    df_annotations_part_2 = synchronize_annotations(df_annotations_part_2, end_code, protocol_tier)

    physical_duration = peakend - peakstart
    df_annotations_part_2 = adjust_annotations(df_annotations_part_2, physical_duration)

    # Combine and clean up
    df_annotations = pd.concat([df_annotations_part_1, df_annotations_part_2]).reset_index(drop=True)
    df_annotations = (
        df_annotations.drop(columns=['duration_s'], errors='ignore')
    )

    return df_annotations


def preprocess_video_annotations(df, code_label_map, d_tier_rename):  
    """
    Preprocess video annotations by mapping codes to labels, renaming tiers, 
    and removing unnecessary data.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing video annotations with 'tier', 'code', 'start', and 'end' columns.
        code_label_map (dict): Mapping of tiers to their corresponding code-to-label dictionaries.
        d_tier_rename (dict): Mapping to rename tiers to standardized names.
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame with mapped labels and renamed tiers.
    """
    required_columns = {'tier', 'code'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Input DataFrame must contain the columns: {required_columns}")
    
    # Map codes to labels for each tier
    def map_labels(df, tier, code_to_label_map):
        mask = df['tier'] == tier
        df.loc[mask, 'label'] = df.loc[mask, 'code'].map(code_to_label_map)
    
    for tier, code_to_label_map in code_label_map.items():
        if tier == 'Arm':  # Handle left and right arms
            for arm_tier in ['Left arm', 'Right arm']:
                map_labels(df, arm_tier, code_to_label_map)
        else:
            map_labels(df, tier, code_to_label_map)

    # Rename tiers using the provided mapping
    df['tier'] = df['tier'].map(d_tier_rename)

    # Remove rows with NaN in the 'tier' column (used for synchronization/setup)
    df = df[df['tier'].notna()] 

    return df


def preprocess_sensor_data(df_sensors):
    """
    Preprocess sensor data for analysis.
    
    Steps:
    - Convert time to seconds and datetime format.
    - Resample the data at the desired frequency.
    - Perform cubic spline interpolation to fill missing accelerometer values.
    - Downsample the data to a lower frequency.
    
    Parameters:
        df_sensors (pd.DataFrame): Sensor data with time and accelerometer columns.
    
    Returns:
        pd.DataFrame: Preprocessed sensor data.
    """
    # Convert time to datetime and resample
    df_sensors[gc.columns.TIME + '_dt'] = pd.to_datetime(df_sensors[gc.columns.TIME], unit='s')
    df_sensors = (
        df_sensors.resample(f"{1 / gc.parameters.SAMPLING_FREQUENCY:.3f}s", on=gc.columns.TIME + '_dt')
        .first()
        .reset_index()
    )

    # Update time column
    df_sensors[gc.columns.TIME] = df_sensors[gc.columns.TIME + '_dt'].astype('int64') / 1e9
    df_sensors[gc.columns.TIME] -= df_sensors[gc.columns.TIME].min()

    # Drop unnecessary columns
    df_sensors = df_sensors.drop(columns=[gc.columns.TIME + '_dt'])

    # Interpolate missing values in all accelerometer columns simultaneously
    accelerometer_data = df_sensors[gc.columns.ACCELEROMETER_COLS].copy()
    times = df_sensors[gc.columns.TIME].values

    # Mask to find valid (non-NaN) entries
    valid_mask = accelerometer_data.notna()

    # Only proceed if there's at least one valid entry for interpolation
    if valid_mask.any().any():
        # Perform interpolation for each accelerometer column
        for col in gc.columns.ACCELEROMETER_COLS:
            # Get the valid values for the current column
            valid_times = times[valid_mask[col]]
            valid_values = accelerometer_data[col].dropna().values

            # Create the interpolator function
            interp_func = interp1d(valid_times, valid_values, kind='linear', fill_value='extrapolate')

            # Fill the NaN values with the interpolated values
            accelerometer_data[col] = accelerometer_data[col].fillna(pd.Series(interp_func(times)))

        # Replace the original accelerometer columns with the interpolated data
        df_sensors[gc.columns.ACCELEROMETER_COLS] = accelerometer_data

    # Downsample
    downsample_rate = int(gc.parameters.SAMPLING_FREQUENCY / gc.parameters.DOWNSAMPLED_FREQUENCY)
    df_sensors = df_sensors.iloc[::downsample_rate].reset_index(drop=True)

    return df_sensors


def attach_label_to_signal(df_sensors, df_labels, label_column, tier_column, start_column, end_column, tiers=None):
    # Ensure that 'start' and 'end' columns are numeric
    df_labels[start_column] = pd.to_numeric(df_labels[start_column], errors='coerce')
    df_labels[end_column] = pd.to_numeric(df_labels[end_column], errors='coerce')

    if tiers is None:
        tiers = df_labels[tier_column].unique()

    # Initialize the result DataFrame with a copy of df_sensors
    df_result = df_sensors.copy()

    # Loop through each tier
    for tier in tiers:
        # Filter labels for the current tier
        tier_labels = df_labels[df_labels[tier_column] == tier]

        # Create a column for the current tier's label
        label_column_name = f'{tier}_label'
        df_result[label_column_name] = None  # Initialize the label column with None

        for _, row in tier_labels.iterrows():
            # Find the time range for the current row's start and end
            mask = (df_result[gc.columns.TIME] >= row[start_column]) & (df_result[gc.columns.TIME] < row[end_column])
            df_result.loc[mask, label_column_name] = row[label_column]

    return df_result.reset_index(drop=True)


def determine_med_stage(df, subject, path_annotations):
    prestart, preend, poststart, postend = load_stage_start_end(path_annotations, subject)

    # Create a new column based on the time conditions
    df[gc.columns.PRE_OR_POST] = np.select(
        [
            (df[gc.columns.TIME] >= prestart) & (df[gc.columns.TIME] <= preend),  # Pre-medication condition
            (df[gc.columns.TIME] >= poststart) & (df[gc.columns.TIME] <= postend),  # Post-medication condition
        ], 
        ['pre', 'post'],  # Corresponding labels
        default=np.nan  # Default value if neither condition is met
    )
    
    return df


def rotate_axes(df, subject, wrist):
    """Rotate axes of the sensor data based on the subject and wrist.
    
    1. Swap x and y axes for accelerometer and gyroscope
    2. Invert x-axis accelerometer and y-axis + z-axis gyroscope based on subject and wrist
        a. Subject left wrist correctly installed: Invert x-axis accelerometer.
        b. Subject right wrist correctly installed: Do nothing.
        c. Subject left wrist incorrectly installed: Invert x-axis accelerometer and y-axis + z-axis gyroscope."""
    for sensor in ['accelerometer', 'gyroscope']:
        # Swap x and y axes
        df[f'{sensor}_x_new'] = df[f'{sensor}_y'].copy()
        df[f'{sensor}_y'] = df[f'{sensor}_x'].copy()
        df = df.drop(columns=[f'{sensor}_x'])
        df = df.rename(columns={f'{sensor}_x_new': f'{sensor}_x'})

    # See: https://www.researchgate.net/publication/344382391_Deep_Learning_for_Intake_Gesture_Detection_From_Wrist-Worn_Inertial_Sensors_The_Effects_of_Data_Preprocessing_Sensor_Modalities_and_Sensor_Positions
    # Note that x-y are inverted here compared to the paper
    opposite_wrist_inversion = [gc.columns.ACCELEROMETER_X, gc.columns.GYROSCOPE_Y, gc.columns.GYROSCOPE_Z]

    if wrist == gc.descriptives.LEFT_WRIST and subject in gc.participant_ids.LEFT_NORMAL:
        # Subject has sensor upside down -> invert x-axis and y-axis accelerometer, and x-axis gyroscope (only 
        # the gyroscope axes that are not inverted when the sensor is worn on the opposite wrist)
        df[opposite_wrist_inversion] *= -1
    elif wrist == gc.descriptives.RIGHT_WRIST and subject not in gc.participant_ids.RIGHT_NORMAL:
        # Subject has sensor correctly, but worn on the right wrist -> invert x-axis accelerometer
        # and y-axis + z-axis gyroscope
        df[opposite_wrist_inversion] *= -1

    if (wrist == gc.descriptives.RIGHT_WRIST and subject == 'hbv065') or (wrist == gc.descriptives.LEFT_WRIST and subject in ['hbv058', 'hbv063']):
        df[[gc.columns.ACCELEROMETER_Y, gc.columns.GYROSCOPE_X, gc.columns.GYROSCOPE_Z]] *= -1

    return df


def arm_label_majority_voting(config, arm_label):
    non_nan_count = sum(~pd.isna(arm_label))
    if non_nan_count > config.window_length_s * gc.parameters.DOWNSAMPLED_FREQUENCY / 2:
        return Counter(arm_label).most_common(1)[0][0]
    return np.nan


def add_segment_category(config, df, activity_colname, segment_nr_colname,
                         segment_cat_colname, activity_value):
    
    is_activity = df[activity_colname] == activity_value

    # Create segments based on video-annotations of gait
    df.loc[is_activity, segment_nr_colname] = create_segments(
        time_array=df.loc[is_activity, gc.columns.TIME].values,
        max_segment_gap_s=config.max
    )

    # Further refine only for valid segments
    valid_segments = is_activity & (df[segment_nr_colname].notna())

    # Map categories to segments of video-annotated gait
    df.loc[valid_segments, segment_cat_colname] = categorize_segments(
        df=df.loc[valid_segments],
        config=config
    )

    # Assign default category (-1) for all other rows
    df[segment_cat_colname] = df[segment_cat_colname].fillna(-1)

    # Map segment categories to segments of video-annotated gait
    df[segment_cat_colname] = df[segment_cat_colname].map(mp.segment_map).fillna(-1)
    
    return df