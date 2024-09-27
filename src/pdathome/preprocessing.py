import datetime
import json
import numpy as np
import os
import pandas as pd

from typing import Union, List

from collections import Counter
from scipy.interpolate import CubicSpline
from scipy import signal

from paradigma.feature_extraction import extract_temporal_domain_features, extract_spectral_domain_features, \
    pca_transform_gyroscope, compute_angle, remove_moving_average_angle, signal_to_ffts, get_dominant_frequency, \
    compute_perc_power, extract_angle_extremes, extract_range_of_motion, extract_peak_angular_velocity
from paradigma.gait_analysis_config import GaitFeatureExtractionConfig, ArmSwingFeatureExtractionConfig
from paradigma.imu_preprocessing import butterworth_filter
from paradigma.preprocessing_config import IMUPreprocessingConfig
from paradigma.windowing import tabulate_windows, create_segments, discard_segments, categorize_segments

from pdathome.constants import global_constants as gc, Mappings as mp
from pdathome.load import load_stage_start_end, load_sensor_data, load_video_annotations
from pdathome.utils import save_to_pickle


import pandas as pd
import numpy as np

def extract_angle_extremes_2(
        df: pd.DataFrame,
        angle_colname: str,
        dominant_frequency_colname: str,
        sampling_frequency: int = 100,
    ) -> pd.Series:
    """Extract the peaks of the angle (minima and maxima) from the smoothed angle signal that adhere to a set of specific requirements.
    
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe containing the angle signal
    angle_colname: str
        The name of the column containing the smoothed angle signal
    dominant_frequency_colname: str
        The name of the column containing the dominant frequency
    sampling_frequency: int
        The sampling frequency of the data (default: 100)

    Returns
    -------
    pd.Series
        The extracted angle extremes (peaks)
    """
    # determine peaks
    df[f'{angle_colname}_maxima'] = df.apply(
        lambda x: signal.find_peaks(
            x[angle_colname],
            distance=sampling_frequency * 0.6 / x[dominant_frequency_colname], 
            prominence=0.5
        )[0], axis=1
    )
    df[f'{angle_colname}_minima'] = df.apply(
        lambda x: signal.find_peaks(
            [-x for x in x[angle_colname]], 
            distance=sampling_frequency * 0.6 / x[dominant_frequency_colname], 
            prominence=0.5
        )[0], axis=1
    ) 

    df[f'{angle_colname}_maxima'] = df[f'{angle_colname}_maxima'].apply(lambda x: x if isinstance(x, list) else x.tolist())
    df[f'{angle_colname}_minima'] = df[f'{angle_colname}_minima'].apply(lambda x: x if isinstance(x, list) else x.tolist())

    df[f'{angle_colname}_new_minima'] = df[f'{angle_colname}_minima'].copy()
    df[f'{angle_colname}_new_maxima'] = df[f'{angle_colname}_maxima'].copy()

    for index, _ in df.iterrows():
        i_pks = 0                                       # iterable to keep track of consecutive min-min and max-max versus min-max
        n_min = len(df.loc[index, f'{angle_colname}_new_minima'])  # number of minima in window
        n_max = len(df.loc[index, f'{angle_colname}_new_maxima'])  # number of maxima in window

        if n_min > 0 and n_max > 0: 
            if df.loc[index, f'{angle_colname}_new_maxima'][0] > df.loc[index, f'{angle_colname}_new_minima'][0]: # if first minimum occurs before first maximum, start with minimum
                while i_pks < n_min - 1 and i_pks < n_max: # only continue if there's enough minima and maxima to perform operations
                    if df.loc[index, f'{angle_colname}_new_minima'][i_pks+1] < df.loc[index, f'{angle_colname}_new_maxima'][i_pks]: # if angle of next minimum comes before the current maxima, we have two minima in a row
                        if df.loc[index, angle_colname][df.loc[index, f'{angle_colname}_new_minima'][i_pks+1]] < df.loc[index, angle_colname][df.loc[index, f'{angle_colname}_new_minima'][i_pks]]: # if second minimum if smaller than first, keep second
                            df.at[index, f'{angle_colname}_new_minima'] = np.delete(df.loc[index, f'{angle_colname}_new_minima'], i_pks).tolist()
                        else: # otherwise keep the first
                            df.at[index, f'{angle_colname}_new_minima'] = np.delete(df.loc[index, f'{angle_colname}_new_minima'], i_pks+1).tolist()
                        n_min -= 1
                        i_pks -= 1
                    if i_pks >= 0 and df.loc[index, f'{angle_colname}_new_minima'][i_pks] > df.loc[index, f'{angle_colname}_new_maxima'][i_pks]:
                        if df.loc[index, angle_colname][df.loc[index, f'{angle_colname}_new_maxima'][i_pks]] < df.loc[index, angle_colname][df.loc[index, f'{angle_colname}_new_maxima'][i_pks-1]]:
                            df.at[index, f'{angle_colname}_new_maxima'] = np.delete(df.loc[index, f'{angle_colname}_new_maxima'], i_pks).tolist()
                        else:
                            df.at[index, f'{angle_colname}_new_maxima'] = np.delete(df.loc[index, f'{angle_colname}_new_maxima'], i_pks-1).tolist()
                        n_max -= 1
                        i_pks -= 1
                    i_pks += 1

            elif df.loc[index, f'{angle_colname}_new_maxima'][0] < df.loc[index, f'{angle_colname}_new_minima'][0]: # if the first maximum occurs before the first minimum, start with the maximum
                while i_pks < n_min and i_pks < n_max - 1:
                    if df.loc[index, f'{angle_colname}_new_minima'][i_pks] > df.loc[index, f'{angle_colname}_new_maxima'][i_pks+1]:
                        if df.loc[index, angle_colname][df.loc[index, f'{angle_colname}_new_maxima'][i_pks+1]] > df.loc[index, angle_colname][df.loc[index, f'{angle_colname}_new_maxima'][i_pks]]:
                            df.at[index, f'{angle_colname}_new_maxima'] = np.delete(df.loc[index, f'{angle_colname}_new_maxima'], i_pks).tolist()
                        else:
                            df.at[index, f'{angle_colname}_new_maxima'] = np.delete(df.loc[index, f'{angle_colname}_new_maxima'], i_pks+1).tolist()
                        n_max -= 1
                        i_pks -= 1
                    if i_pks > 0 and df.loc[index, f'{angle_colname}_new_minima'][i_pks] < df.loc[index, f'{angle_colname}_new_maxima'][i_pks]:
                        if df.loc[index, angle_colname][df.loc[index, f'{angle_colname}_new_minima'][i_pks]] < df.loc[index, angle_colname][df.loc[index, f'{angle_colname}_new_minima'][i_pks-1]]:
                            df.at[index, f'{angle_colname}_new_minima'] = np.delete(df.loc[index, f'{angle_colname}_new_minima'], i_pks-1).tolist()
                        else:
                            df.at[index, f'{angle_colname}_new_minima'] = np.delete(df.loc[index, f'{angle_colname}_new_minima'], i_pks).tolist()
                        n_min -= 1
                        i_pks -= 1
                    i_pks += 1

    # for some peculiar reason, if a single item remains in the row for {angle_colname}_new_minima or
    # angle_new_maxima, it could be either a scalar or a vector.
    for col in [f'{angle_colname}_new_minima', f'{angle_colname}_new_maxima']:
        df[col] = df[col].apply(lambda x: [x] if not isinstance(x, list) else x)

    # extract amplitude
    df[f'{angle_colname}_extrema_values'] = df.apply(
        lambda x: [x[angle_colname][i] for i in sorted(x[f'{angle_colname}_new_minima'] + x[f'{angle_colname}_new_maxima'])],
        axis=1
    ) 

    return df[f'{angle_colname}_extrema_values']


def prepare_data(subject):
    print(f"Time {datetime.datetime.now()} - {subject} - Starting preparing data ...")
    with open(os.path.join(gc.gc.paths.PATH_CLINICAL_DATA, 'distribution_participants.json'), 'r') as f:
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

        config = IMUPreprocessingConfig()
        config.acceleration_units = 'g'

        # Extract relevant gc.columns for accelerometer data
        accel_cols = list(config.d_channels_accelerometer.keys())

        # Change to correct units [g]
        df[accel_cols] = df[accel_cols] / 9.81 if config.acceleration_units == 'm/s^2' else df[accel_cols]

        # Define filtering passbands
        passbands = ['hp', 'lp'] 

        # Apply Butterworth filter for each passband
        for col in accel_cols:
            for result, passband in zip(['filt', 'grav'], passbands):
                df[f'{result}_{col}'] = butterworth_filter(
                    single_sensor_col=np.array(df[col]),
                    order=config.filter_order,
                    cutoff_frequency=config.lower_cutoff_frequency,
                    passband=passband,
                    sampling_frequency=gc.parameters.DOWNSAMPLED_FREQUENCY
                )

        # Drop original accelerometer gc.columns and append filtered results
        df = df.drop(columns=accel_cols).rename(columns={f'filt_{col}': col for col in accel_cols})
        
        df[gc.columns.TRUE_GAIT_SEGMENT_NR] = create_segments(
            df=df.loc[df['gait_boolean'] == 1],
            time_column_name=gc.columns.TIME,
            gap_threshold_s=1.5
        )

        config = GaitFeatureExtractionConfig()

        config.l_data_point_level_cols += [gc.columns.TIME, gc.columns.TRUE_GAIT_SEGMENT_NR, gc.columns.FREE_LIVING_LABEL]
        l_ts_cols = [gc.columns.TIME, gc.columns.TRUE_GAIT_SEGMENT_NR, gc.columns.WINDOW_NR, gc.columns.FREE_LIVING_LABEL]
        l_export_cols = [gc.columns.TIME, gc.columns.WINDOW_NR, gc.columns.ACTIVITY_LABEL_MAJORITY_VOTING, gc.columns.GAIT_MAJORITY_VOTING] + list(config.d_channels_values.keys())
        l_single_value_cols = None
        if subject in gc.participant_ids.L_PD_IDS:
            config.l_data_point_level_cols.append(gc.columns.ARM_LABEL)
            l_ts_cols += [gc.columns.PRE_OR_POST, gc.columns.ARM_LABEL]
            l_export_cols += [gc.columns.PRE_OR_POST, gc.columns.ARM_LABEL_MAJORITY_VOTING]
            l_single_value_cols = [gc.columns.PRE_OR_POST]
        if subject in gc.participant_ids.L_TREMOR_IDS:
            config.l_data_point_level_cols.append(gc.columns.TREMOR_LABEL)
            l_ts_cols.append(gc.columns.TREMOR_LABEL)


        df_windowed = tabulate_windows(
                df=df,
                window_size=config.window_length_s * gc.parameters.DOWNSAMPLED_FREQUENCY,
                step_size=config.window_step_size_s * gc.parameters.DOWNSAMPLED_FREQUENCY,
                time_column_name=gc.columns.TIME,
                single_value_cols=l_single_value_cols,
                list_value_cols=config.l_data_point_level_cols,
                agg_func='first',
        )
        
        # store windows with timestamps for later use
        df_windowed[l_ts_cols].to_pickle(os.path.join(gc.paths.PATH_GAIT_FEATURES, f'{subject}_{side}_ts.pkl'))

        # Determine most prevalent activity
        df_windowed[gc.columns.ACTIVITY_LABEL_MAJORITY_VOTING] = df_windowed[gc.columns.FREE_LIVING_LABEL].apply(lambda x: pd.Series(x).mode()[0])

        # Determine if the majority of the window is walking
        df_windowed[gc.columns.GAIT_MAJORITY_VOTING] = df_windowed[gc.columns.FREE_LIVING_LABEL].apply(lambda x: x.count('Walking') >= len(x)/2)

        if subject in gc.participant_ids.L_PD_IDS:
            df_windowed[gc.columns.ARM_LABEL_MAJORITY_VOTING] = df_windowed[gc.columns.ARM_LABEL].apply(lambda x: arm_label_majority_voting(config, x))

        df_windowed = df_windowed.drop(columns=[x for x in l_ts_cols if x not in [gc.columns.WINDOW_NR, gc.columns.PRE_OR_POST]])

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

        save_to_pickle(
            df=df_windowed[l_export_cols],
            path=gc.paths.PATH_GAIT_FEATURES,
            filename=f'{subject}_{side}.pkl'
        )

    print(f"Time {datetime.datetime.now()} - {subject} - Finished preprocessing gait detection.")


def preprocess_filtering_gait(subject):
    print(f"Time {datetime.datetime.now()} - {subject} - Starting preprocessing filtering gait ...")
    for side in [gc.descriptives.MOST_AFFECTED_SIDE, gc.descriptives.LEAST_AFFECTED_SIDE]:
        df_pred = pd.read_pickle(os.path.join(gc.paths.PATH_GAIT_PREDICTIONS, gc.classifiers.GAIT_DETECTION_CLASSIFIER_SELECTED, f'{subject}.pkl'))

        with open(os.path.join(gc.paths.PATH_THRESHOLDS, 'gait', f'{gc.classifiers.GAIT_DETECTION_CLASSIFIER_SELECTED}.txt'), 'r') as f:
            threshold = float(f.read())

        # Configure gc.columns based on cohort
        if subject in gc.participant_ids.L_PD_IDS:
            l_cols_to_export = [gc.columns.TIME, gc.columns.PRED_GAIT_SEGMENT_NR, gc.columns.WINDOW_NR, gc.columns.ARM_LABEL, gc.columns.PRE_OR_POST]
        else:
            l_cols_to_export = [gc.columns.TIME, gc.columns.PRED_GAIT_SEGMENT_NR, gc.columns.WINDOW_NR]

        # Load sensor data
        df_sensors = pd.read_pickle(os.path.join(gc.paths.PATH_PREPARED_DATA, f'{subject}_{side}.pkl'))
        df_pred_side = df_pred.loc[df_pred[gc.columns.SIDE]==side].copy()

        imu_config = IMUPreprocessingConfig()
        arm_activity_config = ArmSwingFeatureExtractionConfig()

        imu_config.acceleration_units = 'g'
        arm_activity_config.l_data_point_level_cols += [gc.columns.TIME]

        # Extract relevant gc.columns for accelerometer data
        accel_cols = list(imu_config.d_channels_accelerometer.keys())

        # Change to correct units [g]
        df_sensors[accel_cols] = df_sensors[accel_cols] / 9.81 if imu_config.acceleration_units == 'm/s^2' else df_sensors[accel_cols]

        # Apply Butterworth filter for each passband
        for col in accel_cols:
            for result, passband in zip(['filt', 'grav'], ['hp', 'lp']):
                df_sensors[f'{result}_{col}'] = butterworth_filter(
                    single_sensor_col=np.array(df_sensors[col]),
                    order=imu_config.filter_order,
                    cutoff_frequency=imu_config.lower_cutoff_frequency,
                    passband=passband,
                    sampling_frequency=imu_config.sampling_frequency
                )

        # Drop original accelerometer gc.columns
        df_sensors = df_sensors.drop(columns=accel_cols).rename(columns={f'filt_{col}': col for col in accel_cols})

        # Merge sensor data with predictions
        l_merge_cols = [gc.columns.TIME, gc.columns.FREE_LIVING_LABEL]
        if subject in gc.participant_ids.L_PD_IDS:
            l_merge_cols.append(gc.columns.ARM_LABEL)
            if gc.columns.PRE_OR_POST in df_pred_side.columns:
                l_merge_cols.append(gc.columns.PRE_OR_POST)

        df = pd.merge(left=df_pred_side, right=df_sensors, how='left', on=l_merge_cols).reset_index(drop=True)

        # Process free living label and remove nans
        df['gait_boolean'] = (df[gc.columns.FREE_LIVING_LABEL] == 'Walking').astype(int)
        df = df.dropna(subset=gc.columns.L_GYROSCOPE)
            
        # Apply threshold and filter data
        df[gc.columns.PRED_GAIT] = (df[gc.columns.PRED_GAIT_PROBA] >= threshold).astype(int)

        # Perform principal component analysis on the gyroscope signals to obtain the angular velocity in the
        # direction of the swing of the arm 
        df[gc.columns.VELOCITY] = pca_transform_gyroscope(
            df=df,
            y_gyro_colname=gc.columns.GYROSCOPE_Y,
            z_gyro_colname=gc.columns.GYROSCOPE_Z,
            pred_gait_colname=gc.columns.PRED_GAIT
        )

        # Integrate the angular velocity to obtain an estimation of the angle
        df[gc.columns.ANGLE] = compute_angle(
            velocity_col=df[gc.columns.VELOCITY],
            time_col=df[gc.columns.TIME]
        )

        # Remove the moving average from the angle to account for possible drift caused by the integration
        # of noise in the angular velocity
        df[gc.columns.ANGLE_SMOOTH] = remove_moving_average_angle(
            angle_col=df[gc.columns.ANGLE],
            sampling_frequency=arm_activity_config.sampling_frequency
        )
        
        # Filter unobserved data
        if subject in gc.participant_ids.L_PD_IDS:
            df = df[df[gc.columns.ARM_LABEL] != 'cant assess']

        # Group consecutive timestamps into segments with new segments starting after a pre-specified gap
        df[gc.columns.TRUE_GAIT_SEGMENT_NR] = create_segments(
            df=df.loc[df['gait_boolean']==1],
            time_column_name=gc.columns.TIME,
            gap_threshold_s=gc.parameters.SEGMENT_GAP_GAIT
        )

        df[gc.columns.TRUE_GAIT_SEGMENT_CAT] = categorize_segments(
            df=df.loc[df['gait_boolean']==1],
            segment_nr_colname=gc.columns.TRUE_GAIT_SEGMENT_NR,
            sampling_frequency=gc.parameters.DOWNSAMPLED_FREQUENCY
        )
        
        # Use only predicted gait for the subsequent steps
        df = df[df[gc.columns.PRED_GAIT] == 1].reset_index(drop=True)

        # Group consecutive timestamps into segments with new segments starting after a pre-specified gap
        df[gc.columns.PRED_GAIT_SEGMENT_NR] = create_segments(
            df=df,
            time_column_name=gc.columns.TIME,
            gap_threshold_s=gc.parameters.SEGMENT_GAP_GAIT
        )

        df[gc.columns.PRED_GAIT_SEGMENT_CAT] = categorize_segments(
            df=df,
            segment_nr_colname=gc.columns.PRED_GAIT_SEGMENT_NR,
            sampling_frequency=gc.parameters.DOWNSAMPLED_FREQUENCY
        )

        # Remove any segments that do not adhere to predetermined criteria
        df = discard_segments(
            df=df,
            segment_nr_colname=gc.columns.PRED_GAIT_SEGMENT_NR,
            min_length_segment_s=arm_activity_config.window_length_s,
            sampling_frequency=arm_activity_config.sampling_frequency
        )

        # Create windows of fixed length and step size from the time series
        arm_activity_config.l_data_point_level_cols += [
            gc.columns.TRUE_GAIT_SEGMENT_NR, gc.columns.TRUE_GAIT_SEGMENT_CAT
        ]
        if subject in gc.participant_ids.L_PD_IDS:
            l_single_value_cols = [
                gc.columns.PRE_OR_POST, gc.columns.PRED_GAIT_SEGMENT_NR, gc.columns.PRED_GAIT_SEGMENT_CAT
            ]
            arm_activity_config.l_data_point_level_cols.append(gc.columns.ARM_LABEL)
        else:
            l_single_value_cols = None

        l_dfs = [
            tabulate_windows(
                df=df[df[gc.columns.PRED_GAIT_SEGMENT_NR] == segment_nr].reset_index(drop=True),
                window_size=int(arm_activity_config.window_length_s * arm_activity_config.sampling_frequency),
                step_size=int(arm_activity_config.window_step_size_s * arm_activity_config.sampling_frequency),
                time_column_name=gc.columns.TIME,
                list_value_cols=arm_activity_config.l_data_point_level_cols,
                single_value_cols=l_single_value_cols,
            )
            for segment_nr in df[gc.columns.PRED_GAIT_SEGMENT_NR].unique()
        ]
        l_dfs = [df for df in l_dfs if not df.empty]

        df_windowed = pd.concat(l_dfs).reset_index(drop=True)

        # Save windows with timestamps for later use
        df_windowed[l_cols_to_export].to_pickle(os.path.join(gc.paths.PATH_ARM_ACTIVITY_FEATURES, f'{subject}_{side}_ts.pkl'))

        df_windowed = df_windowed.drop(columns=[gc.columns.TIME])

        # Majority voting for labels per window
        if subject in gc.participant_ids.L_PD_IDS:
            df_windowed[gc.columns.OTHER_ARM_ACTIVITY_MAJORITY_VOTING] = df_windowed[gc.columns.ARM_LABEL].apply(lambda x: x.count('Gait without other behaviours or other positions') < len(x)/2)
            df_windowed[gc.columns.ARM_LABEL_MAJORITY_VOTING] = df_windowed[gc.columns.ARM_LABEL].apply(lambda x: arm_label_majority_voting(arm_activity_config, x))
            df_windowed = df_windowed.drop(columns=[gc.columns.ARM_LABEL])

        # Transform the angle from the temporal domain to the spectral domain using the fast fourier transform
        df_windowed[f'{gc.columns.ANGLE}_freqs'], df_windowed[f'{gc.columns.ANGLE}_fft'] = signal_to_ffts(
            sensor_col=df_windowed[gc.columns.ANGLE_SMOOTH],
            window_type=arm_activity_config.window_type,
            sampling_frequency=arm_activity_config.sampling_frequency
        )

        # Obtain the dominant frequency of the angle signal in the frequency band of interest
        # defined by the highest peak in the power spectrum
        df_windowed[f'{gc.columns.ANGLE}_dominant_frequency'] = df_windowed.apply(
            lambda x: get_dominant_frequency(
                signal_ffts=x[f'{gc.columns.ANGLE}_fft'],
                signal_freqs=x[f'{gc.columns.ANGLE}_freqs'],
                fmin=arm_activity_config.power_band_low_frequency,
                fmax=arm_activity_config.power_band_high_frequency
                ),
                axis=1
        )

        df_windowed = df_windowed.drop(columns=[f'{gc.columns.ANGLE}_fft', f'{gc.columns.ANGLE}_freqs'])

        # Compute the percentage of power in the frequency band of interest (i.e., the frequency band of the arm swing)
        df_windowed[f'{gc.columns.ANGLE}_perc_power'] = df_windowed[gc.columns.ANGLE_SMOOTH].apply(
            lambda x: compute_perc_power(
                sensor_col=x,
                fmin_band=arm_activity_config.power_band_low_frequency,
                fmax_band=arm_activity_config.power_band_high_frequency,
                fmin_total=arm_activity_config.spectrum_low_frequency,
                fmax_total=arm_activity_config.spectrum_high_frequency,
                sampling_frequency=arm_activity_config.sampling_frequency,
                window_type=arm_activity_config.window_type
            )
        )

        # Determine the extrema (minima and maxima) of the angle signal
        df_windowed[f'{angle_colname}_extrema_values'] = extract_angle_extremes_2(
            df=df_windowed,
            angle_colname=gc.columns.ANGLE_SMOOTH,
            dominant_frequency_colname=f'{gc.columns.ANGLE}_dominant_frequency',
            sampling_frequency=arm_activity_config.sampling_frequency
        )

        # Calculate the change in angle between consecutive extrema (minima and maxima) of the angle signal inside the window
        df_windowed[f'{gc.columns.ANGLE}_amplitudes'] = extract_range_of_motion(angle_extrema_values_col=df_windowed[f'{gc.columns.ANGLE}_extrema_values'])

        # Aggregate the changes in angle between consecutive extrema to obtain the range of motion
        df_windowed['range_of_motion'] = df_windowed[f'{gc.columns.ANGLE}_amplitudes'].apply(lambda x: np.mean(x) if len(x) > 0 else 0).replace(np.nan, 0)
        df_windowed = df_windowed.drop(columns=[f'{gc.columns.ANGLE}_amplitudes'])

        # Compute the forward and backward peak angular velocity using the extrema of the angular velocity
        extract_peak_angular_velocity(
            df=df_windowed,
            velocity_colname=gc.columns.VELOCITY,
            angle_minima_colname=f'{gc.columns.ANGLE}_minima',
            angle_maxima_colname=f'{gc.columns.ANGLE}_maxima'
        )

        # Compute aggregated measures of the peak angular velocity
        for dir in ['forward', 'backward']:
            df_windowed[f'{dir}_peak_ang_vel_mean'] = df_windowed[f'{dir}_peak_ang_vel'].apply(lambda x: np.mean(x) if len(x) > 0 else 0)
            df_windowed[f'{dir}_peak_ang_vel_std'] = df_windowed[f'{dir}_peak_ang_vel'].apply(lambda x: np.std(x) if len(x) > 0 else 0)
            df_windowed = df_windowed.drop(columns=[f'{dir}_peak_ang_vel'])

        # Compute statistics of the temporal domain accelerometer signals
        df_windowed = extract_temporal_domain_features(arm_activity_config, df_windowed, l_gravity_stats=['mean', 'std'])

        # Transform the accelerometer and gyroscope signals from the temporal domain to the spectral domain
        # using the fast fourier transform and extract spectral features
        for sensor, l_sensor_colnames in zip(['accelerometer', 'gyroscope'], [gc.columns.L_ACCELEROMETER, gc.columns.L_GYROSCOPE]):
            df_windowed = extract_spectral_domain_features(arm_activity_config, df_windowed, sensor, l_sensor_colnames)
        
        df_windowed.fillna(0, inplace=True)
        df_windowed[gc.columns.SIDE] = side

        l_export_cols = [
            gc.columns.TIME, gc.columns.WINDOW_NR, gc.columns.PRED_GAIT_SEGMENT_NR, gc.columns.PRED_GAIT_SEGMENT_CAT,
            gc.columns.TRUE_GAIT_SEGMENT_NR, gc.columns.TRUE_GAIT_SEGMENT_CAT] + list(arm_activity_config.d_channels_values.keys())

        if subject in gc.participant_ids.L_PD_IDS:
            l_export_cols += [gc.columns.PRE_OR_POST, gc.columns.ARM_LABEL_MAJORITY_VOTING, gc.columns.OTHER_ARM_ACTIVITY_MAJORITY_VOTING]

        save_to_pickle(
            df=df_windowed[l_export_cols],
            path=gc.paths.PATH_ARM_ACTIVITY_FEATURES,
            filename=f'{subject}_{side}.pkl'
        )

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

    if 'start' in df.gc.columns:
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