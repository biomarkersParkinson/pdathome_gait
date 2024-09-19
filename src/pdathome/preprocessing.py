import datetime
import json
import numpy as np
import os
import pandas as pd

from collections import Counter
from scipy.interpolate import CubicSpline

from paradigma.feature_extraction import extract_temporal_domain_features, extract_spectral_domain_features, \
    pca_transform_gyroscope, compute_angle, remove_moving_average_angle, signal_to_ffts, get_dominant_frequency, \
    compute_perc_power, extract_angle_extremes, extract_range_of_motion, extract_peak_angular_velocity
from paradigma.gait_analysis_config import GaitFeatureExtractionConfig, ArmSwingFeatureExtractionConfig
from paradigma.imu_preprocessing import butterworth_filter
from paradigma.preprocessing_config import IMUPreprocessingConfig
from paradigma.windowing import tabulate_windows, create_segments, discard_segments

from pdathome.constants import classifiers, columns, descriptives, tiers_labels_map, \
    tiers_rename, parameters, participant_ids, paths
from pdathome.load import load_stage_start_end, load_sensor_data, load_video_annotations
from pdathome.utils import save_to_pickle


def prepare_data(subject):
    print(f"Time {datetime.datetime.now()} - {subject} - Preparing data ...")
    with open(os.path.join(paths.PATH_CLINICAL_DATA, 'distribution_participants.json'), 'r') as f:
        d_participant_distribution = json.load(f)

    l_time_cols = [columns.TIME]
    l_sensor_cols = columns.L_ACCELEROMETER + columns.L_GYROSCOPE
    l_other_cols = [columns.FREE_LIVING_LABEL]

    if subject in participant_ids.L_PD_IDS:
        file_sensor_data = 'phys_cur_PD_merged.mat'
        path_annotations = paths.PATH_ANNOTATIONS_PD
        l_other_cols += [columns.ARM_LABEL, columns.PRE_OR_POST]
    else:
        file_sensor_data = 'phys_cur_HC_merged.mat'
        path_annotations = paths.PATH_ANNOTATIONS_CONTROLS

    if subject in participant_ids.L_TREMOR_IDS:
        l_other_cols.append(columns.TREMOR_LABEL)

    l_cols_to_export = l_time_cols + l_sensor_cols + l_other_cols

    for side in [descriptives.MOST_AFFECTED_SIDE, descriptives.LEAST_AFFECTED_SIDE]:        

        ## loading_annotations
        wrist_pos = determine_wrist_pos(subject, side, d_participant_distribution)
        df_sensors, peakstart, peakend = load_sensor_data(paths.PATH_SENSOR_DATA, file_sensor_data, 'phys', subject, wrist_pos)

        if subject in participant_ids.L_PD_IDS:
            df_annotations_part_1, df_annotations_part_2 = load_video_annotations(path_annotations, subject)
            df_annotations = sync_video_annotations(df_annotations_part_1, df_annotations_part_2, peakstart, peakend)
        else:
            df_annotations = load_video_annotations(path_annotations, subject)

        df_annotations = preprocess_video_annotations(df=df_annotations, code_label_map=tiers_labels_map, d_tier_rename=tiers_rename)
        
        ## merging
        df_sensors[columns.SIDE] = side

        df_sensors = preprocess_sensor_data(df_sensors)

        df_sensors = attach_label_to_signal(df_sensors, df_annotations, 'label', 'tier', 'start_s', 'end_s')

        if subject in participant_ids.L_PD_IDS:
            if wrist_pos == descriptives.RIGHT_WRIST:
                df_sensors[columns.ARM_LABEL] = df_sensors['right_arm_label']
            else:
                df_sensors[columns.ARM_LABEL] = df_sensors['left_arm_label']

            df_sensors[columns.ARM_LABEL] = df_sensors[columns.ARM_LABEL].fillna('non_gait')

        df_sensors = determine_med_stage(df=df_sensors, subject=subject,
                                         watch_side=side,
                                         path_annotations=path_annotations)
        
        df_sensors = rotate_axes(df=df_sensors, subject=subject, wrist=wrist_pos)

        df_sensors = df_sensors.loc[~df_sensors[columns.FREE_LIVING_LABEL].isna()]
        df_sensors = df_sensors.loc[df_sensors[columns.FREE_LIVING_LABEL]!='Unknown']
        df_sensors = df_sensors.reset_index(drop=True)

        

        l_drop_cols = ['clinical_tests_label']
        if subject in participant_ids.L_PD_IDS:
            l_drop_cols += ['med_and_motor_status_label', 'left_arm_label', 'right_arm_label']

        df_sensors = df_sensors.drop(columns=l_drop_cols)

        # temporarily store as pickle until tsdf issue is resolved
        save_to_pickle(
            df=df_sensors[l_cols_to_export],
            path=paths.PATH_DATAFRAMES,
            filename=f'{subject}_{side}.pkl'
        )


def preprocess_gait_detection(subject):
    for side in [descriptives.MOST_AFFECTED_SIDE, descriptives.LEAST_AFFECTED_SIDE]:
        print(f"Time {datetime.datetime.now()} - {subject} {side} - Preprocessing gait ...")
        df = pd.read_pickle(os.path.join(paths.PATH_DATAFRAMES, f'{subject}_{side}.pkl'))

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

        save_to_pickle(
            df=df_windowed[l_export_cols],
            path=paths.PATH_GAIT_FEATURES,
            filename=f'{subject}_{side}.pkl'
        )


def preprocess_filtering_gait(subject):
    for side in [descriptives.MOST_AFFECTED_SIDE, descriptives.LEAST_AFFECTED_SIDE]:
        print(f"Time {datetime.datetime.now()} - {subject} {side} - Processing ...")
        df_pred = pd.read_pickle(os.path.join(paths.PATH_GAIT_PREDICTIONS, classifiers.GAIT_DETECTION_CLASSIFIER_SELECTED, f'{subject}.pkl'))

        with open(os.path.join(paths.PATH_THRESHOLDS, 'gait', f'threshold_{classifiers.GAIT_DETECTION_CLASSIFIER_SELECTED}.txt'), 'r') as f:
            threshold = float(f.read())

        # Configure columns based on cohort
        if subject in participant_ids.L_PD_IDS:
            l_cols_to_export = [columns.TIME, columns.SEGMENT_NR, columns.WINDOW_NR, columns.ARM_LABEL, columns.PRE_OR_POST]
        else:
            l_cols_to_export = [columns.TIME, columns.SEGMENT_NR, columns.WINDOW_NR]

        # Load sensor data
        df_sensors = pd.read_pickle(os.path.join(paths.PATH_DATAFRAMES, f'{subject}_{side}.pkl'))
        df_pred_side = df_pred.loc[df_pred[columns.SIDE]==side].copy()

        imu_config = IMUPreprocessingConfig()
        arm_activity_config = ArmSwingFeatureExtractionConfig()

        imu_config.acceleration_units = 'g'
        arm_activity_config.l_data_point_level_cols += [columns.TIME]

        # Extract relevant columns for accelerometer data
        accel_cols = list(imu_config.d_channels_accelerometer.keys())

        # Change to correct units [g]
        df_sensors[accel_cols] = df_sensors[accel_cols] / 9.81 if imu_config.acceleration_units == 'm/s^2' else df_sensors[accel_cols]

        # Extract the accelerometer data as a 2D array
        accel_data = df_sensors[accel_cols].values

        # Define filtering passbands
        passbands = ['hp', 'lp'] 
        filtered_data = {}

        # Apply Butterworth filter for each passband and result type
        for result, passband in zip(['filt', 'grav'], passbands):
            filtered_data[result] = butterworth_filter(
                sensor_data=accel_data,
                order=imu_config.filter_order,
                cutoff_frequency=imu_config.lower_cutoff_frequency,
                passband=passband,
                sampling_frequency=imu_config.sampling_frequency
            )

        # Create DataFrames from filtered data
        filtered_dfs = {f'{result}_{col}': pd.Series(data[:, i]) for i, col in enumerate(accel_cols) for result, data in filtered_data.items()}

        # Combine filtered columns into DataFrame
        filtered_df = pd.DataFrame(filtered_dfs)

        # Drop original accelerometer columns and append filtered results
        df_sensors = df_sensors.drop(columns=accel_cols).join(filtered_df).rename(columns={col: col.replace('filt_', '') for col in filtered_df.columns})

        # Merge sensor data with predictions
        l_merge_cols = [columns.TIME, columns.FREE_LIVING_LABEL]
        if subject in participant_ids.L_PD_IDS:
            l_merge_cols += [columns.PRE_OR_POST, columns.ARM_LABEL]

        df = pd.merge(left=df_pred_side, right=df_sensors, how='left', on=l_merge_cols).reset_index(drop=True)

        # Process free living label and remove nans
        df['gait_boolean'] = (df[columns.FREE_LIVING_LABEL] == 'Walking').astype(int)
        df = df.dropna(subset=columns.L_GYROSCOPE)
            
        # Apply threshold and filter data
        df[columns.PRED_GAIT] = (df[columns.PRED_GAIT_PROBA] >= threshold).astype(int)

        # Perform principal component analysis on the gyroscope signals to obtain the angular velocity in the
        # direction of the swing of the arm 
        df[columns.VELOCITY] = pca_transform_gyroscope(
            df=df,
            y_gyro_colname=columns.GYROSCOPE_Y,
            z_gyro_colname=columns.GYROSCOPE_Z,
            pred_gait_colname=columns.PRED_GAIT
        )

        # Integrate the angular velocity to obtain an estimation of the angle
        df[columns.ANGLE] = compute_angle(
            velocity_col=df[columns.VELOCITY],
            time_col=df[columns.TIME]
        )

        # Remove the moving average from the angle to account for possible drift caused by the integration
        # of noise in the angular velocity
        df[columns.ANGLE_SMOOTH] = remove_moving_average_angle(
            angle_col=df[columns.ANGLE],
            sampling_frequency=arm_activity_config.sampling_frequency
        )
        
        # Filter unobserved data
        if subject in participant_ids.L_PD_IDS:
            df = df[df[columns.ARM_LABEL] != 'cant assess']
        
        # Use only predicted gait for the subsequent steps
        df = df[df[columns.PRED_GAIT] == 1].reset_index(drop=True)

        # Group consecutive timestamps into segments with new segments starting after a pre-specified gap
        df[columns.SEGMENT_NR] = create_segments(
            time_series=df[columns.TIME],
            minimum_gap_s=arm_activity_config.window_length_s
        )

        # Remove any segments that do not adhere to predetermined criteria
        df = discard_segments(
            df=df,
            time_colname=columns.TIME,
            segment_nr_colname=columns.SEGMENT_NR,
            minimum_segment_length_s=arm_activity_config.window_length_s
        )

        # Create windows of fixed length and step size from the time series
        l_data_point_level_cols = arm_activity_config.l_data_point_level_cols + ([columns.PRE_OR_POST, columns.ARM_LABEL] if subject in participant_ids.L_PD_IDS else [])

        l_dfs = [
            tabulate_windows(
                df=df[df[columns.SEGMENT_NR] == segment_nr].reset_index(drop=True),
                time_column_name=columns.TIME,
                data_point_level_cols=l_data_point_level_cols,
                segment_nr_colname=columns.SEGMENT_NR,
                window_length_s=arm_activity_config.window_length_s,
                window_step_size_s=arm_activity_config.window_step_size_s,
                segment_nr=segment_nr,
                sampling_frequency=arm_activity_config.sampling_frequency
            )
            for segment_nr in df[columns.SEGMENT_NR].unique()
        ]
        l_dfs = [df for df in l_dfs if not df.empty]

        df_windowed = pd.concat(l_dfs).reset_index(drop=True)

        # Save windows with timestamps for later use
        df_windowed[l_cols_to_export].to_pickle(os.path.join(paths.PATH_ARM_ACTIVITY_FEATURES, f'{subject}_{side}_ts.pkl'))

        df_windowed = df_windowed.drop(columns=[columns.TIME])

        # Majority voting for labels per window
        if subject in participant_ids.L_PD_IDS:
            df_windowed[columns.PRE_OR_POST] = df_windowed[columns.PRE_OR_POST].str[0]
            df_windowed[columns.OTHER_ARM_ACTIVITY_MAJORITY_VOTING] = df_windowed[columns.ARM_LABEL].apply(lambda x: x.count('Gait without other behaviours or other positions') < len(x)/2)
            df_windowed[columns.ARM_LABEL_MAJORITY_VOTING] = df_windowed[columns.ARM_LABEL].apply(lambda x: arm_label_majority_voting(arm_activity_config, x))
            df_windowed = df_windowed.drop(columns=[columns.ARM_LABEL])

        # Transform the angle from the temporal domain to the spectral domain using the fast fourier transform
        df_windowed[f'{columns.ANGLE}_freqs'], df_windowed[f'{columns.ANGLE}_fft'] = signal_to_ffts(
            sensor_col=df_windowed[columns.ANGLE_SMOOTH],
            window_type=arm_activity_config.window_type,
            sampling_frequency=arm_activity_config.sampling_frequency
        )

        # Obtain the dominant frequency of the angle signal in the frequency band of interest
        # defined by the highest peak in the power spectrum
        df_windowed[f'{columns.ANGLE}_dominant_frequency'] = df_windowed.apply(
            lambda x: get_dominant_frequency(
                signal_ffts=x[f'{columns.ANGLE}_fft'],
                signal_freqs=x[f'{columns.ANGLE}_freqs'],
                fmin=arm_activity_config.power_band_low_frequency,
                fmax=arm_activity_config.power_band_high_frequency
                ),
                axis=1
        )

        df_windowed = df_windowed.drop(columns=[f'{columns.ANGLE}_fft', f'{columns.ANGLE}_freqs'])

        # Compute the percentage of power in the frequency band of interest (i.e., the frequency band of the arm swing)
        df_windowed[f'{columns.ANGLE}_perc_power'] = df_windowed[columns.ANGLE_SMOOTH].apply(
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
        extract_angle_extremes(
            df=df_windowed,
            angle_colname=columns.ANGLE_SMOOTH,
            dominant_frequency_colname=f'{columns.ANGLE}_dominant_frequency',
            sampling_frequency=arm_activity_config.sampling_frequency
        )

        # Calculate the change in angle between consecutive extrema (minima and maxima) of the angle signal inside the window
        df_windowed[f'{columns.ANGLE}_amplitudes'] = extract_range_of_motion(angle_extrema_values_col=df_windowed[f'{columns.ANGLE}_extrema_values'])

        # Aggregate the changes in angle between consecutive extrema to obtain the range of motion
        df_windowed['range_of_motion'] = df_windowed[f'{columns.ANGLE}_amplitudes'].apply(lambda x: np.mean(x) if len(x) > 0 else 0).replace(np.nan, 0)
        df_windowed = df_windowed.drop(columns=[f'{columns.ANGLE}_amplitudes'])

        # Compute the forward and backward peak angular velocity using the extrema of the angular velocity
        extract_peak_angular_velocity(
            df=df_windowed,
            velocity_colname=columns.VELOCITY,
            angle_minima_colname=f'{columns.ANGLE}_minima',
            angle_maxima_colname=f'{columns.ANGLE}_maxima'
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
        for sensor, l_sensor_colnames in zip(['accelerometer', 'gyroscope'], [columns.L_ACCELEROMETER, columns.L_GYROSCOPE]):
            df_windowed = extract_spectral_domain_features(arm_activity_config, df_windowed, sensor, l_sensor_colnames)
        
        df_windowed.fillna(0, inplace=True)
        df_windowed[columns.SIDE] = side

        l_export_cols = [columns.TIME, columns.PRE_OR_POST, columns.SEGMENT_NR, columns.WINDOW_NR, columns.OTHER_ARM_ACTIVITY_MAJORITY_VOTING, columns.ARM_LABEL_MAJORITY_VOTING] + list(arm_activity_config.d_channels_values.keys())

        save_to_pickle(
            df=df_windowed[l_export_cols],
            path=paths.PATH_ARM_ACTIVITY_FEATURES,
            filename=f'{subject}_{side}.pkl'
        )


def determine_wrist_pos(subject, watch_side, d_participant_distribution):
    if watch_side == descriptives.MOST_AFFECTED_SIDE and subject in d_participant_distribution['most_affected']['right']:
        return descriptives.RIGHT_WRIST
    elif watch_side == descriptives.MOST_AFFECTED_SIDE and subject in d_participant_distribution['most_affected']['left']:
        return descriptives.LEFT_WRIST
    elif watch_side == descriptives.LEAST_AFFECTED_SIDE and subject in d_participant_distribution['least_affected']['right']:
        return descriptives.RIGHT_WRIST
    else:
        return descriptives.LEFT_WRIST
    

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
    df_sensors[columns.TIME+'_s'] = df_sensors[columns.TIME].copy()
    df_sensors[columns.TIME+'_dt'] = df_sensors[columns.TIME+'_s'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    df_sensors = df_sensors.resample(str(1/parameters.SAMPLING_FREQUENCY)+'s', on=columns.TIME+'_dt').first().reset_index()
    df_sensors[columns.TIME] = df_sensors[columns.TIME+'_dt'].apply(lambda x: x.timestamp())
    df_sensors = df_sensors.drop(columns=[columns.TIME+'_dt', columns.TIME+'_s'])
    df_sensors[columns.TIME] = df_sensors[columns.TIME] - df_sensors[columns.TIME].min()    

    for col in columns.L_ACCELEROMETER:
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

    df[columns.PRE_OR_POST] = df.apply(lambda x: 'pre' if x[columns.TIME] >= prestart and x[columns.TIME] <= preend else 
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