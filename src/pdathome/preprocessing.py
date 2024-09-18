import datetime
import numpy as np
import pandas as pd

from collections import Counter

from pdathome.constants import *
from pdathome.load import load_stage_start_end

from scipy.interpolate import CubicSpline


def determine_wrist_pos(subject, watch_side, d_participant_distribution):
    if watch_side == MOST_AFFECTED_SIDE and subject in d_participant_distribution['most_affected']['right']:
        return RIGHT_WRIST
    elif watch_side == MOST_AFFECTED_SIDE and subject in d_participant_distribution['most_affected']['left']:
        return LEFT_WRIST
    elif watch_side == LEAST_AFFECTED_SIDE and subject in d_participant_distribution['least_affected']['right']:
        return RIGHT_WRIST
    else:
        return LEFT_WRIST
    

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
    df_sensors[TIME_COLNAME+'_s'] = df_sensors[TIME_COLNAME].copy()
    df_sensors[TIME_COLNAME+'_dt'] = df_sensors[TIME_COLNAME+'_s'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    df_sensors = df_sensors.resample(str(1/SAMPLING_FREQUENCY)+'s', on=TIME_COLNAME+'_dt').first().reset_index()
    df_sensors[TIME_COLNAME] = df_sensors[TIME_COLNAME+'_dt'].apply(lambda x: x.timestamp())
    df_sensors = df_sensors.drop(columns=[TIME_COLNAME+'_dt', TIME_COLNAME+'_s'])
    df_sensors[TIME_COLNAME] = df_sensors[TIME_COLNAME] - df_sensors[TIME_COLNAME].min()    

    for col in L_ACCEL_COLS:
        cs = CubicSpline(
            df_sensors.loc[
                df_sensors[col].isna()==False,
                TIME_COLNAME
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
                TIME_COLNAME
                ]
            )
        
    downsample_rate = DOWNSAMPLED_FREQUENCY/SAMPLING_FREQUENCY

    df_sensors = df_sensors[0::int(1/downsample_rate)]

    return df_sensors


def attach_label_to_signal(df_sensors, df_labels, label_column, tier_column, start_column, end_column, l_tiers=None):
    if l_tiers == None:
        l_tiers = df_labels[tier_column].unique()
    for tier in l_tiers:
        for _, row in df_labels.loc[df_labels[tier_column]==tier].iterrows():
            df_sensors.loc[(df_sensors[TIME_COLNAME] >= row[start_column]) & (df_sensors[TIME_COLNAME] < row[end_column]), '{}_label'.format(tier)] = row[label_column]

    df_sensors = df_sensors.reset_index(drop=True)
    return df_sensors


def determine_med_stage(df, subject, watch_side, path_annotations):
    prestart, preend, poststart, postend = load_stage_start_end(path_annotations, subject)

    if subject == 'hbv051' and watch_side == MOST_AFFECTED_SIDE:
        df = df.loc[df[TIME_COLNAME]>=5491.138] 

    df['pre_or_post'] = df.apply(lambda x: 'pre' if x[TIME_COLNAME] >= prestart and x[TIME_COLNAME] <= preend else 
                                        'post' if x[TIME_COLNAME] >= poststart and x[TIME_COLNAME] <= postend else
                                        np.nan, axis=1)
    
    return df


def rotate_axes(df, subject, wrist):
    for sensor in ['accelerometer', 'gyroscope']:
        df[f'{sensor}_x_new'] = df[f'{sensor}_y'].copy()
        df[f'{sensor}_y'] = df[f'{sensor}_x'].copy()
        df = df.drop(columns=[f'{sensor}_x'])
        df = df.rename(columns={f'{sensor}_x_new': f'{sensor}_x'})

    if wrist == LEFT_WRIST and subject in L_L_NORMAL:
        df[DataColumns.ACCELEROMETER_Y] *= -1
    elif wrist == LEFT_WRIST and subject not in L_L_NORMAL:
        df[DataColumns.ACCELEROMETER_X] *= -1
        df[DataColumns.GYROSCOPE_X] *= -1
        df[DataColumns.GYROSCOPE_Y] *= -1
    elif wrist == RIGHT_WRIST and subject in L_R_NORMAL:
        df[DataColumns.ACCELEROMETER_X] *= -1
        df[DataColumns.ACCELEROMETER_Y] *= -1
        df[DataColumns.GYROSCOPE_Z] *= -1
        df[DataColumns.GYROSCOPE_Y] *= -1
    elif wrist == RIGHT_WRIST and subject not in L_R_NORMAL:
        df[DataColumns.GYROSCOPE_X] *= -1
        df[DataColumns.GYROSCOPE_Z] *= -1

    return df


def arm_label_majority_voting(config, arm_label):
    non_nan_count = sum(~pd.isna(arm_label))
    if non_nan_count > config.window_length_s * config.sampling_frequency / 2:
        return Counter(arm_label).most_common(1)[0][0]
    return np.nan