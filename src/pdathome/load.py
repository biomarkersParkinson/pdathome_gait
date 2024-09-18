import h5py
import os
import pandas as pd
import numpy as np

from pdathome.constants import *

def load_sensor_data(path, file_name, tab, subject, wrist_pos):
    with h5py.File(os.path.join(path, file_name)) as opened_file:
        l_subjects_in_file = []
        for ind in range(opened_file[tab][ID_COLNAME].shape[0]):
            l_subjects_in_file.append(''.join([chr(x) for x in np.array(opened_file[opened_file[tab][ID_COLNAME][ind, 0]]).flatten()]))

        if subject in l_subjects_in_file:
            subject_index = l_subjects_in_file.index(subject)
            df_acc =  pd.DataFrame(np.array(opened_file[opened_file[tab][wrist_pos][subject_index,0]]['accel'])).T.set_axis([TIME_COLNAME, DataColumns.ACCELEROMETER_X, DataColumns.ACCELEROMETER_Y, DataColumns.ACCELEROMETER_Z], axis=1)
            df_gyro = pd.DataFrame(np.array(opened_file[opened_file[tab][wrist_pos][subject_index,0]]['gyro'])).T.set_axis([TIME_COLNAME, DataColumns.GYROSCOPE_X, DataColumns.GYROSCOPE_Y, DataColumns.GYROSCOPE_Z], axis=1)
            
            peakstart = opened_file[opened_file[tab]['peakstart'][subject_index][0]][0,0]
            peakend = opened_file[opened_file[tab]['peakend'][subject_index][0]][0,0]
            return pd.merge(left=df_acc, right=df_gyro, how='left', on=TIME_COLNAME), peakstart, peakend
        else:
            print(f"Subject {subject} not in file")
            return
        

def load_video_annotations(path_annotations, subject):

    if subject in L_PD_IDS:
        file_name = f'{subject}_annotations.csv'
        file_name_part_1 = f'{subject}_annotations_part1.csv'
        file_name_part_2 = f'{subject}_annotations_part2.csv'
    else:
        file_name = f'table_{subject}.csv'

    if subject in L_TREMOR_IDS:
        df_tremor = pd.read_csv(os.path.join(path_annotations, f'table_tremor_{subject}.csv'))
        df_tremor = df_tremor.drop(columns=['Label'])
        df_tremor = df_tremor.rename(columns={'Tier': 'tier',
                                            'Start': 'start_s',
                                            'End': 'end_s',
                                            'Duration': 'duration_s',
                                            'Code': 'code'})
        
    if subject not in L_W_PARTS and subject in L_PD_IDS:
        df_annotations = pd.read_csv(os.path.join(path_annotations, file_name), delimiter=',',
                        header=None, names=L_ARM_ACTIVITY_ANNOTATION_COLNAMES)

        split_time = df_annotations.loc[(df_annotations['tier']=='General protocol structure') & (df_annotations['code']==5), 'end_s'].values[0]

        df_annotations_part_1 = df_annotations.loc[df_annotations['end_s']<=split_time].copy()
        df_annotations_part_2 = df_annotations.loc[df_annotations['end_s']>split_time].copy()

        if subject in L_TREMOR_IDS:
            df_tremor_part_1 = df_tremor.loc[df_tremor['end_s']<=split_time].copy()
            df_tremor_part_2 = df_tremor.loc[df_tremor['end_s']>split_time].copy()

    elif subject in L_PD_IDS:
        df_annotations_part_1 = pd.read_csv(os.path.join(path_annotations, file_name_part_1), delimiter=',',
                                            header=None, names=L_ARM_ACTIVITY_ANNOTATION_COLNAMES)
        df_annotations_part_2 = pd.read_csv(os.path.join(path_annotations, file_name_part_2), delimiter=',',
                                            header=None, names=L_ARM_ACTIVITY_ANNOTATION_COLNAMES)
        
        if subject in L_TREMOR_IDS:
            df_tremor_part_1 = df_tremor.loc[df_tremor['end_s']<=df_annotations_part_1['end_s'].max()]
            df_tremor_part_2 = df_tremor.loc[df_tremor['start_s']>=df_annotations_part_1['start_s'].min()]
                                             
    else:
        df_annotations = pd.read_csv(os.path.join(path_annotations, file_name))
    
    if subject in L_PD_IDS:
        df_annotations_part_1.columns = df_annotations_part_1.columns.str.lower()
        df_annotations_part_2.columns = df_annotations_part_2.columns.str.lower()

        if 'duration' in df_annotations_part_1.columns:
            df_annotations_part_1 = df_annotations_part_1.drop(columns=['duration'])

        if 'duration' in df_annotations_part_2.columns:
            df_annotations_part_2 = df_annotations_part_2.drop(columns=['duration'])

        if 'nan' in df_annotations_part_1.columns:
            df_annotations_part_1 = df_annotations_part_1.drop(columns=['nan'])

        if 'nan' in df_annotations_part_2.columns:
            df_annotations_part_2 = df_annotations_part_2.drop(columns=['nan'])

        if subject in L_TREMOR_IDS:
            df_annotations_part_1 = pd.concat([df_annotations_part_1, df_tremor_part_1]).reset_index(drop=True)
            df_annotations_part_2 = pd.concat([df_annotations_part_2, df_tremor_part_2]).reset_index(drop=True)

        df_annotations_part_1 = df_annotations_part_1.reset_index(drop=True)
        df_annotations_part_2 = df_annotations_part_2.reset_index(drop=True)

        return df_annotations_part_1, df_annotations_part_2
    
    else:
        df_annotations.columns = df_annotations.columns.str.lower()

        if 'duration' in df_annotations.columns:
            df_annotations = df_annotations.drop(columns=['duration'])

        df_annotations = df_annotations.reset_index(drop=True)

        return df_annotations
    

def load_stage_start_end(path_labels, subject):
    if subject in L_PD_IDS:
        file_name = 'labels_PD_phys_sharing_v7-3.mat'
        prestart = 'premedstart'
        preend = 'premedend'
        poststart = 'postmedstart'
        postend = 'postmedend'
    else:
        file_name = 'labels_HC_phys_sharing_v7-3.mat'
        prestart = 'prestart'
        preend = 'preend'
        poststart = 'poststart'
        postend = 'postend'
        
    with h5py.File(os.path.join(path_labels, file_name)) as f:
        n_subjects_in_table = f['labels'][ID_COLNAME].shape[0]
        l_subjects_in_table = []

        for ind in range(n_subjects_in_table):
            l_subjects_in_table.append(''.join([chr(x) for x in np.array(f[f['labels'][ID_COLNAME][ind, 0]]).flatten()]))

        if subject in l_subjects_in_table:
            subject_index = l_subjects_in_table.index(subject)
            prestart = f[f['labels'][prestart][subject_index][0]][0,0]
            preend = f[f['labels'][preend][subject_index][0]][0,0]

            if not subject in L_PRE_IDS:
                poststart = f[f['labels'][poststart][subject_index][0]][0,0]
                postend = f[f['labels'][postend][subject_index][0]][0,0]
            else:
                poststart = np.nan
                postend = np.nan
        else:
            print("Subject not in population")
            return
        
        return prestart, preend, poststart, postend
    

def load_dataframes_directory(directory_path, l_ids):
    l_dfs = []
    for id in l_ids:
        df_subj_mas = pd.read_pickle(os.path.join(directory_path, f'{id}_MAS.pkl'))
        df_subj_las = pd.read_pickle(os.path.join(directory_path, f'{id}_LAS.pkl'))
        df_subj_mas['side'] = MOST_AFFECTED_SIDE
        df_subj_las['side'] = LEAST_AFFECTED_SIDE
        df_subj = pd.concat([df_subj_mas, df_subj_las])
        df_subj[ID_COLNAME] = id
        df_subj = df_subj.reset_index(drop=True)
        l_dfs.append(df_subj)
        
    return pd.concat(l_dfs).reset_index(drop=True)