import h5py
import os
import pandas as pd
import numpy as np

from pdathome.constants import global_constants as gc

def load_sensor_data(path, file_name, tab, subject, wrist_pos):
    with h5py.File(os.path.join(path, file_name), 'r') as opened_file:
        ids_dataset = opened_file[tab][gc.columns.ID]
        l_subjects_in_file = [
            ''.join(map(chr, opened_file[ids_dataset[ind, 0]][()].flatten()))
            for ind in range(ids_dataset.shape[0])
        ]

        if subject not in l_subjects_in_file:
            print(f"Subject {subject} not in file")
            return None, None, None

    
        subject_index = l_subjects_in_file.index(subject)
        wrist_data = opened_file[opened_file[tab][wrist_pos][subject_index, 0]]

        df_acc = pd.DataFrame(wrist_data['accel'][()].T, columns=[
            gc.columns.TIME, gc.columns.ACCELEROMETER_X, gc.columns.ACCELEROMETER_Y, gc.columns.ACCELEROMETER_Z
        ])
        df_gyro = pd.DataFrame(wrist_data['gyro'][()].T, columns=[
            gc.columns.TIME, gc.columns.GYROSCOPE_X, gc.columns.GYROSCOPE_Y, gc.columns.GYROSCOPE_Z
        ])
        
        peakstart = opened_file[opened_file[tab]['peakstart'][subject_index][0]][0,0]
        peakend = opened_file[opened_file[tab]['peakend'][subject_index][0]][0,0]

        df_sensors = pd.merge(left=df_acc, right=df_gyro, how='left', on=gc.columns.TIME)
        return df_sensors, peakstart, peakend

        

def load_video_annotations(path_annotations, subject):
    def process_annotations(file_path, column_names):
        """Reads and processes an annotation file."""
        if f'{subject}_annotations' in file_path:
            df = pd.read_csv(file_path, delimiter=',', header=None, names=column_names)
        else:
            df = pd.read_csv(file_path, delimiter=',')

        df.columns = df.columns.str.lower()

        if 'duration' in df.columns:
            df = df.drop(columns=['duration'])
        if 'nan' in df.columns:
            df = df.drop(columns=['nan'])

        return df.reset_index(drop=True)
    
    # Define file names
    if subject in gc.participant_ids.PD_IDS:
        file_name = f"{subject}_annotations.csv"
        file_name_part_1 = f"{subject}_annotations_part1.csv"
        file_name_part_2 = f"{subject}_annotations_part2.csv"
    else:
        file_name = f"table_{subject}.csv"

    # Load tremor annotations if applicable
    if subject in gc.participant_ids.TREMOR_IDS:
        tremor_file = os.path.join(path_annotations, f"table_tremor_{subject}.csv")
        df_tremor = process_annotations(tremor_file, gc.columns.ARM_ACTIVITY_ANNOTATIONS)
    else:
        df_tremor = None
        
    # Handle PD subjects without parts
    if subject in gc.participant_ids.PD_IDS and subject not in gc.participant_ids.W_PARTS:
        annotations_path = os.path.join(path_annotations, file_name)
        df_annotations = process_annotations(annotations_path, gc.columns.ARM_ACTIVITY_ANNOTATIONS)

        # Split based on protocol structure
        split_time = df_annotations.loc[
            (df_annotations['tier'] == 'General protocol structure') & (df_annotations['code'] == 5),
            'end'
        ].values[0]

        df_annotations_part_1 = df_annotations.loc[df_annotations['end'] <= split_time].copy()
        df_annotations_part_2 = df_annotations.loc[df_annotations['end'] > split_time].copy()

        if df_tremor is not None:
            df_tremor_part_1 = df_tremor.loc[df_tremor['end'] <= split_time].copy()
            df_tremor_part_2 = df_tremor.loc[df_tremor['end'] > split_time].copy()
            
            df_annotations_part_1 = pd.concat([df_annotations_part_1, df_tremor_part_1], ignore_index=True)
            df_annotations_part_2 = pd.concat([df_annotations_part_2, df_tremor_part_2], ignore_index=True)

    # Handle PD subjects with parts
    elif subject in gc.participant_ids.PD_IDS:
        annotations_path_part_1 = os.path.join(path_annotations, file_name_part_1)
        annotations_path_part_2 = os.path.join(path_annotations, file_name_part_2)
        df_annotations_part_1 = process_annotations(annotations_path_part_1, gc.columns.ARM_ACTIVITY_ANNOTATIONS)
        df_annotations_part_2 = process_annotations(annotations_path_part_2, gc.columns.ARM_ACTIVITY_ANNOTATIONS)
        
        if df_tremor is not None:
            df_tremor_part_1 = df_tremor.loc[df_tremor['end'] <= df_annotations_part_1['end'].max()]
            df_tremor_part_2 = df_tremor.loc[df_tremor['start'] >= df_annotations_part_1['start'].min()]

            df_annotations_part_1 = pd.concat([df_annotations_part_1, df_tremor_part_1], ignore_index=True)
            df_annotations_part_2 = pd.concat([df_annotations_part_2, df_tremor_part_2], ignore_index=True)
                                             
    # Handle non-PD subjects
    else:
        annotations_path = os.path.join(path_annotations, file_name)
        df_annotations = process_annotations(annotations_path, gc.columns.ARM_ACTIVITY_ANNOTATIONS)
        return df_annotations

    return df_annotations_part_1, df_annotations_part_2
    

def load_stage_start_end(path_labels, subject_id):
    if subject_id in gc.participant_ids.PD_IDS:
        file_name = 'labels_PD_phys_sharing_v7-3.mat'
        label_map = {'prestart': 'premedstart', 'preend': 'premedend', 'poststart': 'postmedstart', 'postend': 'postmedend'}
    else:
        file_name = 'labels_HC_phys_sharing_v7-3.mat'
        label_map = {'prestart': 'prestart', 'preend': 'preend', 'poststart': 'poststart', 'postend': 'postend'}

    file_path = os.path.join(path_labels, file_name)
        
    with h5py.File(file_path) as f:
        # Extract subject IDs from the .mat file
        subjects_table = f['labels'][gc.columns.ID]
        subject_ids = [
            ''.join([chr(x) for x in np.array(f[subjects_table[ind, 0]]).flatten()])
            for ind in range(subjects_table.shape[0])
        ]

        if subject_id not in subject_ids:
            print(f"Subject {subject_id} not in population")
            return None
        
        # Get the subject's index
        subject_index = subject_ids.index(subject_id)

        # Fetch pre-med and post-med start/end times
        prestart = f[f['labels'][label_map['prestart']][subject_index][0]][0, 0]
        preend = f[f['labels'][label_map['preend']][subject_index][0]][0, 0]

        if subject_id in gc.participant_ids.PRE_IDS:
            poststart, postend = np.nan, np.nan
        else:
            poststart = f[f['labels'][label_map['poststart']][subject_index][0]][0, 0]
            postend = f[f['labels'][label_map['postend']][subject_index][0]][0, 0]
        
        return prestart, preend, poststart, postend
    

def load_dataframes_directory(directory_path, subject_ids):
    dfs = []
    for subject_id in subject_ids:
        mas_path = os.path.join(directory_path, f'{subject_id}_{gc.descriptives.MOST_AFFECTED_SIDE}.parquet')
        las_path = os.path.join(directory_path, f'{subject_id}_{gc.descriptives.LEAST_AFFECTED_SIDE}.parquet')

        try:
            df_mas = pd.read_parquet(mas_path)
            df_las = pd.read_parquet(las_path)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

        df_mas['affected_side'] = gc.descriptives.MOST_AFFECTED_SIDE
        df_las['affected_side'] = gc.descriptives.LEAST_AFFECTED_SIDE

        df_subject = pd.concat([df_mas, df_las], ignore_index=True)
        df_subject[gc.columns.ID] = subject_id
        dfs.append(df_subject)
        
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()