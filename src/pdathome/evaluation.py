import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from paradigma.windowing import create_segments, categorize_segments

from sklearn.metrics import roc_curve, auc

from pdathome.constants import activity_map, arm_labels_rename, classifiers, columns, descriptives, \
    parameters, participant_ids, paths, PlotParameters, segment_map, updrs_3_map


def calculate_sens(df, pred_colname, true_colname):
    try:
        return df.loc[(df[pred_colname]==1) & (df[true_colname]==1)].shape[0] / df.loc[df[true_colname]==1].shape[0]
    except ZeroDivisionError:
        return np.nan
    
    
def calculate_spec(df, pred_colname, true_colname):
    try:
        return df.loc[(df[pred_colname]==0) & (df[true_colname]==0)].shape[0] / df.loc[df[true_colname]==0].shape[0]
    except ZeroDivisionError:
        return np.nan
    

def plot_coefs(d_coefs, classifier, color=PlotParameters().COLOR_PALETTE_FIRST_COLOR, figsize=(10,20)):
    if classifier==classifiers.LOGISTIC_REGRESSION:
        coefs = 'coefficient'
    elif classifier==classifiers.RANDOM_FOREST:
        coefs = 'impurity_score'

    df_coefs = pd.DataFrame(d_coefs.items(), columns=['feature', coefs])

    sorter = list(df_coefs.groupby('feature')[coefs].mean().sort_values(ascending=False).keys())
    df_coefs = df_coefs.set_index('feature').loc[sorter].reset_index()

    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(data=df_coefs, y='feature', x=coefs, orient='h', color=color)
    ax.set_xlabel(coefs.replace('_', ' ').capitalize())
    ax.set_ylabel("Feature")

    fig.tight_layout()

    plt.show()


def generate_clinical_scores(l_ids):
    df_patient_info = pd.read_pickle(os.path.join(paths.PATH_CLINICAL_DATA, 'df_patient_info_updrs_3.pkl'))
    df_patient_info = df_patient_info.loc[df_patient_info['record_id'].isin(participant_ids.L_PD_IDS)].reset_index(drop=True)
    df_patient_info['age'] = datetime.datetime.now().year - df_patient_info['year_of_birth']
    df_patient_info['years_since_diagnosis'] = datetime.datetime.now().year - df_patient_info['year_diagnosis']
    df_patient_info['gender'] = df_patient_info['gender'].apply(lambda x: 'male' if x==1 else 'female')

    for col in ['age', 'years_since_diagnosis']:
        df_patient_info[col] = df_patient_info[col].apply(lambda x: int(x))

    d_clinical_scores = {}

    for subject in l_ids:
        d_clinical_scores[subject] = {}
        d_clinical_scores[subject]['updrs'] = {}
        for med_stage, med_prefix in zip([descriptives.PRE_MED, descriptives.POST_MED], ['OFF', 'ON']):
            d_clinical_scores[subject]['updrs'][med_stage] = {}
            for side in ['right', 'left']:
                if subject in participant_ids.L_PD_MOST_AFFECTED_RIGHT:
                    if side == 'right':
                        affected_side = descriptives.MOST_AFFECTED_SIDE
                    else:
                        affected_side = descriptives.LEAST_AFFECTED_SIDE
                else:
                    if side == 'left':
                        affected_side = descriptives.MOST_AFFECTED_SIDE
                    else:
                        affected_side = descriptives.LEAST_AFFECTED_SIDE

                updrs_3_hypokinesia_stage_cols = [f'{med_prefix}_{x}' for x in updrs_3_map[side]['hypokinesia'].keys()]
                updrs_3_stage_cols = updrs_3_hypokinesia_stage_cols + [f'{med_prefix}_{x}' for x in updrs_3_map[side]['tremor'].keys()]
                
                d_clinical_scores[subject]['updrs'][med_stage][affected_side] = {
                    'subscore': np.sum(df_patient_info.loc[df_patient_info['record_id']==subject, updrs_3_hypokinesia_stage_cols], axis=1).values[0],
                    'total': np.sum(df_patient_info.loc[df_patient_info['record_id']==subject, updrs_3_stage_cols], axis=1).values[0]
                }

    return d_clinical_scores


def generate_results_step(step, l_pd, l_controls, segment_gap_s):

    d_output = {
        descriptives.PARKINSONS_DISEASE: {},
        descriptives.CONTROLS: {}
    }

    for subject in l_pd + l_controls:
        d_performance = {}
        
        for model in [classifiers.LOGISTIC_REGRESSION, classifiers.RANDOM_FOREST]:
            d_performance[model] = {}
            
            # thresholds
            with open(os.path.join(paths.PATH_THRESHOLDS, step, f'{model}_threshold.txt'), 'r') as f:
                clf_threshold = np.mean(float(f.read()))

            # predictions
            df_predictions = pd.read_pickle(os.path.join(paths.PATH_GAIT_PREDICTIONS, model, f'{subject}.pkl'))

            # TEMPORARY
            df_predictions = df_predictions.rename(columns={'watch_side': 'side'})

            # PREPROCESS DATA
            df_predictions.loc[df_predictions[columns.PRED_GAIT_PROBA]>=clf_threshold, columns.PRED_GAIT] = 1
            df_predictions.loc[df_predictions[columns.PRED_GAIT_PROBA]<clf_threshold, columns.PRED_GAIT] = 0

            # boolean for gait
            df_predictions.loc[df_predictions[columns.FREE_LIVING_LABEL]=='Walking', 'gait_boolean'] = 1
            df_predictions.loc[df_predictions[columns.FREE_LIVING_LABEL]!='Walking', 'gait_boolean'] = 0

            if subject in participant_ids.L_HC_IDS:
                df_predictions[columns.PRE_OR_POST] = descriptives.CONTROLS
            else:
                # boolean for arm swing
                df_predictions.loc[df_predictions[columns.ARM_LABEL]=='Gait without other behaviours or other positions', 'arm_swing_boolean'] = 1
                df_predictions.loc[df_predictions[columns.ARM_LABEL]!='Gait without other behaviours or other positions', 'arm_swing_boolean'] = 0
                df_predictions.loc[df_predictions[columns.ARM_LABEL]=='Holding an object behind ', columns.ARM_LABEL] = 'Holding an object behind'
                df_predictions[columns.ARM_LABEL] = df_predictions.loc[~df_predictions[columns.ARM_LABEL].isna(), columns.ARM_LABEL].apply(lambda x: arm_labels_rename[x])

            # PROCESS DATA

            # make segments and segment duration categories
            for affected_side in [descriptives.MOST_AFFECTED_SIDE, descriptives.LEAST_AFFECTED_SIDE]:
                df_side = df_predictions.loc[df_predictions[columns.SIDE]==affected_side]

                if subject in participant_ids.L_TREMOR_IDS:
                    df_ts = pd.read_pickle(os.path.join(paths.PATH_GAIT_FEATURES, f'{subject}_{affected_side}_ts.pkl'))

                    df_ts = df_ts.explode(column=[columns.TIME, columns.FREE_LIVING_LABEL, columns.ARM_LABEL, columns.TREMOR_LABEL])
                    df_ts = df_ts.drop_duplicates(subset=[columns.TIME, columns.FREE_LIVING_LABEL, columns.PRE_OR_POST, columns.ARM_LABEL, columns.TREMOR_LABEL])
                    df_ts = df_ts.loc[df_ts[columns.PRE_OR_POST].isin([descriptives.PRE_MED, descriptives.POST_MED])]

                    df_ts.loc[df_ts[columns.ARM_LABEL]=='Holding an object behind ', columns.ARM_LABEL] = 'Holding an object behind'
                    df_ts[columns.ARM_LABEL] = df_ts.loc[~df_ts[columns.ARM_LABEL].isna(), columns.ARM_LABEL].apply(lambda x: arm_labels_rename[x])

                fpr, tpr, _ = roc_curve(y_true=np.array(df_side['gait_boolean']), y_score=np.array(df_side[columns.PRED_GAIT_PROBA]), pos_label=1)
                roc = auc(fpr, tpr)

                d_performance[model][affected_side] = {
                    'sens': calculate_sens(df=df_side, pred_colname=columns.PRED_GAIT, true_colname='gait_boolean'),
                    'spec': calculate_spec(df=df_side, pred_colname=columns.PRED_GAIT, true_colname='gait_boolean'),
                    'auc': roc
                }

                if subject in participant_ids.L_PD_IDS and columns.PRE_OR_POST not in df_side.columns:
                    df_raw = pd.read_pickle(os.path.join(paths.PATH_DATAFRAMES, f'{subject}_{affected_side}.pkl'))
                    df_side = pd.merge(left=df_side, right=df_raw[[columns.TIME, columns.PRE_OR_POST]], how='left', on=[columns.TIME])

                for med_stage in df_side[columns.PRE_OR_POST].unique():
                    df_med_stage = df_side.loc[df_side[columns.PRE_OR_POST]==med_stage].copy()

                    fpr, tpr, _ = roc_curve(y_true=np.array(df_med_stage['gait_boolean']), y_score=np.array(df_med_stage[columns.PRED_GAIT_PROBA]), pos_label=1)
                    roc = auc(fpr, tpr)

                    d_performance[model][affected_side][med_stage] = {
                        'sens': calculate_sens(df=df_med_stage, pred_colname=columns.PRED_GAIT, true_colname='gait_boolean'),
                        'spec': calculate_spec(df=df_med_stage, pred_colname=columns.PRED_GAIT, true_colname='gait_boolean'),
                        'auc': roc,
                        'size': {
                            'gait_s': df_med_stage.loc[df_med_stage['gait_boolean']==1].shape[0] / parameters.DOWNSAMPLED_FREQUENCY,
                            'non_gait_s': df_med_stage.loc[df_med_stage['gait_boolean']==0].shape[0] / parameters.DOWNSAMPLED_FREQUENCY,
                        }
                    }

                    df_gait = df_med_stage.loc[df_med_stage[columns.FREE_LIVING_LABEL]=='Walking'].copy()

                    # df, time_column_name, gap_threshold

                    df_gait[columns.SEGMENT_NR] = create_segments(
                        df=df_gait,
                        time_column_name=columns.TIME,
                        segment_column_name=columns.SEGMENT_NR,
                        gap_threshold_s=segment_gap_s
                    )

                    df_gait[columns.SEGMENT_CAT] = categorize_segments(
                        df=df_gait,
                        segment_nr_colname=columns.SEGMENT_NR,
                        sampling_frequency=parameters.DOWNSAMPLED_FREQUENCY,
                    )

                    df_gait[columns.SEGMENT_CAT] = df_gait[columns.SEGMENT_CAT].apply(lambda x: segment_map[x])
    
                    # minutes of data per med stage, per affected side, per segment duration category
                    d_performance[model][affected_side][med_stage]['segment_duration'] = {}
                    for segment_duration in df_gait[columns.SEGMENT_CAT].unique():
                        df_segments_cat = df_gait.loc[df_gait[columns.SEGMENT_CAT]==segment_duration]

                        d_performance[model][affected_side][med_stage]['segment_duration'][segment_duration] = {
                            'sens': calculate_sens(df=df_segments_cat, pred_colname=columns.PRED_GAIT, true_colname='gait_boolean'),
                        }

                        d_performance[model][affected_side][med_stage]['segment_duration'][segment_duration]['minutes'] = df_segments_cat.shape[0]/parameters.DOWNSAMPLED_FREQUENCY/60

                        if subject in participant_ids.L_PD_IDS:
                            d_performance[model][affected_side][med_stage]['segment_duration'][segment_duration]['arm_activities'] = {}

                            for arm_label in df_segments_cat[columns.ARM_LABEL].unique():
                                df_arm_activity = df_segments_cat.loc[df_segments_cat[columns.ARM_LABEL]==arm_label]

                                d_performance[model][affected_side][med_stage]['segment_duration'][segment_duration]['arm_activities'][arm_label] = {
                                    'mins': df_arm_activity.shape[0],
                                    'sens': calculate_sens(df=df_arm_activity, pred_colname=columns.PRED_GAIT, true_colname='gait_boolean')
                                }

                    # minutes of data per activity of MAS
                    df_med_stage['label_agg'] = df_med_stage[columns.FREE_LIVING_LABEL].apply(lambda x: activity_map[x] if x in activity_map.keys() else x)
                    d_performance[model][affected_side][med_stage]['activities'] = {}

                    for activity_label in df_med_stage['label_agg'].unique():
                        df_activity = df_med_stage.loc[df_med_stage['label_agg']==activity_label]
                        d_performance[model][affected_side][med_stage]['activities'][activity_label] = {
                            'spec': calculate_spec(df=df_activity, pred_colname=columns.PRED_GAIT, true_colname='gait_boolean'),
                        }

                    # minutes of data per arm activity of MAS
                    if subject in participant_ids.L_PD_IDS:
                        d_performance[model][affected_side][med_stage]['arm_activities'] = {}

                        for arm_label in df_med_stage[columns.ARM_LABEL].unique():
                            df_arm_activity = df_med_stage.loc[df_med_stage[columns.ARM_LABEL]==arm_label]

                            d_performance[model][affected_side][med_stage]['arm_activities'][arm_label] = {
                                'mins': df_arm_activity.shape[0],
                                'sens': calculate_sens(df=df_arm_activity, pred_colname=columns.PRED_GAIT, true_colname='gait_boolean')
                            }

                    # effect of tremor on specificity
                    if subject in participant_ids.L_TREMOR_IDS:

                        df_med_stage = df_side.loc[df_side[columns.PRE_OR_POST]==med_stage].copy()

                        df_tremor = pd.merge(left=df_med_stage, right=df_ts.loc[df_ts[columns.PRE_OR_POST]==med_stage], on=[columns.TIME, columns.FREE_LIVING_LABEL, columns.PRE_OR_POST, columns.ARM_LABEL], how='left')

                        df_tremor['tremor_label_binned'] = df_tremor[columns.TREMOR_LABEL].apply(
                            lambda x: 'tremor' if x in ['Slight or mild tremor', 'Moderate tremor', 'Severe tremor', 'Tremor with significant upper limb activity'] else
                            ('no_tremor' if x in ['No tremor', 'Periodic activity of hand/arm similar frequency to tremor', 'No tremor with significant upper limb activity'] else
                            np.nan
                            )
                        )

                        for tremor_type in [x for x in df_tremor['tremor_label_binned'].unique() if not pd.isna(x)]:
                            d_performance[model][affected_side][med_stage][f'{tremor_type}_spec'] = calculate_spec(df=df_tremor.loc[df_tremor['tremor_label_binned']==tremor_type], pred_colname=columns.PRED_GAIT, true_colname='gait_boolean')

    
        if subject in participant_ids.L_PD_IDS:
            d_output[descriptives.PARKINSONS_DISEASE][subject] = d_performance
        else:
            d_output[descriptives.CONTROLS][subject] = d_performance
            
        print(f"Time {datetime.datetime.now()} - {subject} - Finished.")
            
    return d_output


def generate_results():

    d_output = {}

    d_output['clinical'] = generate_clinical_scores(participant_ids.L_PD_IDS)

    for step in ['gait', 'arm_activity']:
        d_output[step] = generate_results_step(
            step=step, 
            l_pd=participant_ids.L_PD_IDS,
            l_controls=participant_ids.L_HC_IDS,
            segment_gap_s=1.5
        )

    return d_output


