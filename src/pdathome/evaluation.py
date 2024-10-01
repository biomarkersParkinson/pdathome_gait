import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from paradigma.windowing import create_segments, categorize_segments

from scipy.stats import wilcoxon, ranksums
from sklearn.metrics import roc_curve, auc

from pdathome.constants import global_constants as gc, mappings as mp
from pdathome.quantification import compute_aggregations, compute_effect_size


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
    
def calculate_pvalue(x, y, test):
    if test == 'wilcoxon':
        return wilcoxon(x, y)
    elif test == 'ranksums':
        return ranksums(x, y)
    else:
        raise ValueError('Invalid test')
    

def plot_coefs(d_coefs, classifier, color=gc.plot_parameters.COLOR_PALETTE_FIRST_COLOR, figsize=(10,20)):
    if classifier==gc.classifiers.LOGISTIC_REGRESSION:
        coefs = 'coefficient'
    elif classifier==gc.classifiers.RANDOM_FOREST:
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


def plot_n_subjects(d_performance, x_loc, ax):
    for j, label in enumerate(d_performance.keys()):
        l_spec = len(d_performance[label])
        if l_spec == 1:
            ax.text(x_loc, j*1, f"{l_spec} subject")
        elif l_spec > 0:
            ax.text(x_loc, j*1, f"{l_spec} subjects")


def plot_significance(ax, x_min, x_max, pvalue, y_min_significance, gap, row, text_size, color='k'):
    if pvalue < 0.0001:
        asterisks = '****'
    elif pvalue < 0.001:
        asterisks = '***'
    elif pvalue < 0.01:
        asterisks = '**'
    elif pvalue < 0.05:
        asterisks = '*'
    else:
        asterisks = 'ns'

    if row == 1:
        gap = y_min_significance
    else:
        gap = (row - 1) * gap + y_min_significance
    
    bottom, top = ax.get_ylim()
    y_range = top - bottom

    bar_height = y_min_significance * 0.04 + gap
    bar_tips = bar_height - y_range * 0.01
    text_height = bar_height + y_range * 0.01

    ax.plot([x_min, x_min, x_max, x_max], [bar_tips, bar_height, bar_height, bar_tips], lw=1, c=color)
    ax.text((x_max + x_min)/2, text_height, asterisks, ha='center', va='bottom', c=color, size=text_size)


def generate_clinical_scores(l_ids):
    df_patient_info = pd.read_pickle(os.path.join(gc.paths.PATH_CLINICAL_DATA, 'df_patient_info_updrs_3.pkl'))
    df_patient_info = df_patient_info.loc[df_patient_info['record_id'].isin(gc.participant_ids.L_PD_IDS)].reset_index(drop=True)
    df_patient_info['age'] = datetime.datetime.now().year - df_patient_info['year_of_birth']
    df_patient_info['years_since_diagnosis'] = datetime.datetime.now().year - df_patient_info['year_diagnosis']
    df_patient_info['gender'] = df_patient_info['gender'].apply(lambda x: 'male' if x==1 else 'female')

    for col in ['age', 'years_since_diagnosis']:
        df_patient_info[col] = df_patient_info[col].apply(lambda x: int(x))

    d_clinical_scores = {}

    for subject in l_ids:
        d_clinical_scores[subject] = {}
        d_clinical_scores[subject]['updrs'] = {}
        for med_stage, med_prefix in zip([gc.descriptives.PRE_MED, gc.descriptives.POST_MED], ['OFF', 'ON']):
            d_clinical_scores[subject]['updrs'][med_stage] = {}
            for side in ['right', 'left']:
                if subject in gc.participant_ids.L_PD_MOST_AFFECTED_RIGHT:
                    if side == 'right':
                        affected_side = gc.descriptives.MOST_AFFECTED_SIDE
                    else:
                        affected_side = gc.descriptives.LEAST_AFFECTED_SIDE
                else:
                    if side == 'left':
                        affected_side = gc.descriptives.MOST_AFFECTED_SIDE
                    else:
                        affected_side = gc.descriptives.LEAST_AFFECTED_SIDE

                updrs_3_hypokinesia_stage_cols = [f'{med_prefix}_{x}' for x in mp.updrs_3_map[side]['hypokinesia'].keys()]
                updrs_3_stage_cols = updrs_3_hypokinesia_stage_cols + [f'{med_prefix}_{x}' for x in mp.updrs_3_map[side]['tremor'].keys()]
                
                d_clinical_scores[subject]['updrs'][med_stage][affected_side] = {
                    'subscore': np.sum(df_patient_info.loc[df_patient_info['record_id']==subject, updrs_3_hypokinesia_stage_cols], axis=1).values[0],
                    'total': np.sum(df_patient_info.loc[df_patient_info['record_id']==subject, updrs_3_stage_cols], axis=1).values[0]
                }

    return d_clinical_scores


def generate_results_classification(step, subject, segment_gap_s):

    d_performance = {}
  
    for model in [gc.classifiers.LOGISTIC_REGRESSION, gc.classifiers.RANDOM_FOREST]:
        d_performance[model] = {}

        if step == 'gait':
            # Paths
            path_predictions = gc.paths.PATH_GAIT_PREDICTIONS
            path_features = gc.paths.PATH_GAIT_FEATURES

            # Columns
            pred_proba_colname = gc.columns.PRED_GAIT_PROBA
            pred_colname = gc.columns.PRED_GAIT
            label_colname = gc.columns.FREE_LIVING_LABEL

            # Values
            value_label = 'Walking'

            # Classification threshold
            with open(os.path.join(gc.paths.PATH_THRESHOLDS, step, f'{model}.txt'), 'r') as f:
                clf_threshold = float(f.read())
        else:
            # Paths
            path_predictions = gc.paths.PATH_ARM_ACTIVITY_PREDICTIONS
            path_features = gc.paths.PATH_ARM_ACTIVITY_FEATURES

            # Columns
            pred_proba_colname = gc.columns.PRED_OTHER_ARM_ACTIVITY_PROBA
            pred_colname = gc.columns.PRED_OTHER_ARM_ACTIVITY
            label_colname = gc.columns.ARM_LABEL

            # Values
            value_label = 'Gait without other behaviours or other positions'

            # Classification threshold
            clf_threshold = 0.5
        
        # Predictions
        df_predictions = pd.read_pickle(os.path.join(path_predictions, model, f'{subject}.pkl'))

        # PREPROCESS DATA
        df_predictions.loc[df_predictions[pred_proba_colname]>=clf_threshold, pred_colname] = 1
        df_predictions.loc[df_predictions[pred_proba_colname]<clf_threshold, pred_colname] = 0

        # boolean for label (ground truth)
        if step == 'gait':
            df_predictions.loc[df_predictions[label_colname]==value_label, 'label_boolean'] = 1
            df_predictions.loc[df_predictions[label_colname]!=value_label, 'label_boolean'] = 0
        elif step == 'arm_activity':
            df_predictions.loc[df_predictions[label_colname]!=value_label, 'label_boolean'] = 1
            df_predictions.loc[df_predictions[label_colname]==value_label, 'label_boolean'] = 0

        if subject in gc.participant_ids.L_PD_IDS:
            df_predictions.loc[df_predictions[gc.columns.ARM_LABEL]=='Holding an object behind ', gc.columns.ARM_LABEL] = 'Holding an object behind'
            df_predictions[gc.columns.ARM_LABEL] = df_predictions.loc[~df_predictions[gc.columns.ARM_LABEL].isna(), gc.columns.ARM_LABEL].apply(lambda x: mp.arm_labels_rename[x])
        else:
            df_predictions[gc.columns.PRE_OR_POST] = gc.descriptives.CONTROLS
            
        # make segments and segment duration categories
        for affected_side in [gc.descriptives.MOST_AFFECTED_SIDE, gc.descriptives.LEAST_AFFECTED_SIDE]:
            df_side = df_predictions.loc[df_predictions[gc.columns.SIDE]==affected_side]

            # if subject in gc.participant_ids.L_TREMOR_IDS:

            #     if step == 'gait':
            #         l_explode_cols = [gc.columns.TIME, gc.columns.FREE_LIVING_LABEL, gc.columns.PRE_OR_POST]
            #     else:
            #         l_explode_cols = [gc.columns.TIME, gc.columns.PRE_OR_POST, gc.columns.ARM_LABEL]

            #     df_ts = pd.read_pickle(os.path.join(path_features, f'{subject}_{affected_side}_ts.pkl'))

            #     df_ts = df_ts.explode(column=[gc.columns.TIME, gc.columns.FREE_LIVING_LABEL, gc.columns.ARM_LABEL, gc.columns.TREMOR_LABEL])
            #     df_ts = df_ts.drop_duplicates(subset=[gc.columns.TIME, gc.columns.FREE_LIVING_LABEL, gc.columns.PRE_OR_POST, gc.columns.ARM_LABEL, gc.columns.TREMOR_LABEL])
            #     df_ts = df_ts.loc[df_ts[gc.columns.PRE_OR_POST].isin([gc.descriptives.PRE_MED, gc.descriptives.POST_MED])]

            #     df_ts.loc[df_ts[gc.columns.ARM_LABEL]=='Holding an object behind ', gc.columns.ARM_LABEL] = 'Holding an object behind'
            #     df_ts[gc.columns.ARM_LABEL] = df_ts.loc[~df_ts[gc.columns.ARM_LABEL].isna(), gc.columns.ARM_LABEL].apply(lambda x: mp.arm_labels_rename[x])

            fpr, tpr, _ = roc_curve(y_true=np.array(df_side['label_boolean']), y_score=np.array(df_side[pred_proba_colname]), pos_label=1)
            roc = auc(fpr, tpr)

            d_performance[model][affected_side] = {
                'sens': calculate_sens(df=df_side, pred_colname=pred_colname, true_colname='label_boolean'),
                'spec': calculate_spec(df=df_side, pred_colname=pred_colname, true_colname='label_boolean'),
                'auc': roc
            }

            l_raw_cols = [gc.columns.TIME, gc.columns.FREE_LIVING_LABEL]
            l_merge_cols = [gc.columns.TIME]
            if gc.columns.PRE_OR_POST in df_side.columns and subject in gc.participant_ids.L_PD_IDS:
                l_raw_cols.append(gc.columns.PRE_OR_POST)
                l_merge_cols.append(gc.columns.PRE_OR_POST)
            if gc.columns.FREE_LIVING_LABEL in df_side.columns:
                l_merge_cols.append(gc.columns.FREE_LIVING_LABEL)

            df_raw = pd.read_pickle(os.path.join(gc.paths.PATH_PREPARED_DATA, f'{subject}_{affected_side}.pkl'))
            if subject in gc.participant_ids.L_PD_IDS:
                df_side = pd.merge(left=df_side, right=df_raw[l_raw_cols], how='left', on=l_merge_cols)
            else:
                df_side = pd.merge(left=df_side, right=df_raw[l_raw_cols], how='left', on=l_merge_cols)

            for med_stage in df_side[gc.columns.PRE_OR_POST].unique():
                df_med_stage = df_side.loc[df_side[gc.columns.PRE_OR_POST]==med_stage].copy()

                fpr, tpr, _ = roc_curve(y_true=np.array(df_med_stage['label_boolean']), y_score=np.array(df_med_stage[pred_proba_colname]), pos_label=1)
                roc = auc(fpr, tpr)

                d_performance[model][affected_side][med_stage] = {
                    'sens': calculate_sens(df=df_med_stage, pred_colname=pred_colname, true_colname='label_boolean'),
                    'spec': calculate_spec(df=df_med_stage, pred_colname=pred_colname, true_colname='label_boolean'),
                    'auc': roc,
                    'size': {
                        'gait_s': df_med_stage.loc[df_med_stage['label_boolean']==1].shape[0] / gc.parameters.DOWNSAMPLED_FREQUENCY,
                        'non_gait_s': df_med_stage.loc[df_med_stage['label_boolean']==0].shape[0] / gc.parameters.DOWNSAMPLED_FREQUENCY,
                    }
                }

                df_gait = df_med_stage.loc[df_med_stage[gc.columns.FREE_LIVING_LABEL]=='Walking'].copy()

                # df, time_column_name, gap_threshold

                df_gait[gc.columns.SEGMENT_NR] = create_segments(
                    df=df_gait,
                    time_column_name=gc.columns.TIME,
                    gap_threshold_s=segment_gap_s
                )

                df_gait[gc.columns.SEGMENT_CAT] = categorize_segments(
                    df=df_gait,
                    segment_nr_colname=gc.columns.SEGMENT_NR,
                    sampling_frequency=gc.parameters.DOWNSAMPLED_FREQUENCY,
                )

                df_gait[gc.columns.SEGMENT_CAT] = df_gait[gc.columns.SEGMENT_CAT].apply(lambda x: mp.segment_map[x])

                # minutes of data per med stage, per affected side, per segment duration category
                d_performance[model][affected_side][med_stage]['segment_duration'] = {}
                for segment_duration in df_gait[gc.columns.SEGMENT_CAT].unique():
                    df_segments_cat = df_gait.loc[df_gait[gc.columns.SEGMENT_CAT]==segment_duration]

                    d_performance[model][affected_side][med_stage]['segment_duration'][segment_duration] = {
                        'sens': calculate_sens(df=df_segments_cat, pred_colname=pred_colname, true_colname='label_boolean'),
                    }

                    d_performance[model][affected_side][med_stage]['segment_duration'][segment_duration]['minutes'] = df_segments_cat.shape[0]/gc.parameters.DOWNSAMPLED_FREQUENCY/60

                    if subject in gc.participant_ids.L_PD_IDS:
                        d_performance[model][affected_side][med_stage]['segment_duration'][segment_duration]['arm_activities'] = {}

                        for arm_label in df_segments_cat[gc.columns.ARM_LABEL].unique():
                            df_arm_activity = df_segments_cat.loc[df_segments_cat[gc.columns.ARM_LABEL]==arm_label]

                            d_performance[model][affected_side][med_stage]['segment_duration'][segment_duration]['arm_activities'][arm_label] = {
                                'minutes': df_arm_activity.shape[0]/gc.parameters.DOWNSAMPLED_FREQUENCY/60,
                                'sens': calculate_sens(df=df_arm_activity, pred_colname=pred_colname, true_colname='label_boolean')
                            }

                # minutes of data per activity of mas
                df_med_stage['label_agg'] = df_med_stage[gc.columns.FREE_LIVING_LABEL].apply(lambda x: mp.activity_map[x] if x in mp.activity_map.keys() else x)
                d_performance[model][affected_side][med_stage]['activities'] = {}

                for activity_label in df_med_stage['label_agg'].unique():
                    df_activity = df_med_stage.loc[df_med_stage['label_agg']==activity_label]
                    d_performance[model][affected_side][med_stage]['activities'][activity_label] = {
                        'spec': calculate_spec(df=df_activity, pred_colname=pred_colname, true_colname='label_boolean'),
                    }

                # minutes of data per arm activity of mas
                if subject in gc.participant_ids.L_PD_IDS:
                    d_performance[model][affected_side][med_stage]['arm_activities'] = {}

                    for arm_label in df_med_stage[gc.columns.ARM_LABEL].unique():
                        df_arm_activity = df_med_stage.loc[df_med_stage[gc.columns.ARM_LABEL]==arm_label]

                        d_performance[model][affected_side][med_stage]['arm_activities'][arm_label] = {
                            'mins': df_arm_activity.shape[0]/gc.parameters.DOWNSAMPLED_FREQUENCY/60,
                            'sens': calculate_sens(df=df_arm_activity, pred_colname=pred_colname, true_colname='label_boolean')
                        }

                # # effect of tremor on specificity
                # if subject in gc.participant_ids.L_TREMOR_IDS:

                #     df_med_stage = df_side.loc[df_side[gc.columns.PRE_OR_POST]==med_stage].copy()

                #     df_tremor = pd.merge(left=df_med_stage, right=df_ts.loc[df_ts[gc.columns.PRE_OR_POST]==med_stage], on=[gc.columns.TIME, gc.columns.FREE_LIVING_LABEL, gc.columns.PRE_OR_POST, gc.columns.ARM_LABEL], how='left')

                #     df_tremor['tremor_label_binned'] = df_tremor[gc.columns.TREMOR_LABEL].apply(
                #         lambda x: 'tremor' if x in ['Slight or mild tremor', 'Moderate tremor', 'Severe tremor', 'Tremor with significant upper limb activity'] else
                #         ('no_tremor' if x in ['No tremor', 'Periodic activity of hand/arm similar frequency to tremor', 'No tremor with significant upper limb activity'] else
                #         np.nan
                #         )
                #     )

                #     for tremor_type in [x for x in df_tremor['tremor_label_binned'].unique() if not pd.isna(x)]:
                #         d_performance[model][affected_side][med_stage][f'{tremor_type}_spec'] = calculate_spec(df=df_tremor.loc[df_tremor['tremor_label_binned']==tremor_type], pred_colname=pred_colname, true_colname='label_boolean')
                        
    return d_performance


def load_arm_activity_features(subject: str, side: str) -> pd.DataFrame:
    """Load arm activity features for a given subject and side."""
    df_features = pd.read_pickle(os.path.join(gc.paths.PATH_ARM_ACTIVITY_FEATURES, f'{subject}_{side}.pkl'))
    df_features[gc.columns.SIDE] = side
    return df_features

def load_arm_activity_timestamps(subject: str, side: str) -> pd.DataFrame:
    """Load arm activity timestamps for a given subject and side."""
    df_ts =  pd.read_pickle(os.path.join(gc.paths.PATH_ARM_ACTIVITY_FEATURES, f'{subject}_{side}_ts.pkl'))
    df_ts[gc.columns.SIDE] = side
    return df_ts


def generate_results_quantification(subject: str) -> tuple[dict, pd.DataFrame]:
    """Generate quantification results for a given subject."""
    classification_threshold = 0.5

    # Load arm activity features for both sides
    df_features_mas = load_arm_activity_features(subject, gc.descriptives.MOST_AFFECTED_SIDE)
    df_features_las = load_arm_activity_features(subject, gc.descriptives.LEAST_AFFECTED_SIDE)

    # Combine features and preprocess
    df_features = pd.concat([df_features_mas, df_features_las], axis=0).reset_index(drop=True)
    df_features['peak_velocity'] = (df_features['forward_peak_ang_vel_mean'] + df_features['backward_peak_ang_vel_mean']) / 2
    df_features = df_features.drop(columns=[gc.columns.TIME])

    # Load timestamps for both sides
    df_ts_mas = load_arm_activity_timestamps(subject, gc.descriptives.MOST_AFFECTED_SIDE)
    df_ts_las = load_arm_activity_timestamps(subject, gc.descriptives.LEAST_AFFECTED_SIDE)

    df_ts = pd.concat([df_ts_mas, df_ts_las], axis=0).reset_index(drop=True)

    # Explode the timestamps DataFrame
    df_ts_exploded = df_ts.explode([
        gc.columns.TIME, gc.columns.ARM_LABEL, gc.columns.TRUE_GAIT_SEGMENT_NR,
        gc.columns.TRUE_GAIT_SEGMENT_CAT, gc.columns.FREE_LIVING_LABEL
        ]
    )
    
    # Merge features with exploded timestamps
    df_features = pd.merge(
        left=df_features, 
        right=df_ts_exploded, 
        how='right', 
        on=[
            gc.columns.SIDE, gc.columns.PRE_OR_POST, gc.columns.PRED_GAIT_SEGMENT_NR,
            gc.columns.WINDOW_NR
        ]
    )
    
    # Group by relevant columns and compute mean for features
    df_features = df_features.groupby([
        gc.columns.SIDE, gc.columns.TIME, gc.columns.PRED_GAIT_SEGMENT_NR,
        gc.columns.PRED_GAIT_SEGMENT_CAT, gc.columns.TRUE_GAIT_SEGMENT_CAT,
        gc.columns.PRE_OR_POST
        ]
    )[['peak_velocity', 'range_of_motion']].mean().reset_index()

    # Load arm activity predictions
    df_predictions = pd.read_pickle(os.path.join(gc.paths.PATH_ARM_ACTIVITY_PREDICTIONS, gc.classifiers.LOGISTIC_REGRESSION, f'{subject}.pkl'))

    # Set pred rounded based on the threshold
    df_predictions[gc.columns.PRED_OTHER_ARM_ACTIVITY] = (df_predictions[gc.columns.PRED_OTHER_ARM_ACTIVITY_PROBA] >= classification_threshold).astype(int)

    # Set other arm activity boolean
    df_predictions.loc[df_predictions[gc.columns.ARM_LABEL]=='Gait without other behaviours or other positions', 'other_arm_activity_boolean'] = 0
    df_predictions.loc[df_predictions[gc.columns.ARM_LABEL]!='Gait without other behaviours or other positions', 'other_arm_activity_boolean'] = 1
    df_predictions.loc[df_predictions[gc.columns.ARM_LABEL]=='Holding an object behind ', gc.columns.ARM_LABEL] = 'Holding an object behind'
    df_predictions[gc.columns.ARM_LABEL] = df_predictions.loc[~df_predictions[gc.columns.ARM_LABEL].isna(), gc.columns.ARM_LABEL].apply(lambda x: mp.arm_labels_rename[x])

    # Merge predictions with features
    df = pd.merge(
        left=df_predictions, 
        right=df_features, 
        how='left', 
        on=[
            gc.columns.TIME, gc.columns.PRE_OR_POST, gc.columns.SIDE, 
            gc.columns.PRED_GAIT_SEGMENT_NR, gc.columns.PRED_GAIT_SEGMENT_CAT
        ]
    )

    d_quantification = {}

    # Compute unfiltered gait aggregations
    d_quantification['unfiltered_gait'] = compute_aggregations(df)

    # Compute filtered gait aggregations
    d_quantification['filtered_gait'] = compute_aggregations(df.loc[df[gc.columns.PRED_OTHER_ARM_ACTIVITY] == 0])

    # Compute annotated no other arm activity for specific participants
    if subject in gc.participant_ids.L_PD_IDS:
        df_diff = pd.DataFrame()
        d_quantification['true_no_other_arm_activity'] = compute_aggregations(df.loc[df['other_arm_activity_boolean'] == 0])

        es_mrom, diff_mrom = compute_effect_size(df, 'range_of_motion', 'median')
        es_prom, diff_prom = compute_effect_size(df, 'range_of_motion', '95')

        d_quantification['effect_size'] = {
            'median_rom': es_mrom,
            '95p_rom': es_prom
        }

        for dataset in diff_mrom.keys():
            df_diff = pd.concat([df_diff, pd.DataFrame([subject, dataset, diff_mrom[dataset], diff_prom[dataset]]).T])

        return d_quantification, df_diff
    else:
        return d_quantification


def generate_results(subject, step):
    if step not in ['gait', 'arm_activity', 'quantification']:
        raise ValueError(f"Invalid step: {step}")

    if subject in gc.participant_ids.L_HC_IDS and step == 'arm_activity':
        return None

    print(f"Processing {subject} - {step}...")
    
    if step in ['gait', 'arm_activity']:
        d_output =  generate_results_classification(
            step=step, 
            subject=subject,
            segment_gap_s=1.5
        )
        return d_output

    else:
        # Only run generate_results_quantification if subject is in L_PD_IDS
        if subject in gc.participant_ids.L_PD_IDS:
            d_output, df_diff = generate_results_quantification(subject)
            return d_output, df_diff

        # Return None or an empty dictionary if the subject is not in L_PD_IDS
        return None