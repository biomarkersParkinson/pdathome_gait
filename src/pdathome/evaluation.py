import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from paradigma.windowing import create_segments, categorize_segments

from sklearn.metrics import roc_curve, auc

from pdathome.constants import global_constants as gc, mappings as mp


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

    if step == 'gait':
        # paths
        path_predictions = gc.paths.PATH_GAIT_PREDICTIONS

        # columns
        pred_proba_colname = gc.columns.PRED_GAIT_PROBA
        pred_colname = gc.columns.PRED_GAIT

        # classification threshold
        with open(os.path.join(gc.paths.PATH_THRESHOLDS, step, f'{model}_threshold.txt'), 'r') as f:
            clf_threshold = np.mean(float(f.read()))
    else:
        # paths
        path_predictions = gc.paths.PATH_ARM_ACTIVITY_PREDICTIONS

        # columns
        pred_proba_colname = gc.columns.PRED_OTHER_ARM_ACTIVITY_PROBA
        pred_colname = gc.columns.PRED_OTHER_ARM_ACTIVITY

        # classification threshold
        clf_threshold = 0.5
    
    for model in [gc.classifiers.LOGISTIC_REGRESSION, gc.classifiers.RANDOM_FOREST]:
        d_performance[model] = {}
        
        # predictions
        df_predictions = pd.read_pickle(os.path.join(path_predictions, model, f'{subject}.pkl'))

        # PREPROCESS DATA
        df_predictions.loc[df_predictions[pred_proba_colname]>=clf_threshold, pred_colname] = 1
        df_predictions.loc[df_predictions[pred_proba_colname]<clf_threshold, pred_colname] = 0

        # boolean for gait
        df_predictions.loc[df_predictions[gc.columns.FREE_LIVING_LABEL]=='Walking', 'gait_boolean'] = 1
        df_predictions.loc[df_predictions[gc.columns.FREE_LIVING_LABEL]!='Walking', 'gait_boolean'] = 0

        if subject in gc.participant_ids.L_HC_IDS:
            df_predictions[gc.columns.PRE_OR_POST] = gc.descriptives.CONTROLS
        else:
            # boolean for arm swing
            df_predictions.loc[df_predictions[gc.columns.ARM_LABEL]=='Gait without other behaviours or other positions', 'arm_swing_boolean'] = 1
            df_predictions.loc[df_predictions[gc.columns.ARM_LABEL]!='Gait without other behaviours or other positions', 'arm_swing_boolean'] = 0
            df_predictions.loc[df_predictions[gc.columns.ARM_LABEL]=='Holding an object behind ', gc.columns.ARM_LABEL] = 'Holding an object behind'
            df_predictions[gc.columns.ARM_LABEL] = df_predictions.loc[~df_predictions[gc.columns.ARM_LABEL].isna(), gc.columns.ARM_LABEL].apply(lambda x: mp.arm_labels_rename[x])

        # PROCESS DATA

        # make segments and segment duration categories
        for affected_side in [gc.descriptives.MOST_AFFECTED_SIDE, gc.descriptives.LEAST_AFFECTED_SIDE]:
            df_side = df_predictions.loc[df_predictions[gc.columns.SIDE]==affected_side]

            if subject in gc.participant_ids.L_TREMOR_IDS:
                df_ts = pd.read_pickle(os.path.join(gc.paths.PATH_GAIT_FEATURES, f'{subject}_{affected_side}_ts.pkl'))

                df_ts = df_ts.explode(column=[gc.columns.TIME, gc.columns.FREE_LIVING_LABEL, gc.columns.ARM_LABEL, gc.columns.TREMOR_LABEL])
                df_ts = df_ts.drop_duplicates(subset=[gc.columns.TIME, gc.columns.FREE_LIVING_LABEL, gc.columns.PRE_OR_POST, gc.columns.ARM_LABEL, gc.columns.TREMOR_LABEL])
                df_ts = df_ts.loc[df_ts[gc.columns.PRE_OR_POST].isin([gc.descriptives.PRE_MED, gc.descriptives.POST_MED])]

                df_ts.loc[df_ts[gc.columns.ARM_LABEL]=='Holding an object behind ', gc.columns.ARM_LABEL] = 'Holding an object behind'
                df_ts[gc.columns.ARM_LABEL] = df_ts.loc[~df_ts[gc.columns.ARM_LABEL].isna(), gc.columns.ARM_LABEL].apply(lambda x: mp.arm_labels_rename[x])

            fpr, tpr, _ = roc_curve(y_true=np.array(df_side['gait_boolean']), y_score=np.array(df_side[gc.columns.PRED_GAIT_PROBA]), pos_label=1)
            roc = auc(fpr, tpr)

            d_performance[model][affected_side] = {
                'sens': calculate_sens(df=df_side, pred_colname=gc.columns.PRED_GAIT, true_colname='gait_boolean'),
                'spec': calculate_spec(df=df_side, pred_colname=gc.columns.PRED_GAIT, true_colname='gait_boolean'),
                'auc': roc
            }

            if subject in gc.participant_ids.L_PD_IDS and gc.columns.PRE_OR_POST not in df_side.columns:
                df_raw = pd.read_pickle(os.path.join(gc.paths.PATH_DATAFRAMES, f'{subject}_{affected_side}.pkl'))
                df_side = pd.merge(left=df_side, right=df_raw[[gc.columns.TIME, gc.columns.PRE_OR_POST]], how='left', on=[gc.columns.TIME])

            for med_stage in df_side[gc.columns.PRE_OR_POST].unique():
                df_med_stage = df_side.loc[df_side[gc.columns.PRE_OR_POST]==med_stage].copy()

                fpr, tpr, _ = roc_curve(y_true=np.array(df_med_stage['gait_boolean']), y_score=np.array(df_med_stage[gc.columns.PRED_GAIT_PROBA]), pos_label=1)
                roc = auc(fpr, tpr)

                d_performance[model][affected_side][med_stage] = {
                    'sens': calculate_sens(df=df_med_stage, pred_colname=gc.columns.PRED_GAIT, true_colname='gait_boolean'),
                    'spec': calculate_spec(df=df_med_stage, pred_colname=gc.columns.PRED_GAIT, true_colname='gait_boolean'),
                    'auc': roc,
                    'size': {
                        'gait_s': df_med_stage.loc[df_med_stage['gait_boolean']==1].shape[0] / gc.parameters.DOWNSAMPLED_FREQUENCY,
                        'non_gait_s': df_med_stage.loc[df_med_stage['gait_boolean']==0].shape[0] / gc.parameters.DOWNSAMPLED_FREQUENCY,
                    }
                }

                df_gait = df_med_stage.loc[df_med_stage[gc.columns.FREE_LIVING_LABEL]=='Walking'].copy()

                # df, time_column_name, gap_threshold

                df_gait[gc.columns.SEGMENT_NR] = create_segments(
                    df=df_gait,
                    time_column_name=gc.columns.TIME,
                    segment_column_name=gc.columns.SEGMENT_NR,
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
                        'sens': calculate_sens(df=df_segments_cat, pred_colname=gc.columns.PRED_GAIT, true_colname='gait_boolean'),
                    }

                    d_performance[model][affected_side][med_stage]['segment_duration'][segment_duration]['minutes'] = df_segments_cat.shape[0]/gc.parameters.DOWNSAMPLED_FREQUENCY/60

                    if subject in gc.participant_ids.L_PD_IDS:
                        d_performance[model][affected_side][med_stage]['segment_duration'][segment_duration]['arm_activities'] = {}

                        for arm_label in df_segments_cat[gc.columns.ARM_LABEL].unique():
                            df_arm_activity = df_segments_cat.loc[df_segments_cat[gc.columns.ARM_LABEL]==arm_label]

                            d_performance[model][affected_side][med_stage]['segment_duration'][segment_duration]['arm_activities'][arm_label] = {
                                'mins': df_arm_activity.shape[0],
                                'sens': calculate_sens(df=df_arm_activity, pred_colname=gc.columns.PRED_GAIT, true_colname='gait_boolean')
                            }

                # minutes of data per activity of mas
                df_med_stage['label_agg'] = df_med_stage[gc.columns.FREE_LIVING_LABEL].apply(lambda x: mp.activity_map[x] if x in mp.activity_map.keys() else x)
                d_performance[model][affected_side][med_stage]['activities'] = {}

                for activity_label in df_med_stage['label_agg'].unique():
                    df_activity = df_med_stage.loc[df_med_stage['label_agg']==activity_label]
                    d_performance[model][affected_side][med_stage]['activities'][activity_label] = {
                        'spec': calculate_spec(df=df_activity, pred_colname=gc.columns.PRED_GAIT, true_colname='gait_boolean'),
                    }

                # minutes of data per arm activity of mas
                if subject in gc.participant_ids.L_PD_IDS:
                    d_performance[model][affected_side][med_stage]['arm_activities'] = {}

                    for arm_label in df_med_stage[gc.columns.ARM_LABEL].unique():
                        df_arm_activity = df_med_stage.loc[df_med_stage[gc.columns.ARM_LABEL]==arm_label]

                        d_performance[model][affected_side][med_stage]['arm_activities'][arm_label] = {
                            'mins': df_arm_activity.shape[0],
                            'sens': calculate_sens(df=df_arm_activity, pred_colname=gc.columns.PRED_GAIT, true_colname='gait_boolean')
                        }

                # effect of tremor on specificity
                if subject in gc.participant_ids.L_TREMOR_IDS:

                    df_med_stage = df_side.loc[df_side[gc.columns.PRE_OR_POST]==med_stage].copy()

                    df_tremor = pd.merge(left=df_med_stage, right=df_ts.loc[df_ts[gc.columns.PRE_OR_POST]==med_stage], on=[gc.columns.TIME, gc.columns.FREE_LIVING_LABEL, gc.columns.PRE_OR_POST, gc.columns.ARM_LABEL], how='left')

                    df_tremor['tremor_label_binned'] = df_tremor[gc.columns.TREMOR_LABEL].apply(
                        lambda x: 'tremor' if x in ['Slight or mild tremor', 'Moderate tremor', 'Severe tremor', 'Tremor with significant upper limb activity'] else
                        ('no_tremor' if x in ['No tremor', 'Periodic activity of hand/arm similar frequency to tremor', 'No tremor with significant upper limb activity'] else
                        np.nan
                        )
                    )

                    for tremor_type in [x for x in df_tremor['tremor_label_binned'].unique() if not pd.isna(x)]:
                        d_performance[model][affected_side][med_stage][f'{tremor_type}_spec'] = calculate_spec(df=df_tremor.loc[df_tremor['tremor_label_binned']==tremor_type], pred_colname=gc.columns.PRED_GAIT, true_colname='gait_boolean')
                        
    return d_performance


def extract_features(df, l_groupby, l_metrics, l_aggregates, l_quantiles=[]):

    # seperate quantiles from other aggregates
    l_df_agg = []
    for metric in l_metrics:
        df_agg = df.groupby(l_groupby)[metric].agg(l_aggregates).reset_index().rename(columns={x: f'{metric}_{x}' for x in l_aggregates})
        df_qs = df.groupby(l_groupby)[metric].quantile(l_quantiles).reset_index()

        for quantile in l_quantiles:
            df_agg[f"{metric}_quantile_{int(quantile*100)}"] = df_qs.loc[df_qs[f'level_{len(l_groupby)}']==quantile, metric].reset_index(drop=True) 

        l_df_agg.append(df_agg)

    for j in range(len(l_df_agg)):
        if j == 0:
            df_agg = l_df_agg[j]
        else:
            df_agg = pd.merge(left=df_agg, right=l_df_agg[j], how='left', on=l_groupby)

    return df_agg


def compute_aggregations(df, l_measures):

    l_metrics = ['range_of_motion', 'peak_velocity']
    l_aggregates = ['median']
    l_quantiles = [0.95]

    l_groupby = [gc.columns.PRE_OR_POST]
    l_groupby_side = l_groupby + ['side']
    l_groupby_segments = l_groupby_side + [gc.columns.PRED_GAIT_SEGMENT_CAT]

    df_agg_side = extract_features(df, l_groupby_side, l_metrics, l_aggregates, l_quantiles)

    d_quant = {}

    for med_stage in df_agg_side[gc.columns.PRE_OR_POST].unique():
        d_quant[med_stage] = {}
        for affected_side in [gc.descriptives.MOST_AFFECTED_SIDE, gc.descriptives.LEAST_AFFECTED_SIDE]:
            d_quant[med_stage][affected_side] = {
                'seconds': {
                    'overall': df.loc[(df[gc.columns.PRE_OR_POST]==med_stage) & (df['side']==affected_side)].shape[0] / gc.parameters.DOWNSAMPLED_FREQUENCY,
                },
                'values': {}
            }
            for measure in l_measures:
                d_quant[med_stage][affected_side]['values'][measure] = {
                    'overall': df_agg_side.loc[(df_agg_side[gc.columns.PRE_OR_POST]==med_stage) & (df_agg_side['side']==affected_side), measure].values[0]
                }

    df_agg_side = extract_features(df, l_groupby_segments, l_metrics, l_aggregates, l_quantiles)

    for med_stage in df_agg_side[gc.columns.PRE_OR_POST].unique():
        df_med_stage = df_agg_side.loc[df_agg_side[gc.columns.PRE_OR_POST]==med_stage]
        for affected_side in df_med_stage['side'].unique():
            df_aff = df_med_stage.loc[df_med_stage['side']==affected_side]
            for seq_duration_category in df_aff['sequence_duration_category'].unique():
                df_seq_duration = df_aff.loc[df_aff['sequence_duration_category']==seq_duration_category].copy().reset_index()
                d_quant[med_stage][affected_side]['seconds'][seq_duration_category] = df.loc[(df[gc.columns.PRE_OR_POST]==med_stage) & (df['side']==affected_side) & (df['sequence_duration_category']==seq_duration_category)].shape[0] / gc.parameters.DOWNSAMPLED_FREQUENCY
                for measure in l_measures:
                    d_quant[med_stage][affected_side]['values'][measure][seq_duration_category] = df_seq_duration[measure].values[0]


            d_quant[med_stage][affected_side]['seconds']['non_gait'] = d_quant[med_stage][affected_side]['seconds']['overall']  - np.sum([d_quant[med_stage][affected_side]['seconds'][seq_duration_category] for seq_duration_category in mp.segment_map.keys() if seq_duration_category in d_quant[med_stage][affected_side]['seconds'].keys()])

    return d_quant


def compute_effect_size(df, subject): 
    df['dataset'] = 'Predicted gait'

    df[gc.columns.PRED_GAIT_SEGMENT_CAT] = categorize_segments(
        df=df,
        segment_nr_colname=gc.columns.SEGMENT_NR_TRUE_GAIT,
        sampling_frequency=100
    )
    # df, segment_nr_colname, sampling_frequency

    df_pred_mas = df.loc[(df['pred_other_arm_activity']==0)].copy()
    df_pred_mas['dataset'] = 'Predicted gait predicted NOAA'

    df_ann_mas = df.loc[(df['other_arm_activity_boolean']==0)].copy()
    df_ann_mas['dataset'] = 'Predicted gait annotated NOAA'

    df_mas = pd.concat([df, df_pred_mas], axis=0).reset_index(drop=True)
    df_mas = pd.concat([df_mas, df_ann_mas], axis=0).reset_index(drop=True)

    # effect size: verschil (point estimate) gedeeld door de std van het verschil (sd van point estimate van bootstrapping)
    measure = 'range_of_motion'

    d_effect_size = {}
    d_diffs = {}

    for stat in ['median', '95']:
        d_effect_size[stat] = {}
        for dataset in df_mas['dataset'].unique():
            d_effect_size[stat][dataset] = {}

            for segment_category in ['short', 'moderately_long', 'long', 'very_long', 'overall']:
                df_pre = df_mas.loc[(df_mas['dataset']==dataset) & (df_mas[gc.columns.PRE_OR_POST]==gc.descriptives.PRE_MED)]
                df_post = df_mas.loc[(df_mas['dataset']==dataset) & (df_mas[gc.columns.PRE_OR_POST]==gc.descriptives.POST_MED)]

                if segment_category != 'overall':
                    df_pre = df_pre.loc[df_pre[gc.columns.PRED_GAIT_SEGMENT_CAT]==segment_category]
                    df_post = df_post.loc[df_post[gc.columns.PRED_GAIT_SEGMENT_CAT]==segment_category]

                range_of_motion_pre_vals = df_pre[measure].values
                range_of_motion_post_vals = df_post[measure].values
                
                if len(range_of_motion_pre_vals) != 0 and len(range_of_motion_post_vals) != 0:
                    d_effect_size[stat][dataset][segment_category] = {}
                    
                    # point estimate (median) of pre-med and post-med for the true sample
                    if stat == 'median':
                        d_effect_size[stat][dataset][segment_category]['mu_pre'] = np.median(range_of_motion_pre_vals)
                        d_effect_size[stat][dataset][segment_category]['mu_post'] = np.median(range_of_motion_post_vals)
                    elif stat == '95':
                        d_effect_size[stat][dataset][segment_category]['mu_pre'] = np.percentile(range_of_motion_pre_vals, 95)
                        d_effect_size[stat][dataset][segment_category]['mu_post'] = np.percentile(range_of_motion_post_vals, 95)

                    # boostrapping
                    bootstrap_pre = np.random.choice(range_of_motion_pre_vals, size=(5000, len(range_of_motion_pre_vals)), replace=True)
                    bootstrap_post = np.random.choice(range_of_motion_post_vals, size=(5000, len(range_of_motion_post_vals)), replace=True)

                    # point estimate using bootstrapping samples
                    if stat == 'median': 
                        bootstrap_samples_pre = np.median(bootstrap_pre, axis=1)
                        bootstrap_samples_post = np.median(bootstrap_post, axis=1)
                    elif stat == '95':
                        bootstrap_samples_pre = np.percentile(bootstrap_pre, 95, axis=1)
                        bootstrap_samples_post = np.percentile(bootstrap_post, 95, axis=1)
                        
                    # compute difference for std
                    bootstrap_samples_diff = bootstrap_samples_post - bootstrap_samples_pre

                    # compute the std
                    std_bootstrap = np.std(bootstrap_samples_diff)
                    d_effect_size[stat][dataset][segment_category]['std'] = std_bootstrap

                    if segment_category == 'overall':
                        d_diffs[dataset] = bootstrap_samples_diff

    return d_effect_size, d_diffs


def generate_results_quantification():
    # arm activity features
    df_features = pd.read_pickle(os.path.join(gc.paths.PATH_ARM_ACTIVITY_FEATURES, f'{subject}_mas.pkl'))
    df_ts = pd.read_pickle(os.path.join(gc.paths.PATH_ARM_ACTIVITY_FEATURES, f'{subject}_mas_ts.pkl'))

    df_features['peak_velocity'] = (df_features['forward_peak_ang_vel_mean'] + df_features['backward_peak_ang_vel_mean'])/2
    df_features = df_features.drop(columns=[gc.columns.TIME])

    df_ts_exploded = df_ts.explode([gc.columns.TIME, gc.columns.ARM_LABEL])

    df_features = pd.merge(left=df_features, right=df_ts_exploded, how='right', on=[gc.columns.PRE_OR_POST, gc.columns.SEGMENT_NR_TRUE_GAIT, 'window_nr'])
    df_features = df_features.groupby([gc.columns.TIME, gc.columns.PRE_OR_POST])[['peak_velocity', 'range_of_motion']].mean().reset_index()

    # arm activity predictions
    df_predictions = pd.read_pickle(os.path.join(gc.paths.PATH_ARM_ACTIVITY_PREDICTIONS, gc.classifiers.LOGISTIC_REGRESSION, f'{subject}.pkl'))
    df_predictions = df_predictions.loc[df_predictions['side']==gc.descriptives.MOST_AFFECTED_SIDE]

    # set pred rounded
    df_predictions['pred_other_arm_activity'] = (df_predictions['pred_other_arm_activity_proba'] >= 0.5).astype(int)

    df_predictions.loc[df_predictions[gc.columns.ARM_LABEL]=='Gait without other behaviours or other positions', 'other_arm_activity_boolean'] = 0
    df_predictions.loc[df_predictions[gc.columns.ARM_LABEL]!='Gait without other behaviours or other positions', 'other_arm_activity_boolean'] = 1
    df_predictions.loc[df_predictions[gc.columns.ARM_LABEL]=='Holding an object behind ', gc.columns.ARM_LABEL] = 'Holding an object behind'
    df_predictions[gc.columns.ARM_LABEL] = df_predictions.loc[~df_predictions[gc.columns.ARM_LABEL].isna(), gc.columns.ARM_LABEL].apply(lambda x: mp.arm_labels_rename[x])

    df = pd.merge(left=df, right=df_features, how='left', on=[gc.columns.TIME, gc.columns.PRE_OR_POST])
    return


def generate_results(l_steps):
    d_clinical_scores = generate_clinical_scores(gc.participant_ids.L_PD_IDS)

    d_output = {}

    for subject in gc.participant_ids.L_PD_IDS + gc.participant_ids.L_HC_IDS:
        for step in ['gait', 'arm_activity']:
            if step in l_steps:
                d_output[step] = generate_results_classification(
                    step=step, 
                    subject=subject,
                    segment_gap_s=1.5
                )

        if 'quantification' in l_steps:
            d_output['quantification'] = generate_results_quantification()

        return d_output, d_clinical_scores


