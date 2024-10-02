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


def hue_rule(df1, df2, hue=None):
    if hue is None:
        return None 
    if hue not in df1.columns or hue not in df2.columns:
        raise ValueError(f"The 'hue' variable '{hue}' is not present in both DataFrames.")

    # Filter only the rows where the 'hue' values are identical in both DataFrames
    common_values = df1[df1[hue].eq(df2[hue])][hue].unique()

    # Create a dictionary that assigns each unique value in 'hue' a color
    color_dict = {val: sns.color_palette("colorblind")[i] for i, val in enumerate(common_values)}

    # Map 'hue' values to colors using the dictionary
    colors = df1[hue].map(color_dict)

    return colors


def bland_altman_plot(df1, df2, *args, x=None, axs=None, hue='population', reference=None, confidence_interval=1.96, **kwargs):
    if axs is None:
        axs = plt.axes()

    data1     = df1[x]
    data2     = df2[x]
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    CI_low    = md - confidence_interval*sd
    CI_high   = md + confidence_interval*sd

    colors = hue_rule(df1, df2, hue=hue)
    
    if reference is None:
        sns.scatterplot(x=mean, y=diff, hue=df1[hue], *args, **kwargs)
    else:
        sns.scatterplot(x=reference, y=diff, hue=df1[hue], *args, **kwargs)

    axs.axhline(md, color='black', linestyle='-')
    axs.axhline(md + confidence_interval*sd, color='gray', linestyle='--')
    axs.axhline(md - confidence_interval*sd, color='gray', linestyle='--')

    xOutPlot = np.min(mean) + (np.max(mean)-np.min(mean))*1.34

    axs.text(xOutPlot, md - confidence_interval*sd, r'-'+str(confidence_interval)+' SD:' + "\n" + "%.2f" % CI_low, ha = "center", va = "center")
    axs.text(xOutPlot, md + confidence_interval*sd, r'+'+str(confidence_interval)+' SD:' + "\n" + "%.2f" % CI_high, ha = "center", va = "center")
    axs.text(xOutPlot, md, r'Mean:' + "\n" + "%.2f" % md, ha = "center", va = "center")

    axs.set_ylim(md - 3.5*sd, md + 3.5*sd)
    
    return axs, colors


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

            # Columns
            pred_proba_colname = gc.columns.PRED_GAIT_PROBA
            pred_colname = gc.columns.PRED_GAIT
            label_colname = gc.columns.FREE_LIVING_LABEL

            # Labels
            boolean_colname = 'gait'
            value_label = 'Walking'

            # Classification threshold
            with open(os.path.join(gc.paths.PATH_THRESHOLDS, step, f'{model}.txt'), 'r') as f:
                clf_threshold = float(f.read())
        else:
            # Paths
            path_predictions = gc.paths.PATH_ARM_ACTIVITY_PREDICTIONS

            # Columns
            pred_proba_colname = gc.columns.PRED_OTHER_ARM_ACTIVITY_PROBA
            pred_colname = gc.columns.PRED_OTHER_ARM_ACTIVITY
            label_colname = gc.columns.ARM_LABEL

            # Labels
            boolean_colname = 'other_arm_activity'
            value_label = 'Gait without other behaviours or other positions'

            # Classification threshold
            clf_threshold = 0.5

        l_raw_cols = [gc.columns.TIME, gc.columns.SIDE, gc.columns.FREE_LIVING_LABEL, 
                      gc.columns.TRUE_GAIT_SEGMENT_NR, gc.columns.TRUE_GAIT_SEGMENT_CAT]
        if subject in gc.participant_ids.L_PD_IDS:
            l_raw_cols.append(gc.columns.ARM_LABEL)
        
        # Predictions
        df_predictions = pd.read_pickle(os.path.join(path_predictions, model, f'{subject}.pkl'))
        
        # Load raw data
        l_dfs = []
        for side in [gc.descriptives.MOST_AFFECTED_SIDE, gc.descriptives.LEAST_AFFECTED_SIDE]:
            df_raw = pd.read_pickle(os.path.join(gc.paths.PATH_PREPARED_DATA, f'{subject}_{affected_side}.pkl')).assign(side=side)

            walking_segments = create_segments(
                df=df_raw.loc[df_raw[gc.columns.FREE_LIVING_LABEL] == 'Walking'],
                time_column_name=gc.columns.TIME,
                gap_threshold_s=segment_gap_s
            )

            df_raw.loc[df_raw[gc.columns.FREE_LIVING_LABEL] == 'Walking', gc.columns.TRUE_GAIT_SEGMENT_NR] = walking_segments
            df_raw[gc.columns.TRUE_GAIT_SEGMENT_NR] = df_raw[gc.columns.TRUE_GAIT_SEGMENT_NR].fillna(-1)

            # Map categories to segments of true gait
            df_raw[gc.columns.TRUE_GAIT_SEGMENT_CAT] = categorize_segments(
                df=df_raw,
                segment_nr_colname=gc.columns.TRUE_GAIT_SEGMENT_NR,
                sampling_frequency=gc.parameters.DOWNSAMPLED_FREQUENCY
            )

            df_raw[gc.columns.TRUE_GAIT_SEGMENT_CAT] = df_raw[gc.columns.TRUE_GAIT_SEGMENT_CAT].apply(
                    lambda x: mp.segment_map[x]
                )

            l_dfs.append(df_raw)

        df_raw = pd.concat(l_dfs, axis=0)

        # Merge labels
        df = pd.merge(left=df_predictions, right=df_raw[l_raw_cols], how='left', on=[gc.columns.TIME, gc.columns.SIDE])

        # PREPROCESS DATA
        df.loc[df[pred_proba_colname]>=clf_threshold, pred_colname] = 1
        df.loc[df[pred_proba_colname]<clf_threshold, pred_colname] = 0

        # boolean for label (ground truth)
        if step == 'gait':
            df.loc[df[label_colname]==value_label, boolean_colname] = 1
            df.loc[df[label_colname]!=value_label, boolean_colname] = 0
        elif step == 'arm_activity':
            df.loc[df[label_colname]!=value_label, boolean_colname] = 1
            df.loc[df[label_colname]==value_label, boolean_colname] = 0

        if subject in gc.participant_ids.L_PD_IDS:
            df.loc[df[gc.columns.ARM_LABEL]=='Holding an object behind ', gc.columns.ARM_LABEL] = 'Holding an object behind'
            df[gc.columns.ARM_LABEL] = df.loc[~df[gc.columns.ARM_LABEL].isna(), gc.columns.ARM_LABEL].apply(lambda x: mp.arm_labels_rename[x])
        else:
            df[gc.columns.PRE_OR_POST] = gc.descriptives.CONTROLS
            
        # make segments and segment duration categories
        for affected_side in [gc.descriptives.MOST_AFFECTED_SIDE, gc.descriptives.LEAST_AFFECTED_SIDE]:
            d_performance[model][affected_side] = {}
            df_side = df.loc[df[gc.columns.SIDE]==affected_side]

            df_prevalence_correction = pd.DataFrame()
            for med_stage in df_side[gc.columns.PRE_OR_POST].unique():
                df_med_stage = df_side.loc[df_side[gc.columns.PRE_OR_POST]==med_stage].copy()

                fpr, tpr, _ = roc_curve(y_true=np.array(df_med_stage[boolean_colname]), y_score=np.array(df_med_stage[pred_proba_colname]), pos_label=1)
                roc = auc(fpr, tpr)

                seconds_true = df_med_stage.loc[df_med_stage[boolean_colname]==1].shape[0] / gc.parameters.DOWNSAMPLED_FREQUENCY
                seconds_false = df_med_stage.loc[df_med_stage[boolean_colname]==0].shape[0] / gc.parameters.DOWNSAMPLED_FREQUENCY

                d_performance[model][affected_side][med_stage] = {
                    'sens': calculate_sens(df=df_med_stage, pred_colname=pred_colname, true_colname=boolean_colname),
                    'spec': calculate_spec(df=df_med_stage, pred_colname=pred_colname, true_colname=boolean_colname),
                    'auc': roc,
                    'size': {
                        f'{boolean_colname}_s': seconds_true,
                        f'non_{boolean_colname}_s': seconds_false,
                    }
                }

                # minutes of data per med stage, per affected side, per segment duration category
                d_performance[model][affected_side][med_stage]['segment_duration'] = {}
                for segment_duration in df_med_stage[gc.columns.TRUE_GAIT_SEGMENT_CAT].unique():
                    df_segments_cat = df_med_stage.loc[df_med_stage[gc.columns.TRUE_GAIT_SEGMENT_CAT]==segment_duration]

                    cat_minutes = df_segments_cat.loc[df_segments_cat[pred_colname]==0].shape[0]/gc.parameters.DOWNSAMPLED_FREQUENCY/60
                    sens = calculate_sens(df=df_segments_cat, pred_colname=pred_colname, true_colname=boolean_colname)
                    spec = calculate_spec(df=df_segments_cat, pred_colname=pred_colname, true_colname=boolean_colname)

                    d_performance[model][affected_side][med_stage]['segment_duration'][segment_duration] = {
                        'sens': sens,
                        'spec': spec,
                        'minutes': cat_minutes
                    }

                    df_prevalence_correction = pd.concat(
                        [
                            df_prevalence_correction, 
                            pd.DataFrame([cat_minutes, spec, segment_duration, med_stage], index=['minutes', 'spec', 'segment_duration', 'pre_or_post']).T
                        ]
                    ).reset_index(drop=True)

                    if subject in gc.participant_ids.L_PD_IDS:
                        d_performance[model][affected_side][med_stage]['segment_duration'][segment_duration]['arm_activities'] = {}

                        for arm_label in df_segments_cat[gc.columns.ARM_LABEL].unique():
                            df_arm_activity = df_segments_cat.loc[df_segments_cat[gc.columns.ARM_LABEL]==arm_label]

                            d_performance[model][affected_side][med_stage]['segment_duration'][segment_duration]['arm_activities'][arm_label] = {
                                'minutes': df_arm_activity.shape[0]/gc.parameters.DOWNSAMPLED_FREQUENCY/60,
                                'sens': calculate_sens(df=df_arm_activity, pred_colname=pred_colname, true_colname=boolean_colname)
                            }

                # minutes of data per activity of mas
                df_med_stage['label_agg'] = df_med_stage[gc.columns.FREE_LIVING_LABEL].apply(lambda x: mp.activity_map[x] if x in mp.activity_map.keys() else x)
                d_performance[model][affected_side][med_stage]['activities'] = {}

                for activity_label in df_med_stage['label_agg'].unique():
                    df_activity = df_med_stage.loc[df_med_stage['label_agg']==activity_label]
                    d_performance[model][affected_side][med_stage]['activities'][activity_label] = {
                        'spec': calculate_spec(df=df_activity, pred_colname=pred_colname, true_colname=boolean_colname),
                    }

                # minutes of data per arm activity of mas
                if subject in gc.participant_ids.L_PD_IDS:
                    d_performance[model][affected_side][med_stage]['arm_activities'] = {}

                    for arm_label in df_med_stage[gc.columns.ARM_LABEL].unique():
                        df_arm_activity = df_med_stage.loc[df_med_stage[gc.columns.ARM_LABEL]==arm_label]

                        d_performance[model][affected_side][med_stage]['arm_activities'][arm_label] = {
                            'mins': df_arm_activity.shape[0]/gc.parameters.DOWNSAMPLED_FREQUENCY/60,
                            'sens': calculate_sens(df=df_arm_activity, pred_colname=pred_colname, true_colname=boolean_colname)
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

            # correct for segment duration differences
            df_prev_overall = (df_prevalence_correction.groupby('segment_duration')['minutes'].sum() / df_prevalence_correction['minutes'].sum()).reset_index(name='prop_overall')
            df_prev_specific = (df_prevalence_correction.groupby(['pre_or_post', 'segment_duration'])['minutes'].sum() / df_prevalence_correction.groupby('pre_or_post')['minutes'].sum()).reset_index(name='prop_specific')

            df_prevalence_correction = pd.merge(left=df_prevalence_correction, right=df_prev_specific, how='left', on=['pre_or_post', 'segment_duration'])
            df_prevalence_correction = pd.merge(left=df_prevalence_correction, right=df_prev_overall, how='left', on=['segment_duration'])            

            for med_stage in df_side[gc.columns.PRE_OR_POST].unique():
                spec_corrected = (df_prevalence_correction.loc[df_prevalence_correction['pre_or_post']==med_stage, 'spec'] * 
                                  df_prevalence_correction.loc[df_prevalence_correction['pre_or_post']==med_stage, 'prop_specific'] / 
                                  df_prevalence_correction.loc[df_prevalence_correction['pre_or_post']==med_stage, 'prop_overall']).values[0]
                
                d_performance[model][affected_side][med_stage]['spec_corrected'] = spec_corrected
    
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

        df_diff.columns = ['id', 'dataset', 'diff_median_rom', 'diff_95p_rom']
        return d_quantification, df_diff
    else:
        return d_quantification


def generate_results(subject, step):
    if step not in ['gait', 'arm_activity', 'quantification']:
        raise ValueError(f"Invalid step: {step}")

    if subject in gc.participant_ids.L_HC_IDS and step == 'arm_activity':
        return
    
    if step in ['gait', 'arm_activity']:
        print(f"Processing {subject} - {step}...")
        d_output =  generate_results_classification(
            step=step, 
            subject=subject,
            segment_gap_s=1.5
        )

        json_filename = f'{subject}.json'

        with open(os.path.join(gc.paths.PATH_OUTPUT, 'classification_performance', step, json_filename), 'w') as f:
            json.dump(d_output, f, indent=4)
        return 

    else:
        # Only run generate_results_quantification if subject is in L_PD_IDS
        if subject in gc.participant_ids.L_PD_IDS:
            print(f"Processing {subject} - {step}...")
            d_output, df_diff = generate_results_quantification(subject)

            json_filename = f'{subject}.json'
            pkl_filename = f'{subject}.pkl'

            with open(os.path.join(gc.paths.PATH_OUTPUT, 'quantification', json_filename), 'w') as f:
                json.dump(d_output, f, indent=4)

            df_diff.to_pickle(os.path.join(gc.paths.PATH_OUTPUT, 'quantification', pkl_filename))

        return