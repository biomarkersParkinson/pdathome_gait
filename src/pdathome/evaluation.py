import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from scipy.stats import wilcoxon, ranksums
from sklearn.metrics import roc_curve, auc

from pdathome.constants import global_constants as gc, mappings as mp
from pdathome.preprocessing import add_segment_category
from pdathome.quantification import compute_aggregations, compute_effect_size

def calculate_metric(df, pred_colname, true_colname, metric, pred_proba_colname=None):
    if metric == 'sens':
        return calculate_sens(df, pred_colname, true_colname)
    elif metric == 'spec':
        return calculate_spec(df, pred_colname, true_colname)
    elif metric == 'auc' and pred_proba_colname is not None:
        fpr, tpr, _ = roc_curve(y_true=np.array(df[true_colname]), y_score=np.array(df[pred_proba_colname]), pos_label=1)
        return auc(fpr, tpr)
    else:
        raise ValueError('Invalid metric')
    

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


def generate_clinical_scores(subject):
    df_patient_info = pd.read_pickle(os.path.join(gc.paths.PATH_CLINICAL_DATA, 'df_patient_info_updrs_3.pkl'))
    df_subject = df_patient_info.loc[df_patient_info['record_id']==subject]

    age = int(datetime.datetime.now().year - df_subject['year_of_birth'].iloc[0])
    ysd = int(datetime.datetime.now().year - df_subject['year_diagnosis'].iloc[0])
    gender = 'male' if df_subject['gender'].iloc[0] == 1 else 'female'

    d_clinical = {
        'age': age,
        'ysd': ysd,
        'gender': gender,
    }

    d_clinical['updrs'] = {}
    for med_stage, med_prefix in zip([gc.descriptives.PRE_MED, gc.descriptives.POST_MED], ['OFF', 'ON']):
        d_clinical['updrs'][med_stage] = {}
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
            
            d_clinical['updrs'][med_stage][affected_side] = {
                'subscore': np.sum(df_subject[updrs_3_hypokinesia_stage_cols], axis=1).values[0],
                'total': np.sum(df_subject[updrs_3_stage_cols], axis=1).values[0]
            }

    return d_clinical


def generate_results_classification(step, subject, segment_gap_s):

    d_performance = {}

    custom_order_segments = ['short', 'moderately_long', 'long', 'very_long']
  
    for model in [gc.classifiers.LOGISTIC_REGRESSION, gc.classifiers.RANDOM_FOREST]:
        d_performance[model] = {}

        if step == 'gait':
            # Paths
            path_predictions = gc.paths.PATH_GAIT_PREDICTIONS

            # Columns
            pred_proba_colname = gc.columns.PRED_GAIT_PROBA
            pred_colname = gc.columns.PRED_GAIT
            label_colname = gc.columns.FREE_LIVING_LABEL

            # Metrics
            metric_to_correct = 'sens'
            arm_label_metric = 'sens'

            # Labels
            boolean_colname = 'gait'
            value_label = 'Walking'

            # Values
            outcome_value = 1 # Walking

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

            # Metrics
            metric_to_correct = 'spec'
            arm_label_metric = 'sens'

            # Values
            outcome_value = 0 # No other arm activity

            with open(os.path.join(gc.paths.PATH_THRESHOLDS, step, f'{model}.txt'), 'r') as f:
                clf_threshold = float(f.read())

        l_raw_cols = [gc.columns.TIME, gc.columns.SIDE, gc.columns.FREE_LIVING_LABEL]
        l_segment_cats = []

        # For PD, keep the segments of annotations for each step
        l_raw_cols += [gc.columns.TRUE_SEGMENT_NR, gc.columns.TRUE_SEGMENT_CAT]
        if subject in gc.participant_ids.L_PD_IDS:
            l_raw_cols += [gc.columns.ARM_LABEL, gc.columns.PRE_OR_POST]

            if step == 'gait' and subject in gc.participant_ids.L_TREMOR_IDS:
                l_raw_cols.append(gc.columns.TREMOR_LABEL)
        
        # Predictions
        df_predictions = pd.read_pickle(os.path.join(path_predictions, model, f'{subject}.pkl'))
        
        # Load raw data
        l_dfs = []

        # Set segment column names for annotations
        if step == 'gait':
            activity_colname = gc.columns.FREE_LIVING_LABEL
            activity_value = 'Walking'

        elif step == 'arm_activity':
            if subject in gc.participant_ids.L_PD_IDS:
                activity_colname = gc.columns.ARM_LABEL
                activity_value = 'Gait without other behaviours or other positions' # TEMPORARILY TRY WITH ARM ACTIVITY INSTEAD OF GAIT
            else:
                activity_colname = gc.columns.FREE_LIVING_LABEL
                activity_value = 'Walking'

        l_segment_cats.append(gc.columns.TRUE_SEGMENT_CAT)

        for side in [gc.descriptives.MOST_AFFECTED_SIDE, gc.descriptives.LEAST_AFFECTED_SIDE]:
            df_raw = pd.read_pickle(os.path.join(gc.paths.PATH_PREPARED_DATA, f'{subject}_{side}.pkl')).assign(side=side)

            df_raw = add_segment_category(
                    df=df_raw, activity_colname=activity_colname,
                    time_colname=gc.columns.TIME, segment_nr_colname=gc.columns.TRUE_SEGMENT_NR,
                    segment_cat_colname=gc.columns.TRUE_SEGMENT_CAT, segment_gap_s=segment_gap_s, 
                    activity_value=activity_value)

            l_dfs.append(df_raw)

        df_raw = pd.concat(l_dfs, axis=0)

        # Merge labels
        df = pd.merge(left=df_predictions, right=df_raw[l_raw_cols], how='left', on=[gc.columns.TIME, gc.columns.SIDE])

        # Use classification threshold to set predictions
        df.loc[df[pred_proba_colname] >= clf_threshold, pred_colname] = 1
        df.loc[df[pred_proba_colname] < clf_threshold, pred_colname] = 0

        if step == 'arm_activity':
            activity_colname = pred_colname
            activity_value = outcome_value

            df = add_segment_category(
                df=df, activity_colname=activity_colname,
                segment_nr_colname=gc.columns.PRED_SEGMENT_NR, segment_cat_colname=gc.columns.PRED_SEGMENT_CAT,
                time_colname=gc.columns.TIME, segment_gap_s=segment_gap_s, 
                activity_value=activity_value)
            
            l_segment_cats.append(gc.columns.PRED_SEGMENT_CAT)

        # Set boolean for label (ground truth)
        if not (subject in gc.participant_ids.L_HC_IDS and step == 'arm_activity'):
            df.loc[df[label_colname] == value_label, boolean_colname] = outcome_value
            df.loc[df[label_colname] != value_label, boolean_colname] = 1 - outcome_value

        if subject in gc.participant_ids.L_PD_IDS:
            df.loc[df[gc.columns.ARM_LABEL]=='Holding an object behind ', gc.columns.ARM_LABEL] = 'Holding an object behind'
            df[gc.columns.ARM_LABEL] = df.loc[~df[gc.columns.ARM_LABEL].isna(), gc.columns.ARM_LABEL].apply(lambda x: mp.arm_labels_rename[x])
        else:
            df[gc.columns.PRE_OR_POST] = gc.descriptives.CONTROLS
     
        # statistics per arm activity
        if subject in gc.participant_ids.L_PD_IDS:
            d_performance[model]['arm_activities'] = {}

            for arm_label in df[gc.columns.ARM_LABEL].unique():
                df_arm_activity = df.loc[df[gc.columns.ARM_LABEL]==arm_label]

                d_performance[model]['arm_activities'][arm_label] = {
                    'mins': df_arm_activity.shape[0]/gc.parameters.DOWNSAMPLED_FREQUENCY/60,
                    arm_label_metric: calculate_metric(df=df_arm_activity, pred_colname=pred_colname, true_colname=boolean_colname, metric=arm_label_metric)
                }
            
        # make segments and segment duration categories
        for affected_side in [gc.descriptives.MOST_AFFECTED_SIDE, gc.descriptives.LEAST_AFFECTED_SIDE]:
            d_performance[model][affected_side] = {}
            df_side = df.loc[df[gc.columns.SIDE]==affected_side]

            prevalence_data = []
            for med_stage in df_side[gc.columns.PRE_OR_POST].unique():
                d_performance[model][affected_side][med_stage] = {}
                df_med_stage = df_side.loc[df_side[gc.columns.PRE_OR_POST]==med_stage].copy()

                pred_seconds_true = df_med_stage.loc[df_med_stage[pred_colname]==1].shape[0] / gc.parameters.DOWNSAMPLED_FREQUENCY
                pred_seconds_false = df_med_stage.loc[df_med_stage[pred_colname]==0].shape[0] / gc.parameters.DOWNSAMPLED_FREQUENCY

                d_performance[model][affected_side][med_stage]['size'] = {
                    f'pred_{boolean_colname}_s': pred_seconds_true,
                    f'pred_no_{boolean_colname}_s': pred_seconds_false,
                }

                if not (subject in gc.participant_ids.L_HC_IDS and step == 'arm_activity'):
                    ann_seconds_true = df_med_stage.loc[df_med_stage[boolean_colname]==1].shape[0] / gc.parameters.DOWNSAMPLED_FREQUENCY
                    ann_seconds_false = df_med_stage.loc[df_med_stage[boolean_colname]==0].shape[0] / gc.parameters.DOWNSAMPLED_FREQUENCY

                    d_performance[model][affected_side][med_stage]['size'][f'ann_{boolean_colname}_s'] = ann_seconds_true
                    d_performance[model][affected_side][med_stage]['size'][f'ann_no_{boolean_colname}_s'] = ann_seconds_false

                    for metric in ['sens', 'spec', 'auc']:
                        d_performance[model][affected_side][med_stage][metric] = calculate_metric(
                            df=df_med_stage, pred_colname=pred_colname, pred_proba_colname=pred_proba_colname,
                            true_colname=boolean_colname, metric=metric)

                # minutes of data per med stage, per affected side, per segment duration category
                for segment_cat_type in l_segment_cats:
                    d_performance[model][affected_side][med_stage][f'{segment_cat_type}_duration'] = {}
                    for segment_duration in df_med_stage[segment_cat_type].unique():
                        d_performance[model][affected_side][med_stage][f'{segment_cat_type}_duration'][segment_duration] = {}
                        df_segments_cat = df_med_stage.loc[df_med_stage[segment_cat_type] == segment_duration]

                        cat_minutes_pred = df_segments_cat.loc[df_segments_cat[pred_colname] == outcome_value].shape[0] / gc.parameters.DOWNSAMPLED_FREQUENCY / 60
                        d_performance[model][affected_side][med_stage][f'{segment_cat_type}_duration'][segment_duration]['minutes_pred'] = cat_minutes_pred

                        if not (subject in gc.participant_ids.L_HC_IDS and step == 'arm_activity'):
                            cat_minutes_true = df_segments_cat.loc[df_segments_cat[boolean_colname] == outcome_value].shape[0] / gc.parameters.DOWNSAMPLED_FREQUENCY / 60
                            d_performance[model][affected_side][med_stage][f'{segment_cat_type}_duration'][segment_duration]['minutes_true'] = cat_minutes_true

                            if segment_duration != 'non_gait':
                                for metric in ['sens', 'spec']:
                                    d_performance[model][affected_side][med_stage][f'{segment_cat_type}_duration'][segment_duration][metric] = calculate_metric(
                                        df=df_segments_cat, pred_colname=pred_colname, true_colname=boolean_colname, metric=metric)

                                # Append to prevalence_data list
                                if segment_cat_type == gc.columns.TRUE_SEGMENT_CAT:
                                    prevalence_data.append({
                                        'minutes': cat_minutes_true,
                                        metric_to_correct: d_performance[model][affected_side][med_stage][f'{segment_cat_type}_duration'][segment_duration][metric_to_correct],
                                        f'{segment_cat_type}_duration': segment_duration,
                                        gc.columns.PRE_OR_POST: med_stage
                                    })

                                if subject in gc.participant_ids.L_PD_IDS:
                                    d_performance[model][affected_side][med_stage][f'{segment_cat_type}_duration'][segment_duration]['arm_activities'] = {}

                                    for arm_label in df_segments_cat[gc.columns.ARM_LABEL].unique():
                                        df_arm_activity = df_segments_cat.loc[df_segments_cat[gc.columns.ARM_LABEL]==arm_label]

                                        d_performance[model][affected_side][med_stage][f'{segment_cat_type}_duration'][segment_duration]['arm_activities'][arm_label] = {
                                            'minutes': df_arm_activity.shape[0]/gc.parameters.DOWNSAMPLED_FREQUENCY/60,
                                            arm_label_metric: calculate_metric(df=df_arm_activity, pred_colname=pred_colname, true_colname=boolean_colname, metric=arm_label_metric)
                                        }

                # minutes of data per activity of mas
                if step == 'gait': 
                    df_med_stage['label_agg'] = df_med_stage[gc.columns.FREE_LIVING_LABEL].apply(lambda x: mp.activity_map[x] if x in mp.activity_map.keys() else x)
                    d_performance[model][affected_side][med_stage]['activities'] = {}

                    for activity_label in df_med_stage['label_agg'].unique():
                        df_activity = df_med_stage.loc[df_med_stage['label_agg']==activity_label]
                        d_performance[model][affected_side][med_stage]['activities'][activity_label] = {
                            'spec': calculate_metric(df=df_activity, pred_colname=pred_colname, true_colname=boolean_colname, metric='spec'),
                        }

                # minutes of data per arm activity of mas
                if subject in gc.participant_ids.L_PD_IDS:
                    d_performance[model][affected_side][med_stage]['arm_activities'] = {}

                    for arm_label in df_med_stage[gc.columns.ARM_LABEL].unique():
                        df_arm_activity = df_med_stage.loc[df_med_stage[gc.columns.ARM_LABEL]==arm_label]

                        d_performance[model][affected_side][med_stage]['arm_activities'][arm_label] = {
                            'mins': df_arm_activity.shape[0]/gc.parameters.DOWNSAMPLED_FREQUENCY/60,
                            arm_label_metric: calculate_metric(df=df_arm_activity, pred_colname=pred_colname, true_colname=boolean_colname, metric=arm_label_metric)
                        }

                # effect of tremor on specificity
                if subject in gc.participant_ids.L_TREMOR_IDS and step == 'gait':

                    df_med_stage['tremor_label_binned'] = df_med_stage[gc.columns.TREMOR_LABEL].apply(
                        lambda x: 'tremor' if x in ['Slight or mild tremor', 'Moderate tremor', 'Severe tremor', 'Tremor with significant upper limb activity'] else
                        ('no_tremor' if x in ['No tremor', 'Periodic activity of hand/arm similar frequency to tremor', 'No tremor with significant upper limb activity'] else
                        np.nan
                        )
                    )

                    for tremor_type in [x for x in df_med_stage['tremor_label_binned'].unique() if not pd.isna(x)]:
                        d_performance[model][affected_side][med_stage][f'{tremor_type}_spec'] = calculate_spec(df=df_med_stage.loc[df_med_stage['tremor_label_binned']==tremor_type], pred_colname=pred_colname, true_colname=boolean_colname)
                
            if not (subject in gc.participant_ids.L_HC_IDS and step == 'arm_activity'):
                # Convert prevalence data into a DataFrame outside of the loop
                df_prevalence_correction = pd.DataFrame(prevalence_data)

                # Calculate prevalence proportions overall and per medication stage
                df_prev_overall = (df_prevalence_correction.groupby(f'{gc.columns.TRUE_SEGMENT_CAT}_duration')['minutes'].sum() /
                                df_prevalence_correction['minutes'].sum()).reset_index(name='prop_overall')

                df_prev_specific = (df_prevalence_correction.groupby([gc.columns.PRE_OR_POST, f'{gc.columns.TRUE_SEGMENT_CAT}_duration'])['minutes'].sum() / 
                                    df_prevalence_correction.groupby(gc.columns.PRE_OR_POST)['minutes'].sum()).reset_index(name='prop_specific')

                # Merge prevalence data back into df_prevalence_correction
                df_prevalence_correction = pd.merge(left=df_prevalence_correction, right=df_prev_specific, how='left', on=[gc.columns.PRE_OR_POST, f'{gc.columns.TRUE_SEGMENT_CAT}_duration'])
                df_prevalence_correction = pd.merge(left=df_prevalence_correction, right=df_prev_overall, how='left', on=f'{gc.columns.TRUE_SEGMENT_CAT}_duration')            

                # Calculate total prevalence of each segment duration across medication stages
                total_prevalence = df_prevalence_correction.groupby(f'{gc.columns.TRUE_SEGMENT_CAT}_duration')['minutes'].sum()	

                # Normalize to get proportions for each segment duration
                total_prevalence_proportions = total_prevalence / total_prevalence.sum()          

                # Loop through each medication stage to calculate corrected specificity
                for med_stage in df_side[gc.columns.PRE_OR_POST].unique():
                    df_med_stage_corrected = df_prevalence_correction.loc[df_prevalence_correction[gc.columns.PRE_OR_POST] == med_stage].copy()
                    df_med_stage_corrected[f'{gc.columns.TRUE_SEGMENT_CAT}_duration'] = pd.Categorical(df_med_stage_corrected[f'{gc.columns.TRUE_SEGMENT_CAT}_duration'], custom_order_segments)
                    df_med_stage_corrected = df_med_stage_corrected.sort_values(by=f'{gc.columns.TRUE_SEGMENT_CAT}_duration').reset_index(drop=True)

                    # Safeguard to avoid missing data or incorrect indexing
                    if not df_med_stage_corrected.empty:
                        metric_values = df_med_stage_corrected[metric_to_correct]

                        # Calculate the weighted sensitivity using the proportions
                        weighted_sensitivity = metric_values * total_prevalence_proportions.loc[df_med_stage_corrected[f'{gc.columns.TRUE_SEGMENT_CAT}_duration']].values

                        # Sum the weighted sensitivities to get a single corrected sensitivity
                        metric_corrected = weighted_sensitivity.sum()

                        d_performance[model][affected_side][med_stage][f'{metric_to_correct}_corrected'] = metric_corrected

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


def generate_results_quantification(subject: str, segment_by: str) -> tuple[dict, pd.DataFrame]:
    """Generate quantification results for a given subject."""
    classification_threshold = 0.5
    use_timestamps = False

    if segment_by == gc.columns.FREE_LIVING_LABEL:
        segment_nr_colname = gc.columns.TRUE_SEGMENT_NR
        segment_cat_colname = gc.columns.TRUE_SEGMENT_CAT
        activity_colname = gc.columns.FREE_LIVING_LABEL
        activity_value = 'Walking'
    elif segment_by == gc.columns.ARM_LABEL:
        segment_nr_colname = gc.columns.TRUE_SEGMENT_NR
        segment_cat_colname = gc.columns.TRUE_SEGMENT_CAT
        activity_colname = gc.columns.ARM_LABEL
        activity_value = 'Gait without other behaviours or other positions'
    elif segment_by == gc.columns.PRED_OTHER_ARM_ACTIVITY:
        segment_nr_colname = gc.columns.PRED_SEGMENT_NR
        segment_cat_colname = gc.columns.PRED_SEGMENT_CAT
        activity_colname = gc.columns.PRED_OTHER_ARM_ACTIVITY
        activity_value = 0
    else:
        raise ValueError('Invalid segment_by')

    # Load timestamps for both sides
    df_ts_mas = load_arm_activity_timestamps(subject, gc.descriptives.MOST_AFFECTED_SIDE)
    df_ts_las = load_arm_activity_timestamps(subject, gc.descriptives.LEAST_AFFECTED_SIDE)
    df_ts = pd.concat([df_ts_mas, df_ts_las], axis=0)

    # Explode the timestamps DataFrame
    df_ts_exploded = df_ts.explode(gc.columns.TIME)

    # Load arm activity predictions
    df_predictions_ts = pd.read_pickle(os.path.join(gc.paths.PATH_ARM_ACTIVITY_PREDICTIONS, gc.classifiers.LOGISTIC_REGRESSION, f'{subject}.pkl'))
    df_predictions = pd.read_pickle(os.path.join(gc.paths.PATH_ARM_ACTIVITY_PREDICTIONS, gc.classifiers.LOGISTIC_REGRESSION, f'{subject}_df.pkl'))

    # Set pred rounded based on the threshold
    df_predictions_ts[gc.columns.PRED_OTHER_ARM_ACTIVITY] = (df_predictions_ts[gc.columns.PRED_OTHER_ARM_ACTIVITY_PROBA] >= classification_threshold).astype(int)
    df_predictions[gc.columns.PRED_OTHER_ARM_ACTIVITY] = (df_predictions[gc.columns.PRED_OTHER_ARM_ACTIVITY_PROBA] >= classification_threshold).astype(int)

    # Add predictions to raw data
    df_ts = pd.merge(
        left=df_predictions_ts, 
        right=df_ts_exploded, 
        how='left', 
        on=[gc.columns.TIME, gc.columns.SIDE]
    )

    df_raw_mas = pd.read_pickle(os.path.join(gc.paths.PATH_PREPARED_DATA, f'{subject}_{gc.descriptives.MOST_AFFECTED_SIDE}.pkl')).assign(side=gc.descriptives.MOST_AFFECTED_SIDE)
    df_raw_las = pd.read_pickle(os.path.join(gc.paths.PATH_PREPARED_DATA, f'{subject}_{gc.descriptives.LEAST_AFFECTED_SIDE}.pkl')).assign(side=gc.descriptives.LEAST_AFFECTED_SIDE)
    df_raw = pd.concat([df_raw_mas, df_raw_las], axis=0)

    df_raw = add_segment_category(df=df_raw, time_colname=gc.columns.TIME, segment_nr_colname=segment_nr_colname,
                                segment_cat_colname=segment_cat_colname, activity_colname=activity_colname,
                                segment_gap_s=gc.parameters.SEGMENT_GAP_GAIT, activity_value=activity_value)
    
    if subject in gc.participant_ids.L_HC_IDS:
        df_raw[gc.columns.PRE_OR_POST] = gc.descriptives.CONTROLS

    # Merge annotations
    l_merge_cols = [gc.columns.TIME, gc.columns.SIDE, gc.columns.PRE_OR_POST, gc.columns.FREE_LIVING_LABEL, segment_nr_colname, segment_cat_colname]
    if subject in gc.participant_ids.L_PD_IDS:
        l_merge_cols += [gc.columns.ARM_LABEL]

    df_ts = pd.merge(left=df_ts, right=df_raw[l_merge_cols],
        how='left', on=[gc.columns.SIDE, gc.columns.TIME]
    )

    # Set other arm activity boolean
    if subject in gc.participant_ids.L_PD_IDS:
        df_ts.loc[df_ts[gc.columns.ARM_LABEL]=='Gait without other behaviours or other positions', 'other_arm_activity_boolean'] = 0
        df_ts.loc[df_ts[gc.columns.ARM_LABEL]!='Gait without other behaviours or other positions', 'other_arm_activity_boolean'] = 1
        df_ts.loc[df_ts[gc.columns.ARM_LABEL]=='Holding an object behind ', gc.columns.ARM_LABEL] = 'Holding an object behind'
        df_ts[gc.columns.ARM_LABEL] = df_ts.loc[~df_ts[gc.columns.ARM_LABEL].isna(), gc.columns.ARM_LABEL].apply(lambda x: mp.arm_labels_rename[x])

     # Load arm activity features for both sides
    df_features_mas = load_arm_activity_features(subject, gc.descriptives.MOST_AFFECTED_SIDE)
    df_features_las = load_arm_activity_features(subject, gc.descriptives.LEAST_AFFECTED_SIDE)
    df_features = pd.concat([df_features_mas, df_features_las], axis=0)

    # Combine features and preprocess
    df_features['peak_velocity'] = (df_features['forward_peak_ang_vel_mean'] + df_features['backward_peak_ang_vel_mean']) / 2
    df_features = df_features.drop(columns=[gc.columns.TIME])

    if subject in gc.participant_ids.L_HC_IDS:
        df_features[gc.columns.PRE_OR_POST] = gc.descriptives.CONTROLS

    # Merge features with exploded timestamps
    l_merge_cols = [gc.columns.SIDE, gc.columns.PRE_OR_POST, gc.columns.WINDOW_NR]
    df_ts = pd.merge(
        left=df_features, 
        right=df_ts, 
        how='right', 
        on=l_merge_cols
    )

    # Map windows to segment categories using majority voting
    true_segment_counts = df_ts.groupby([gc.columns.WINDOW_NR, segment_cat_colname]).size().reset_index(name='count')

    max_true_segment_per_window = true_segment_counts.sort_values(by=[gc.columns.WINDOW_NR, 'count'], ascending=[True, False]) \
                                        .drop_duplicates(subset=gc.columns.WINDOW_NR, keep='first')

    df = pd.merge(
        left=df_features,
        right=df_predictions[[gc.columns.SIDE, gc.columns.WINDOW_NR, gc.columns.PRED_OTHER_ARM_ACTIVITY]],
        how='right',
        on=[gc.columns.SIDE, gc.columns.WINDOW_NR]
    )

    df = pd.merge(
        left=df,
        right=max_true_segment_per_window[[gc.columns.WINDOW_NR, segment_cat_colname]],
        how='left',
        on=gc.columns.WINDOW_NR
    )

    if use_timestamps:
        true_value_colname = 'other_arm_activity_boolean'
        # Group by relevant columns and compute mean for features
        l_groupby_cols = [gc.columns.SIDE, gc.columns.WINDOW_NR, gc.columns.PRE_OR_POST, segment_cat_colname, gc.columns.PRED_OTHER_ARM_ACTIVITY]
        if subject in gc.participant_ids.L_PD_IDS:
            l_groupby_cols += [true_value_colname]

        df = df_ts.groupby(l_groupby_cols)[['peak_velocity', 'range_of_motion']].mean().reset_index()
    
    else:
        true_value_colname = 'other_arm_activity_majority_voting'
        l_groupby_cols = [gc.columns.SIDE, gc.columns.WINDOW_NR, gc.columns.PRE_OR_POST, segment_cat_colname, gc.columns.PRED_OTHER_ARM_ACTIVITY]
        if subject in gc.participant_ids.L_PD_IDS:
            l_groupby_cols += [true_value_colname]
        df = df.groupby(l_groupby_cols)[['peak_velocity', 'range_of_motion']].mean().reset_index()

    d_quantification = {}

    # Compute unfiltered gait aggregations
    d_quantification['unfiltered_gait'] = compute_aggregations(subject, df, segment_cat_colname=segment_cat_colname, use_timestamps=use_timestamps)

    # Compute filtered gait aggregations
    d_quantification['filtered_gait'] = compute_aggregations(subject, df.loc[df[gc.columns.PRED_OTHER_ARM_ACTIVITY] == 0], segment_cat_colname=segment_cat_colname, use_timestamps=use_timestamps)

    # Compute annotated no other arm activity for specific participants
    if subject in gc.participant_ids.L_PD_IDS:
        df_diff = pd.DataFrame()
        d_quantification['true_no_other_arm_activity'] = compute_aggregations(subject, df.loc[df[true_value_colname] == 0], segment_cat_colname=segment_cat_colname, use_timestamps=use_timestamps)

        df = df.loc[df[gc.columns.SIDE]=='MAS']

        es_mrom, diff_mrom = compute_effect_size(df, 'range_of_motion', 'median', segment_cat_colname=segment_cat_colname)
        es_prom, diff_prom = compute_effect_size(df, 'range_of_motion', '95', segment_cat_colname=segment_cat_colname)

        d_quantification['effect_size'] = {
            'median_rom': es_mrom,
            '95p_rom': es_prom
        }

        for dataset in diff_mrom.keys():
            df_diff = pd.concat([df_diff, pd.DataFrame([subject, dataset, diff_mrom[dataset], diff_prom[dataset]]).T])

        df_diff.columns = [gc.columns.ID, 'dataset', 'diff_median_rom', 'diff_95p_rom']
        return d_quantification, df_diff
    else:
        return d_quantification


def generate_results(subject, step):
    if step not in ['gait', 'arm_activity', 'quantification']:
        raise ValueError(f"Invalid step: {step}")

    
    if step in ['gait', 'arm_activity']:
        print(f"Processing {subject} - {step}...")
        d_output =  generate_results_classification(
            step=step, 
            subject=subject,
            segment_gap_s=1.5
        )

        if subject in gc.participant_ids.L_PD_IDS:
            d_output['clinical'] = generate_clinical_scores(subject)

        json_filename = f'{subject}.json'

        with open(os.path.join(gc.paths.PATH_OUTPUT, 'classification_performance', step, json_filename), 'w') as f:
            json.dump(d_output, f, indent=4)

        return 

    else:
        segment_by = gc.columns.FREE_LIVING_LABEL

        if subject in gc.participant_ids.L_PD_IDS + gc.participant_ids.L_HC_IDS:
            print(f"Processing {subject} - {step}...")

            json_filename = f'{subject}.json'
            pkl_filename = f'{subject}.pkl'

            if subject in gc.participant_ids.L_PD_IDS:
                d_output, df_diff = generate_results_quantification(subject, segment_by)
                df_diff.to_pickle(os.path.join(gc.paths.PATH_OUTPUT, 'quantification', pkl_filename))
            else:
                d_output = generate_results_quantification(subject, segment_by)

            with open(os.path.join(gc.paths.PATH_OUTPUT, 'quantification', json_filename), 'w') as f:
                json.dump(d_output, f, indent=4)

        return