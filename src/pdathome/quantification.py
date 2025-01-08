import numpy as np
import os
import pandas as pd

from paradigma.config import ArmActivityFeatureExtractionConfig

from pdathome.constants import global_constants as gc

def quantify_arm_swing(subject, affected_side):
        arm_swing_parameters = {}
        config = ArmActivityFeatureExtractionConfig()

        # load timestamp data
        df_ts = pd.read_parquet(os.path.join(gc.paths.PATH_PREPARED_DATA, f'{subject}_{affected_side}.parquet'))

        df_gait = df_ts.loc[df_ts[gc.columns.FREE_LIVING_LABEL] == 'Walking'].copy()

        df_gait[gc.columns.SEGMENT_NR] = create_segments(
            time_array=df_gait[gc.columns.TIME].values,
            max_segment_gap_s=config.max_segment_gap_s,
        )

        df_gait = discard_segments(
            df=df_gait,
            segment_nr_colname=gc.columns.SEGMENT_NR,
            min_segment_length_s=config.min_segment_length_s,
            sampling_frequency=config.sampling_frequency,
        )

        df_gait[gc.columns.SEGMENT_CAT] = categorize_segments(
            config=config,
            df=df_gait
        )

        # load arm swing predictions
        df_pred = pd.read_parquet(os.path.join(gc.paths.PATH_ARM_ACTIVITY_PREDICTIONS, gc.classifiers.ARM_ACTIVITY_CLASSIFIER_SELECTED, f'{subject}_{affected_side}.parquet'))

        # Load classification threshold
        with open(os.path.join(gc.paths.PATH_THRESHOLDS, 'arm_activity', f'{gc.classifiers.ARM_ACTIVITY_CLASSIFIER_SELECTED}.txt'), 'r') as f:
            threshold = float(f.read())

        # merge timestamp data into arm swing predictions (keep only predicted gait timestamps)        

        df_ts = merge_timestamps_and_predictions(
            df_ts=df_ts,
            df_pred=df_pred,
            time_colname=gc.columns.TIME,
            pred_proba_colname=gc.columns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA,
            window_length_s=config.window_length_s,
            fs=config.sampling_frequency,
        )

        df_ts = pd.merge(df_ts, df_gait[[gc.columns.TIME, gc.columns.SEGMENT_CAT]], on=gc.columns.TIME, how='left')

        df_ts = df_ts.dropna(subset=gc.columns.GYROSCOPE_COLS).reset_index(drop=True)

        df_ts[gc.columns.PRED_NO_OTHER_ARM_ACTIVITY] = df_ts[gc.columns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA] >= threshold

        if subject not in gc.participant_ids.PD_IDS:
            df_ts[gc.columns.PRE_OR_POST] = gc.descriptives.CONTROLS

        for med_stage in df_ts[gc.columns.PRE_OR_POST].unique():
            arm_swing_parameters[subject][affected_side][med_stage] = {}

            df_med_stage = df_ts[df_ts[gc.columns.PRE_OR_POST] == med_stage].copy()
            # Perform principal component analysis on the gyroscope signals to obtain the angular velocity in the
            # direction of the swing of the arm 
            df_med_stage[gc.columns.VELOCITY] = pca_transform_gyroscope(
                df=df_med_stage,
                y_gyro_colname=gc.columns.GYROSCOPE_Y,
                z_gyro_colname=gc.columns.GYROSCOPE_Z,
                pred_colname=gc.columns.PRED_NO_OTHER_ARM_ACTIVITY,
            )

            # PER SEGMENT
            time_array = np.array(df_med_stage[gc.columns.TIME])
            df_med_stage[gc.columns.SEGMENT_NR] = create_segments(
                time_array=time_array,
                max_segment_gap_s=config.max_segment_gap_s,
            )

            segmented_data = []
            df_grouped = df_med_stage.groupby(gc.columns.SEGMENT_NR, sort=False)

            for _, group in df_grouped:
                time_array = np.array(group[gc.columns.TIME])
                velocity_array = np.array(group[gc.columns.VELOCITY])

                # Integrate the angular velocity to obtain an estimation of the angle
                angle_array = compute_angle(
                    time_array=time_array,
                    velocity_array=velocity_array,
                )

                # Remove the moving average from the angle to account for possible drift caused by the integration
                # of noise in the angular velocity
                angle_array = remove_moving_average_angle(
                    angle_array=angle_array,
                    fs=config.sampling_frequency,
                )
                if len(angle_array) > 0:  # Skip if no windows are created
                    segmented_data.append(angle_array)

            if len(segmented_data) > 0:
                angle_array = np.concatenate(segmented_data, axis=0)
            else:
                raise ValueError("No windows were created from the given data.")

            df_med_stage[gc.columns.ANGLE] = angle_array

            for filtered in [True, False]:
                if filtered:
                    df_subset = df_med_stage[df_med_stage[gc.columns.PRED_NO_OTHER_ARM_ACTIVITY]==1]
                    key = 'filtered'
                else:
                    df_subset = df_med_stage
                    key = 'unfiltered'

                arm_swing_parameters[subject][affected_side][med_stage][key] = {}

                segment_cats = [x for x in df_subset[gc.columns.SEGMENT_CAT].unique() if pd.notna(x)]

                for segment_length in segment_cats:
                    df_subset_segment_length = df_subset[df_subset[gc.columns.SEGMENT_CAT] == segment_length]

                    angle_array = np.array(df_subset_segment_length[gc.columns.ANGLE])
                    velocity_array = np.array(df_subset_segment_length[gc.columns.VELOCITY])

                    angle_extrema_indices, minima_indices, maxima_indices = extract_angle_extremes(
                        angle_array=angle_array,
                        sampling_frequency=config.sampling_frequency,
                        max_frequency_activity=1.75
                    )

                    feature_dict = {
                        'time_s': len(angle_array) / config.sampling_frequency,
                    }

                    if len(angle_extrema_indices) > 1:
                        # Calculate range of motion based on extrema indices
                        feature_dict['range_of_motion'] = compute_range_of_motion(
                            angle_array=angle_array,
                            extrema_indices=list(angle_extrema_indices),
                        )

                        # Compute the forward and backward peak angular velocities
                        feature_dict['forward_pav'], feature_dict['backward_pav'] = compute_peak_angular_velocity(
                            velocity_array=velocity_array,
                            angle_extrema_indices=angle_extrema_indices,
                            minima_indices=minima_indices,
                            maxima_indices=maxima_indices,
                        )

                    arm_swing_parameters[subject][affected_side][med_stage][key][segment_length] = feature_dict


def compute_effect_size(df, parameter, stat, segment_cat_colname): 
    """
    Computes effect size between pre-med and post-med parameters using either bootstrapped standard deviation or pooled standard deviation.
    
    Args:
    - df (pd.DataFrame): DataFrame containing the data.
    - parameter (str): The parameter column to analyze.
    - stat (str): The point estimate statistic ('median' or '95').
    - segment_cat_colname (str): Column name for segment category.
    
    Returns:
    - d_effect_size (dict): Dictionary containing effect size information.
    - d_diffs (dict): Dictionary containing bootstrapped differences if applicable.
    """

    d_effect_size = {}
    # d_diffs = {}

    for dataset in ['predicted_gait', 'pred_gait_predicted_noaa', 'pred_gait_annotated_noaa']:
        if dataset == 'predicted_gait':
            df_subset = df.copy()
        elif dataset == 'pred_gait_predicted_noaa':
            df_subset = df.loc[df[gc.columns.PRED_NO_OTHER_ARM_ACTIVITY] == 1].copy()
        elif dataset == 'pred_gait_annotated_noaa':
            df_subset = df.loc[df['no_other_arm_activity_boolean'] == 1].copy()
        else:
            raise ValueError("Invalid dataset provided.")

        d_effect_size[dataset] = {}

        # Split into pre-med and post-med
        df_pre = df_subset.loc[df_subset[gc.columns.PRE_OR_POST] == gc.descriptives.PRE_MED]
        df_post = df_subset.loc[df_subset[gc.columns.PRE_OR_POST] == gc.descriptives.POST_MED]

        for segment_category in ['short', 'moderately_long', 'long', 'very_long', 'overall']:
            # Filter by segment category only if not 'overall'
            if segment_category != 'overall':
                df_pre_cat = df_pre.loc[df_pre[segment_cat_colname] == segment_category]
                df_post_cat = df_post.loc[df_post[segment_cat_colname] == segment_category]
            else:
                df_pre_cat = df_pre
                df_post_cat = df_post

            # Get parameter values for pre and post
            pre_vals = df_pre_cat[parameter].dropna().values
            post_vals = df_post_cat[parameter].dropna().values
            
            if len(pre_vals) != 0 and len(post_vals) != 0:
                d_effect_size[dataset][segment_category] = {}
                
                # Point estimate of pre- and post-med (median or 95th percentile)
                if stat == 'median':
                    mu_pre = np.median(pre_vals)
                    mu_post = np.median(post_vals)
                elif stat == '95':
                    mu_pre = np.percentile(pre_vals, 95)
                    mu_post = np.percentile(post_vals, 95)

                d_effect_size[dataset][segment_category]['mu_pre'] = mu_pre
                d_effect_size[dataset][segment_category]['mu_post'] = mu_post

                # # Perform bootstrapping for pre- and post-medication values
                # bootstrapped_pre = bootstrap_samples(pre_vals, stat)
                # bootstrapped_post = bootstrap_samples(post_vals, stat)

                # # Compute bootstrapped differences
                # bootstrapped_differences = bootstrapped_post - bootstrapped_pre
                # std_bootstrap = np.std(bootstrapped_differences)
                    
                # # Handle the case where std_bootstrap is zero
                # if std_bootstrap == 0:
                #     d_effect_size[dataset][segment_category]['effect_size'] = np.nan 
                # else:
                #     d_effect_size[dataset][segment_category]['effect_size'] = (mu_post - mu_pre) / std_bootstrap
                
                # d_effect_size[dataset][segment_category]['std'] = std_bootstrap

                # if segment_category == 'overall':
                #     d_diffs[dataset] = bootstrapped_differences
        
    return d_effect_size #, d_diffs