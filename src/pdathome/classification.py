import json
import numpy as np
import os
import pandas as pd
import pickle
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from typing import Callable, List

from paradigma.gait_analysis_config import GaitFeatureExtractionConfig, ArmSwingFeatureExtractionConfig

from pdathome.constants import classifiers, columns, descriptives, participant_ids, paths, classifier_hyperparameters
from pdathome.load import load_dataframes_directory
from pdathome.utils import save_to_pickle

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def train_test(
    subject,
    l_ids: List[str],
    config_class: Callable,
    l_classifiers: List[str],
    target_column_name: str,
    pred_proba_colname: str,
    pred_colname: str,
    step: str,
    path_features: str,
    path_predictions: str
):
    # Initialize configuration
    config = config_class()

    # Define predictors
    l_predictors = list(config.d_channels_values.keys())
    l_predictors_scale = [x for x in l_predictors if 'dominant' not in x]

    df_all_subjects = load_dataframes_directory(
        directory_path=path_features,
        l_ids=l_ids
    )

    print(f"... Processing subject {subject} ...")

    for classifier_name in l_classifiers:

        # Train and test model
        df_test, classification_threshold = cv_train_test_model(
            subject=subject,
            df=df_all_subjects,
            classifier_name=classifier_name,
            l_predictors=l_predictors,
            l_predictors_scale=l_predictors_scale,
            target_column_name=target_column_name,
            pred_proba_colname=pred_proba_colname,
            pred_colname=pred_colname,
            step=step
        )

        # Save predictions
        windows_to_timestamps(
            subject=subject, df=df_test,
            path_output=os.path.join(path_predictions, classifier_name),
            pred_proba_colname=pred_proba_colname,
            step=step
        )

        with open(os.path.join(paths.PATH_THRESHOLDS, step, f'{classifier_name}_{subject}.txt'), 'w') as f:
            f.write(str(classification_threshold))


def train_test_gait_detection(subject, l_classifiers):
    train_test(
        subject=subject,
        l_ids=participant_ids.L_PD_IDS + participant_ids.L_HC_IDS,
        config_class=GaitFeatureExtractionConfig,
        l_classifiers=l_classifiers,
        target_column_name=columns.GAIT_MAJORITY_VOTING,
        pred_proba_colname=columns.PRED_GAIT_PROBA,
        pred_colname=columns.PRED_GAIT,
        step='gait',
        path_features=paths.PATH_GAIT_FEATURES,
        path_predictions=paths.PATH_GAIT_PREDICTIONS
    )


def train_test_filtering_gait(subject, l_classifiers):
    train_test(
        subject=subject,
        l_ids=participant_ids.L_PD_IDS,
        config_class=ArmSwingFeatureExtractionConfig,
        l_classifiers=l_classifiers,
        target_column_name=columns.OTHER_ARM_ACTIVITY_MAJORITY_VOTING,
        pred_proba_colname=columns.PRED_OTHER_ARM_ACTIVITY_PROBA,
        pred_colname=columns.PRED_OTHER_ARM_ACTIVITY,
        step='arm_activity',
        path_features=paths.PATH_ARM_ACTIVITY_FEATURES,
        path_predictions=paths.PATH_ARM_ACTIVITY_PREDICTIONS
    )


def cv_train_test_model(subject, df, classifier_name, l_predictors, l_predictors_scale, target_column_name, 
                        pred_proba_colname, pred_colname, step):

    # Check for valid step
    if step not in ['gait', 'arm_activity']:
        raise ValueError("Step not recognized")
    
    # Set class weight based on the step
    class_weight = None if step == 'gait' else 'balanced'

    # Split the data for training and testing
    df_train = df[df[columns.ID] != subject].copy()
    df_test = df[df[columns.ID] == subject].copy()
    
    # Fit a scaler on the pd participants
    pd_train_mask = df_train[columns.ID].isin(participant_ids.L_PD_IDS)
    scaler = StandardScaler()

    # Scale the data
    df_train.loc[pd_train_mask, l_predictors_scale] = scaler.fit_transform(df_train.loc[pd_train_mask, l_predictors_scale])
    df_test[l_predictors_scale] = scaler.transform(df_test[l_predictors_scale])

    # Define train and test sets
    X_train = df_train[l_predictors]
    y_train = df_train[target_column_name].astype(int)
    X_test = df_test[l_predictors]
    y_test = df_test[target_column_name].astype(int)

    # Initialize the model
    if classifier_name == classifiers.RANDOM_FOREST:
        clf = RandomForestClassifier(
            **classifier_hyperparameters[classifiers.RANDOM_FOREST],
            class_weight=class_weight
        )
    elif classifier_name == classifiers.LOGISTIC_REGRESSION:
        clf = LogisticRegression(
            **classifier_hyperparameters[classifiers.LOGISTIC_REGRESSION],
            class_weight=class_weight,
        )
        
    # Train the model
    clf.fit(X_train, y_train)

    # Predict probabilities on the test set
    df_test['true'] = y_test
    df_test[pred_proba_colname] = clf.predict_proba(X_test)[:,1]
    
    # Threshold determination for 'gait' step
    if step == 'gait':
        X_pop_train = df_train.loc[pd_train_mask, l_predictors]
        y_pop_train = df_train.loc[pd_train_mask, target_column_name].astype(int)
        y_train_pred_proba_pop = clf.predict_proba(X_pop_train)[:,1]

        # set threshold to obtain at least 95% train specificity
        fpr, _, thresholds = roc_curve(y_true=y_pop_train, y_score=y_train_pred_proba_pop, pos_label=1)
        threshold_index = np.argmax(fpr >= 0.05) - 1
        classification_threshold = thresholds[threshold_index]

        df_test[pred_colname] = (df_test[pred_proba_colname] >= classification_threshold).astype(int)
    else:
        df_test[pred_colname] = clf.predict(X_test)
        classification_threshold = df_test.loc[df_test[pred_colname] == 1, pred_proba_colname].min()
    
    return df_test, classification_threshold


def windows_to_timestamps(subject, df, path_output, pred_proba_colname, step):

    if step not in ['gait', 'arm_activity']:
        raise ValueError("Step not recognized")

    # Define base columns
    l_subj_cols = [columns.SIDE, columns.WINDOW_NR, pred_proba_colname]
    l_merge_ts_cols = [columns.WINDOW_NR, columns.SIDE]
    l_merge_points_cols = [columns.TIME, columns.SIDE]
    l_groupby_cols = [columns.TIME, columns.SIDE]
    l_explode_cols = [columns.TIME]
    l_dupl_cols = [columns.TIME, columns.SIDE]

    # Add PD-specific columns
    if subject in participant_ids.L_PD_IDS:
        pd_specific_cols = [columns.PRE_OR_POST, columns.ARM_LABEL]
        l_subj_cols.append(columns.PRE_OR_POST)
        l_groupby_cols += pd_specific_cols
        l_explode_cols.append(columns.ARM_LABEL)
        l_merge_ts_cols.append(columns.PRE_OR_POST)
        l_merge_points_cols.append(columns.PRE_OR_POST)
        l_dupl_cols.append(columns.PRE_OR_POST)

    if step == 'gait':
        l_subj_cols += [columns.GAIT_MAJORITY_VOTING, columns.ACTIVITY_LABEL_MAJORITY_VOTING]
        path_features = paths.PATH_GAIT_FEATURES
        l_explode_cols.append(columns.FREE_LIVING_LABEL)
        l_groupby_cols.append(columns.FREE_LIVING_LABEL)
    elif step == 'arm_activity':
        path_features = paths.PATH_ARM_ACTIVITY_FEATURES
        if subject in participant_ids.L_PD_IDS:
            l_subj_cols += [columns.OTHER_ARM_ACTIVITY_MAJORITY_VOTING, columns.ARM_LABEL_MAJORITY_VOTING]

    # Select relevant columns
    df = df[l_subj_cols]     
    
    # Load and combine timestamps data
    df_ts_mas = pd.read_pickle(os.path.join(path_features, f'{subject}_{descriptives.MOST_AFFECTED_SIDE}_ts.pkl')).assign(side=descriptives.MOST_AFFECTED_SIDE)
    df_ts_las = pd.read_pickle(os.path.join(path_features, f'{subject}_{descriptives.LEAST_AFFECTED_SIDE}_ts.pkl')).assign(side=descriptives.LEAST_AFFECTED_SIDE)

    df_ts = pd.concat([df_ts_mas, df_ts_las], ignore_index=True)

    # Explode timestamp data for merging
    df_ts_exploded = df_ts.explode(l_explode_cols)

    # Merge the exploded data with windowed data
    df_single_points = pd.merge(left=df_ts_exploded, right=df, how='right', on=l_merge_ts_cols)
        
    # Reset index after merging
    df_single_points.reset_index(drop=True, inplace=True)

    # Group by relevant columns and calculate mean prediction probability
    if step == 'gait':
        df_pred_per_point = df_single_points.groupby(l_groupby_cols)[pred_proba_colname].mean().reset_index()
    else:
        df_pred_per_point = df_single_points.groupby(l_groupby_cols)[pred_proba_colname].mean().reset_index()

    # Save the final result
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    save_to_pickle(
        df=df_pred_per_point,
        path=path_output,
        filename=f'{subject}.pkl'
    )


def store_model_output(df, step):
    if step not in ['gait', 'arm_activity']:
        raise ValueError("Step not recognized")
    
    # Determine class weight based on the step
    if step == 'gait':
        class_weight = None
        target_column_name = columns.GAIT_MAJORITY_VOTING
        config = GaitFeatureExtractionConfig()
    else:
        class_weight = 'balanced'
        target_column_name = columns.OTHER_ARM_ACTIVITY_MAJORITY_VOTING
        config = ArmSwingFeatureExtractionConfig()

    l_predictors = list(config.d_channels_values.keys())
    l_predictors_scaled = [x for x in l_predictors if 'dominant' not in x]

    # Standardize features based on the PD subjects    
    scaler = StandardScaler()
    df_pd = df.loc[df[columns.ID].isin(participant_ids.L_PD_IDS), l_predictors_scaled]
    df_scaled = df.copy()
    
    scaler.fit(df_pd[l_predictors_scaled])
    df_scaled[l_predictors_scaled] = scaler.transform(df_scaled[l_predictors_scaled])

    X = df_scaled[l_predictors]
    y = df_scaled[target_column_name].astype(int)

    for classifier_name in [classifiers.LOGISTIC_REGRESSION, classifiers.RANDOM_FOREST]:

        # train on all subjects and store model
        if classifier_name == classifiers.RANDOM_FOREST:
            clf = RandomForestClassifier(
                **classifier_hyperparameters[classifiers.RANDOM_FOREST],
                class_weight=class_weight
            )
        elif classifier_name == classifiers.LOGISTIC_REGRESSION:
            clf = LogisticRegression(
                **classifier_hyperparameters[classifiers.LOGISTIC_REGRESSION],
                class_weight=class_weight,
            )

        # Fit the model
        clf.fit(X, y)

        # Save the model as a pickle file
        model_path = os.path.join(paths.PATH_CLASSIFIERS, step, f'{classifier_name}.pkl')

        if not os.path.exists(paths.PATH_CLASSIFIERS):
            os.makedirs(paths.PATH_CLASSIFIERS)

        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)

        # Save coefficients as json
        with open(os.path.join(paths.PATH_COEFFICIENTS, step, f'{classifier_name}.json'), 'w') as f:
            if type(clf).__name__.lower() == 'logisticregression': 
                coefficients = {k: v for k, v in zip(l_predictors, clf.coef_[0])}
            elif type(clf).__name__.lower() == 'randomforestclassifier':
                coefficients = {k: v for k, v in zip(l_predictors, clf.feature_importances_)}
            
            json.dump(coefficients, f, indent=4)

        # Load individual thresholds and store the mean
        thresholds = []
        
        if step == 'gait':
            l_ids = participant_ids.L_PD_IDS + participant_ids.L_HC_IDS
        else:
            l_ids = participant_ids.L_PD_IDS 

        for subject in l_ids:
            with open(os.path.join(paths.PATH_THRESHOLDS, step, f'{classifier_name}_{subject}.txt'), 'r') as f:
                thresholds.append(float(f.read()))

        mean_threshold = np.mean(thresholds)
        with open(os.path.join(paths.PATH_THRESHOLDS, step, f'{classifier_name}.txt'), 'w') as f:
            f.write(str(mean_threshold))

        # Delete individual thresholds
        for subject in l_ids:
            os.remove(os.path.join(paths.PATH_THRESHOLDS, step, f'{classifier_name}_{subject}.txt')) 

    # Save the scaler parameters as JSON
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'var': scaler.var_.tolist(),
        'scale': scaler.scale_.tolist()
    }

    scaler_path = os.path.join(paths.PATH_SCALERS, step, f'scaler_params.json')

    if not os.path.exists(paths.PATH_SCALERS):
        os.makedirs(paths.PATH_SCALERS)
        
    with open(scaler_path, 'w') as f:
        json.dump(scaler_params, f)

    if step == 'arm_activity':       
        # Predict arm activity for the controls
        predict_controls(clf, scaler, classifier_name, l_predictors, l_predictors_scaled, step)


def predict_controls(clf, scaler, classifier_name, l_predictors, l_predictors_scaled, step):
    for subject in participant_ids.L_HC_IDS:
        df_subj_mas = pd.read_pickle(os.path.join(paths.PATH_ARM_ACTIVITY_FEATURES, f'{subject}_{descriptives.MOST_AFFECTED_SIDE}.pkl'))
        df_subj_las = pd.read_pickle(os.path.join(paths.PATH_ARM_ACTIVITY_FEATURES, f'{subject}_{descriptives.LEAST_AFFECTED_SIDE}.pkl'))
        df_subj_mas['side'] = descriptives.MOST_AFFECTED_SIDE
        df_subj_las['side'] = descriptives.LEAST_AFFECTED_SIDE
        df_subj = pd.concat([df_subj_mas, df_subj_las]).reset_index(drop=True)

        df_subj[l_predictors_scaled] = scaler.transform(df_subj[l_predictors_scaled])
        df_subj[columns.PRED_OTHER_ARM_ACTIVITY_PROBA] = clf.predict_proba(df_subj[l_predictors])[:,1]
        df_subj[columns.PRED_OTHER_ARM_ACTIVITY] = clf.predict(df_subj[l_predictors])

        windows_to_timestamps(
            subject=subject, df=df_subj,
            path_output=os.path.join(paths.PATH_ARM_ACTIVITY_PREDICTIONS, classifier_name),
            pred_proba_colname=columns.PRED_OTHER_ARM_ACTIVITY_PROBA,
            step=step
        )


def store_gait_detection(classifier):
    df = load_dataframes_directory(
        directory_path=paths.PATH_GAIT_FEATURES,
        l_ids=participant_ids.L_PD_IDS + participant_ids.L_HC_IDS
    )

    store_model_output(
        df=df,
        classifier=classifier,
        step='gait',
    )


def store_filtering_gait(classifier):
    df = load_dataframes_directory(
        directory_path=paths.PATH_ARM_ACTIVITY_FEATURES,
        l_ids=participant_ids.L_PD_IDS
    )

    store_model_output(
        df=df,
        step='arm_activity'
    )