import json
import numpy as np
import os
import pandas as pd
import pickle
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, make_scorer, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from typing import Callable, List

from paradigma.gait.gait_analysis_config import GaitFeatureExtractionConfig, ArmActivityFeatureExtractionConfig

from pdathome.constants import global_constants as gc
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
    step: str,
    gsearch: bool,
    path_features: str,
    path_predictions: str,
    n_jobs: int = -1,
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

    for classifier_name in l_classifiers:

        # Train and test model
        df_test, classification_threshold, best_params, best_score = cv_train_test_model(
            subject=subject,
            df=df_all_subjects,
            classifier_name=classifier_name,
            l_predictors=l_predictors,
            l_predictors_scale=l_predictors_scale,
            target_column_name=target_column_name,
            pred_proba_colname=pred_proba_colname,
            gsearch=gsearch,
            n_jobs=n_jobs,
            step=step,
        )

        # Save predictions
        windows_to_timestamps(
            subject=subject, df=df_test,
            path_output=os.path.join(path_predictions, classifier_name),
            pred_proba_colname=pred_proba_colname,
            step=step
        )

        with open(os.path.join(gc.paths.PATH_THRESHOLDS, step, f'{classifier_name}_{subject}.txt'), 'w') as f:
            f.write(str(classification_threshold))

        if gsearch:
            best_params['score'] = best_score
            with open(os.path.join(gc.paths.PATH_THRESHOLDS, step, f'{classifier_name}_{subject}_params.json'), 'w') as f:
                json.dump(best_params, f, indent=4)


def train_test_gait_detection(subject, l_classifiers, gsearch=False, n_jobs=-1):
    print(f"Gait detection - Train-testing with LOSO - {subject} ...")
    train_test(
        subject=subject,
        l_ids=gc.participant_ids.L_PD_IDS + gc.participant_ids.L_HC_IDS,
        config_class=GaitFeatureExtractionConfig,
        l_classifiers=l_classifiers,
        target_column_name=gc.columns.GAIT_MAJORITY_VOTING,
        pred_proba_colname=gc.columns.PRED_GAIT_PROBA,
        step='gait',
        gsearch=gsearch,
        path_features=gc.paths.PATH_GAIT_FEATURES,
        path_predictions=gc.paths.PATH_GAIT_PREDICTIONS,
        n_jobs=n_jobs
    )


def train_test_filtering_gait(subject, l_classifiers, gsearch=False, n_jobs=-1):
    print(f"Filtering gait - Train-testing with LOSO - {subject} ...")
    if subject in gc.participant_ids.L_PD_IDS:
        train_test(
            subject=subject,
            l_ids=gc.participant_ids.L_PD_IDS,
            config_class=ArmActivityFeatureExtractionConfig,
            l_classifiers=l_classifiers,
            target_column_name=gc.columns.OTHER_ARM_ACTIVITY_MAJORITY_VOTING,
            pred_proba_colname=gc.columns.PRED_OTHER_ARM_ACTIVITY_PROBA,
            step='arm_activity',
            gsearch=gsearch,
            path_features=gc.paths.PATH_ARM_ACTIVITY_FEATURES,
            path_predictions=gc.paths.PATH_ARM_ACTIVITY_PREDICTIONS,
            n_jobs=n_jobs
        )


def cv_train_test_model(subject, df, classifier_name, l_predictors, l_predictors_scale, target_column_name, 
                        pred_proba_colname, step, gsearch, n_jobs=-1):

    # Check for valid step
    if step not in ['gait', 'arm_activity']:
        raise ValueError("Step not recognized")
    
    # Set class weight based on the step
    class_weight = None if step == 'gait' else 'balanced'

    # Split the data for training and testing
    df_train = df[(df[gc.columns.ID] != subject) & (df[gc.columns.PRE_OR_POST]=='pre')].copy()
    df_test = df[df[gc.columns.ID] == subject].copy()
    
    # Fit a scaler on the pd participants
    pd_train_mask = df_train[gc.columns.ID].isin(gc.participant_ids.L_PD_IDS)
    scaler = StandardScaler()

    # Scale the data
    scaler.fit(df_train.loc[pd_train_mask, l_predictors_scale])
    df_train[l_predictors_scale] = scaler.transform(df_train[l_predictors_scale])
    df_test[l_predictors_scale] = scaler.transform(df_test[l_predictors_scale])

    # Define train and test sets
    X_train = df_train[l_predictors]
    y_train = df_train[target_column_name].astype(int)
    X_test = df_test[l_predictors]
    y_test = df_test[target_column_name].astype(int)

    # Initialize the model
    if classifier_name == gc.classifiers.RANDOM_FOREST:
        clf = RandomForestClassifier(
            **gc.classifiers.RANDOM_FOREST_HYPERPARAMETERS,
            class_weight=class_weight,
            n_jobs=n_jobs
        )
        param_grid = gc.classifiers.RANDOM_FOREST_PARAM_GRID
    elif classifier_name == gc.classifiers.LOGISTIC_REGRESSION:
        clf = LogisticRegression(
            **gc.classifiers.LOGISTIC_REGRESSION_HYPERPARAMETERS,
            class_weight=class_weight,
            n_jobs=n_jobs,
        )
        param_grid = gc.classifiers.LOGISTIC_REGRESSION_PARAM_GRID
        
    if gsearch:
        # Perform Grid Search with cross-validation
        randomized_search = RandomizedSearchCV(
            clf, param_distributions=param_grid, scoring=make_scorer(roc_auc_score), n_iter=10,
            cv=len(df_train[gc.columns.ID].unique()), n_jobs=n_jobs, random_state=42
        )
        randomized_search.fit(X_train, y_train)

        # Retrieve the best model from Grid Search
        clf = randomized_search.best_estimator_

    clf.fit(X_train, y_train)

    # Predict probabilities on the test set
    df_test['true'] = y_test

    df_train[pred_proba_colname] = clf.predict_proba(X_train)[:,1]
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
    else:
        # For non-gait steps, calculate a threshold per participant in training set and average them
        subject_thresholds = []
        for train_subject in df_train[gc.columns.ID].unique():
            X_subject = df_train[df_train[gc.columns.ID] == train_subject][l_predictors]
            y_subject = df_train[df_train[gc.columns.ID] == train_subject][target_column_name].astype(int)
            y_subject_pred_proba = clf.predict_proba(X_subject)[:, 1]

            fpr, tpr, thresholds = roc_curve(y_true=y_subject, y_score=y_subject_pred_proba, pos_label=1)
            youden_j = tpr - fpr
            threshold_index = np.argmax(youden_j)
            subject_thresholds.append(thresholds[threshold_index])

        # Average the thresholds across subjects
        classification_threshold = np.mean(subject_thresholds)
    
    if gsearch:
        # Store the grid search results
        best_params = randomized_search.best_params_
        best_score = randomized_search.best_score_
    else:
        best_params = None
        best_score = None

    return df_test, classification_threshold, best_params, best_score


def windows_to_timestamps(subject, df, path_output, pred_proba_colname, step):

    if step not in ['gait', 'arm_activity']:
        raise ValueError("Step not recognized")
    
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    # Define base gc.columns
    l_subj_cols = [gc.columns.SIDE, gc.columns.WINDOW_NR, pred_proba_colname]
    l_merge_ts_cols = [gc.columns.WINDOW_NR, gc.columns.SIDE]
    l_groupby_cols = [gc.columns.TIME, gc.columns.SIDE]
    l_explode_cols = [gc.columns.TIME]

    if step == 'gait':
        path_features = gc.paths.PATH_GAIT_FEATURES
        l_subj_cols += [gc.columns.GAIT_MAJORITY_VOTING, gc.columns.ACTIVITY_LABEL_MAJORITY_VOTING]
    elif step == 'arm_activity':
        path_features = gc.paths.PATH_ARM_ACTIVITY_FEATURES
        if subject in gc.participant_ids.L_PD_IDS:
            l_subj_cols += [gc.columns.OTHER_ARM_ACTIVITY_MAJORITY_VOTING, gc.columns.ARM_LABEL_MAJORITY_VOTING]

    # Select relevant gc.columns
    df = df[l_subj_cols]     

    df.to_pickle(os.path.join(path_output, f'{subject}_df.pkl'))
    
    # Load and combine timestamps data
    df_ts_mas = pd.read_pickle(os.path.join(path_features, f'{subject}_{gc.descriptives.MOST_AFFECTED_SIDE}_ts.pkl')).assign(side=gc.descriptives.MOST_AFFECTED_SIDE)
    df_ts_las = pd.read_pickle(os.path.join(path_features, f'{subject}_{gc.descriptives.LEAST_AFFECTED_SIDE}_ts.pkl')).assign(side=gc.descriptives.LEAST_AFFECTED_SIDE)

    df_ts_mas = df_ts_mas.reset_index(drop=True)
    df_ts_las = df_ts_las.reset_index(drop=True)

    df_ts = pd.concat([df_ts_mas, df_ts_las], ignore_index=True)

    # Explode timestamp data for merging
    df_ts_exploded = df_ts.explode(l_explode_cols)

    # Merge the exploded data with windowed data
    df_single_points = pd.merge(left=df_ts_exploded, right=df, how='left', on=l_merge_ts_cols)
        
    # Reset index after merging
    df_single_points.reset_index(drop=True, inplace=True)

    # Group by relevant gc.columns and calculate mean prediction probability
    df_pred_per_point = df_single_points.groupby(l_groupby_cols)[pred_proba_colname].mean().reset_index()

    # Save the final result
    save_to_pickle(
        df=df_pred_per_point,
        path=path_output,
        filename=f'{subject}.pkl'
    )


def store_model_output(df, classifier_name, step, n_jobs=-1):
    print()
    print(f"Storing model output at {step} step for classifier {classifier_name} ...")

    if step not in ['gait', 'arm_activity']:
        raise ValueError("Step not recognized")
    
    # Determine class weight based on the step
    if step == 'gait':
        class_weight = None
        target_column_name = gc.columns.GAIT_MAJORITY_VOTING
        config = GaitFeatureExtractionConfig()
    else:
        class_weight = 'balanced'
        target_column_name = gc.columns.OTHER_ARM_ACTIVITY_MAJORITY_VOTING
        config = ArmActivityFeatureExtractionConfig()

    l_predictors = list(config.d_channels_values.keys())
    l_predictors_scaled = [x for x in l_predictors if 'dominant' not in x]

    # Standardize features based on the PD subjects    
    scaler = StandardScaler()
    df_pd = df.loc[df[gc.columns.ID].isin(gc.participant_ids.L_PD_IDS), l_predictors_scaled]
    df_scaled = df.copy()
    
    scaler.fit(df_pd[l_predictors_scaled])
    df_scaled[l_predictors_scaled] = scaler.transform(df_scaled[l_predictors_scaled])

    X = df_scaled[l_predictors]
    y = df_scaled[target_column_name].astype(int)

    # train on all subjects and store model
    if classifier_name == gc.classifiers.RANDOM_FOREST:
        clf = RandomForestClassifier(
            **gc.classifiers.RANDOM_FOREST_HYPERPARAMETERS,
            class_weight=class_weight,
            n_jobs=n_jobs
        )
    elif classifier_name == gc.classifiers.LOGISTIC_REGRESSION:
        clf = LogisticRegression(
            **gc.classifiers.LOGISTIC_REGRESSION_HYPERPARAMETERS,
            class_weight=class_weight,
            n_jobs=n_jobs
        )
    else:
        raise ValueError("Classifier not recognized")

    # Fit the model
    clf.fit(X, y)

    # Save the model as a pickle file
    model_path = os.path.join(gc.paths.PATH_CLASSIFIERS, step, f'{classifier_name}.pkl')

    if not os.path.exists(gc.paths.PATH_CLASSIFIERS):
        os.makedirs(gc.paths.PATH_CLASSIFIERS)

    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)

    # Save coefficients as json
    with open(os.path.join(gc.paths.PATH_COEFFICIENTS, step, f'{classifier_name}.json'), 'w') as f:
        if type(clf).__name__.lower() == 'logisticregression': 
            coefficients = {k: v for k, v in zip(l_predictors, clf.coef_[0])}
        elif type(clf).__name__.lower() == 'randomforestclassifier':
            coefficients = {k: v for k, v in zip(l_predictors, clf.feature_importances_)}
        
        json.dump(coefficients, f, indent=4)

    # Save the scaler parameters as JSON
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'var': scaler.var_.tolist(),
        'scale': scaler.scale_.tolist()
    }

    scaler_path = os.path.join(gc.paths.PATH_SCALERS, step, 'scaler_pdh_params.json')

    if not os.path.exists(gc.paths.PATH_SCALERS):
        os.makedirs(gc.paths.PATH_SCALERS)
        
    with open(scaler_path, 'w') as f:
        json.dump(scaler_params, f)

    # Load individual thresholds and store the mean
    if step == 'gait':
         l_ids = gc.participant_ids.L_PD_IDS + gc.participant_ids.L_HC_IDS
    else:
        l_ids = gc.participant_ids.L_PD_IDS

    if os.path.exists(os.path.join(gc.paths.PATH_THRESHOLDS, step, f'{classifier_name}_hbv002.txt')):
        thresholds = []
        for subject in l_ids:
            with open(os.path.join(gc.paths.PATH_THRESHOLDS, step, f'{classifier_name}_{subject}.txt'), 'r') as f:
                thresholds.append(float(f.read()))

        mean_threshold = np.mean(thresholds)
    
        with open(os.path.join(gc.paths.PATH_THRESHOLDS, step, f'{classifier_name}.txt'), 'w') as f:
            f.write(str(mean_threshold))

        # Delete individual thresholds
        for subject in l_ids:
            os.remove(os.path.join(gc.paths.PATH_THRESHOLDS, step, f'{classifier_name}_{subject}.txt')) 

    if step == 'arm_activity':       
        # Predict arm activity for the controls
        predict_controls(clf, scaler, classifier_name, l_predictors, l_predictors_scaled, step)


def predict_controls(clf, scaler, classifier_name, l_predictors, l_predictors_scaled, step):
    for subject in gc.participant_ids.L_HC_IDS:
        df_subj_mas = pd.read_pickle(os.path.join(gc.paths.PATH_ARM_ACTIVITY_FEATURES, f'{subject}_{gc.descriptives.MOST_AFFECTED_SIDE}.pkl'))
        df_subj_las = pd.read_pickle(os.path.join(gc.paths.PATH_ARM_ACTIVITY_FEATURES, f'{subject}_{gc.descriptives.LEAST_AFFECTED_SIDE}.pkl'))
        df_subj_mas['side'] = gc.descriptives.MOST_AFFECTED_SIDE
        df_subj_las['side'] = gc.descriptives.LEAST_AFFECTED_SIDE
        df_subj = pd.concat([df_subj_mas, df_subj_las]).reset_index(drop=True)

        df_subj[l_predictors_scaled] = scaler.transform(df_subj[l_predictors_scaled])
        df_subj[gc.columns.PRED_OTHER_ARM_ACTIVITY_PROBA] = clf.predict_proba(df_subj[l_predictors])[:,1]
        df_subj[gc.columns.PRED_OTHER_ARM_ACTIVITY] = clf.predict(df_subj[l_predictors])

        windows_to_timestamps(
            subject=subject, df=df_subj,
            path_output=os.path.join(gc.paths.PATH_ARM_ACTIVITY_PREDICTIONS, classifier_name),
            pred_proba_colname=gc.columns.PRED_OTHER_ARM_ACTIVITY_PROBA,
            step=step
        )


def store_gait_detection(l_classifiers, n_jobs=-1):
    df = load_dataframes_directory(
        directory_path=gc.paths.PATH_GAIT_FEATURES,
        l_ids=gc.participant_ids.L_PD_IDS + gc.participant_ids.L_HC_IDS
    )

    for classifier in l_classifiers:
        store_model_output(
            df=df,
            classifier_name=classifier,
            step='gait',
            n_jobs=n_jobs
        )


def store_filtering_gait(l_classifiers, n_jobs=-1):
    df = load_dataframes_directory(
        directory_path=gc.paths.PATH_ARM_ACTIVITY_FEATURES,
        l_ids=gc.participant_ids.L_PD_IDS,
    )

    for classifier in l_classifiers:
        store_model_output(
            df=df,
            classifier_name=classifier,
            step='arm_activity',
            n_jobs=n_jobs
        )