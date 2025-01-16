import json
import numpy as np
import os
import pandas as pd
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, make_scorer, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from typing import Callable, List

from paradigma.config import GaitFeatureExtractionConfig, ArmActivityFeatureExtractionConfig
from paradigma.classification import ClassifierPackage

from pdathome.constants import global_constants as gc
from pdathome.load import load_dataframes_directory

warnings.filterwarnings("ignore", category=ConvergenceWarning)
        

def train_test(
    subject_id: str,
    subject_ids: List[str],
    config_class: Callable,
    classifier_names: List[str],
    target_column_name: str,
    pred_proba_colname: str,
    step: str,
    gsearch: bool,
    path_features: str,
    path_predictions: str,
    n_jobs: int = -1,
):
    # Initialize configuration and predictors
    config = config_class()
    
    predictors = list(config.d_channels_values.keys())
    predictors_scale = [x for x in predictors if 'dominant' not in x]

    df_all_subjects = load_dataframes_directory(
        directory_path=path_features,
        subject_ids=subject_ids
    )

    for classifier_name in classifier_names:
        # Train and test model
        results = cv_train_test_model(
            subject_id=subject_id,
            df=df_all_subjects,
            classifier_name=classifier_name,
            predictors=predictors,
            predictors_scale=predictors_scale,
            target_column_name=target_column_name,
            pred_proba_colname=pred_proba_colname,
            gsearch=gsearch,
            n_jobs=n_jobs,
            step=step,
        )
        df_test, classification_threshold, best_params, best_score = results

        # Save predictions for each affected side
        save_predictions(df_test, subject_id, path_predictions, classifier_name)

        # Save classification threshold
        save_threshold(
            classification_threshold, subject_id, classifier_name, step, gc.paths.PATH_THRESHOLDS
        )
        with open(os.path.join(gc.paths.PATH_THRESHOLDS, step, f'{classifier_name}_{subject_id}.txt'), 'w') as f:
            f.write(str(classification_threshold))

        # Save best hyperparameters if grid search was performed
        if gsearch:
            save_best_params(best_params, best_score, subject_id, classifier_name, step, gc.paths.PATH_THRESHOLDS)


def train_test_gait_detection(subject_id, classifier_names, gsearch=False, n_jobs=-1):
    print(f"Gait detection - Train-testing with LOSO - {subject_id} ...")
    train_test(
        subject_id=subject_id,
        subject_ids=gc.participant_ids.PD_IDS + gc.participant_ids.HC_IDS,
        config_class=GaitFeatureExtractionConfig,
        classifier_names=classifier_names,
        target_column_name=gc.columns.GAIT_MAJORITY_VOTING,
        pred_proba_colname=gc.columns.PRED_GAIT_PROBA,
        step='gait',
        gsearch=gsearch,
        path_features=gc.paths.PATH_GAIT_FEATURES,
        path_predictions=gc.paths.PATH_GAIT_PREDICTIONS,
        n_jobs=n_jobs
    )


def train_test_filtering_gait(subject_id, classifier_names, gsearch=False, n_jobs=-1):
    print(f"Filtering gait - Train-testing with LOSO - {subject_id} ...")
    if subject_id in gc.participant_ids.PD_IDS:
        train_test(
            subject_id=subject_id,
            subject_ids=gc.participant_ids.PD_IDS,
            config_class=ArmActivityFeatureExtractionConfig,
            classifier_names=classifier_names,
            target_column_name=gc.columns.NO_OTHER_ARM_ACTIVITY_MAJORITY_VOTING,
            pred_proba_colname=gc.columns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA,
            step='arm_activity',
            gsearch=gsearch,
            path_features=gc.paths.PATH_ARM_ACTIVITY_FEATURES,
            path_predictions=gc.paths.PATH_ARM_ACTIVITY_PREDICTIONS,
            n_jobs=n_jobs
        )


def cv_train_test_model(subject_id, df, classifier_name, predictors, predictors_scale, target_column_name, 
                        pred_proba_colname, step, gsearch, n_jobs=-1):

    # Check for valid step
    if step not in ['gait', 'arm_activity']:
        raise ValueError("Step not recognized")
    
    # Set class weight based on the step
    class_weight = None if step == 'gait' else 'balanced'

    # Split the data for training and testing
    df_train = df[(df[gc.columns.ID] != subject_id)].copy()
    df_test = df[df[gc.columns.ID] == subject_id].copy()
    
    # Scale the data
    pd_train_mask = df_train[gc.columns.ID].isin(gc.participant_ids.PD_IDS)
    pd_train_mask = df_train[gc.columns.ID].isin(gc.participant_ids.PD_IDS)
    df_train, df_test = scale_data(df_train, df_test, predictors_scale, pd_train_mask)

    # Prepare train and test sets
    X_train = df_train[predictors]
    y_train = df_train[target_column_name].astype(int)
    X_test = df_test[predictors]
    y_test = df_test[target_column_name].astype(int)

    # Initialize classifier
    clf, param_grid = initialize_classifier(classifier_name, class_weight, n_jobs)
        
    # Perform grid search if enabled
    if gsearch:
        randomized_search = RandomizedSearchCV(
            clf, 
            param_distributions=param_grid, 
            scoring=make_scorer(roc_auc_score), 
            n_iter=10,
            cv=len(df_train[gc.columns.ID].unique()), 
            n_jobs=n_jobs, 
            random_state=42
        )
        randomized_search.fit(X_train, y_train)
        clf = randomized_search.best_estimator_

    # Train the model
    clf.fit(X_train, y_train)

    # Predict probabilities for train and test sets
    df_train[pred_proba_colname] = clf.predict_proba(X_train)[:,1]
    df_test[pred_proba_colname] = clf.predict_proba(X_test)[:,1]
    df_test['true'] = y_test
    
    # Determine classification threshold
    classification_threshold = determine_classification_threshold(
        step, clf, df_train, predictors, target_column_name, pd_train_mask
    )
    
    # Store grid search results if applicable
    best_params, best_score = None, None
    if gsearch:
        best_params = randomized_search.best_params_
        best_score = randomized_search.best_score_

    return df_test, classification_threshold, best_params, best_score    


def store_model_output(df, classifier_name, step, n_jobs=-1):
    print(f"Storing model output at {step} step for classifier {classifier_name} ...")

    if step not in ['gait', 'arm_activity']:
        raise ValueError("Step not recognized")
    
    # Determine configurations
    config = GaitFeatureExtractionConfig() if step == 'gait' else ArmActivityFeatureExtractionConfig()
    class_weight = None if step == 'gait' else 'balanced'
    target_column_name = gc.columns.GAIT_MAJORITY_VOTING if step == 'gait' else gc.columns.NO_OTHER_ARM_ACTIVITY_MAJORITY_VOTING

    predictors = list(config.d_channels_values.keys())
    predictors_scaled = [x for x in predictors if 'dominant' not in x]

    # Standardize features based on the PD subjects    
    scaler = StandardScaler()
    df_pd = df.loc[df[gc.columns.ID].isin(gc.participant_ids.PD_IDS), predictors_scaled]    
    scaler.fit(df_pd)

    df_train = df.copy()
    df_train[predictors_scaled] = scaler.transform(df_train[predictors_scaled])

    X = df_train[predictors]
    y = df_train[target_column_name].astype(int)

    # Initialize classifier
    clf, _ = initialize_classifier(classifier_name, class_weight, n_jobs)
    
    # Fit the model
    clf.fit(X, y)

    # Remove decision function matrix to save memory
    if classifier_name == gc.classifiers.RANDOM_FOREST:
        del clf.oob_decision_function_

    classification_threshold = determine_classification_threshold(step, clf, df_train, predictors, target_column_name, pd_train_mask=df_pd.index)

    # Save model, coefficients, and scaler
    clf_package = ClassifierPackage(
        classifier=clf,
        scaler=scaler,
        threshold=classification_threshold
    )

    clf_filepath = os.path.join(gc.paths.PATH_CLASSIFIERS, step, f'{classifier_name}.pkl')
    clf_package.save(filepath=clf_filepath)

    # Special case for arm activity: predict control group
    if step == 'arm_activity':
        predict_controls(clf, scaler, classifier_name, predictors, predictors_scaled)


def predict_controls(clf, scaler, classifier_name, predictors, predictors_scaled):
    """Predict arm activity for control group and save results."""
    for subject_id in gc.participant_ids.HC_IDS:
        df_mas = pd.read_parquet(os.path.join(gc.paths.PATH_ARM_ACTIVITY_FEATURES, f'{subject_id}_{gc.descriptives.MOST_AFFECTED_SIDE}.parquet'))
        df_las = pd.read_parquet(os.path.join(gc.paths.PATH_ARM_ACTIVITY_FEATURES, f'{subject_id}_{gc.descriptives.LEAST_AFFECTED_SIDE}.parquet'))
        df_mas['affected_side'] = gc.descriptives.MOST_AFFECTED_SIDE
        df_las['affected_side'] = gc.descriptives.LEAST_AFFECTED_SIDE
        df_test = pd.concat([df_mas, df_las]).reset_index(drop=True)

        df_test[predictors_scaled] = scaler.transform(df_test[predictors_scaled])
        df_test[gc.columns.PRED_NO_OTHER_ARM_ACTIVITY_PROBA] = clf.predict_proba(df_test[predictors])[:,1]
       
        save_predictions(df_test, subject=subject_id, path_predictions=gc.paths.PATH_ARM_ACTIVITY_PREDICTIONS, classifier_name=classifier_name)



def initialize_classifier(classifier_name, class_weight, n_jobs):
    """Initialize the classifier and return its parameter grid."""
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
    else:
        raise ValueError("Unsupported classifier")
    return clf, param_grid


def scale_data(df_train, df_test, predictors_scale, pd_train_mask):
    """Scale training and testing data using StandardScaler."""
    scaler = StandardScaler()
    scaler.fit(df_train.loc[pd_train_mask, predictors_scale])
    df_train[predictors_scale] = scaler.transform(df_train[predictors_scale])
    df_test[predictors_scale] = scaler.transform(df_test[predictors_scale])
    return df_train, df_test


def determine_classification_threshold(step, clf, df_train, predictors, target_column_name, pd_train_mask):
    """Determine the classification threshold based on specificity or Youden's J."""
    if step == 'gait':
        X_pop_train = df_train.loc[pd_train_mask, predictors]
        y_pop_train = df_train.loc[pd_train_mask, target_column_name].astype(int)
        y_train_pred_proba_pop = clf.predict_proba(X_pop_train)[:, 1]

        # Set threshold for at least 95% train specificity
        fpr, _, thresholds = roc_curve(y_true=y_pop_train, y_score=y_train_pred_proba_pop, pos_label=1)
        threshold_index = np.argmax(fpr >= 0.05) - 1
        return thresholds[threshold_index]
    else:
        subject_thresholds = []
        for train_subject_id in df_train[gc.columns.ID].unique():
            X_subject = df_train[df_train[gc.columns.ID] == train_subject_id][predictors]
            y_subject = df_train[df_train[gc.columns.ID] == train_subject_id][target_column_name].astype(int)
            y_subject_pred_proba = clf.predict_proba(X_subject)[:, 1]

            fpr, tpr, thresholds = roc_curve(y_true=y_subject, y_score=y_subject_pred_proba, pos_label=1)
            youden_j = tpr - fpr
            threshold_index = np.argmax(youden_j)
            subject_thresholds.append(thresholds[threshold_index])

        return np.mean(subject_thresholds)


def store_gait_detection(classifier_names, n_jobs=-1):
    df = load_dataframes_directory(
        directory_path=gc.paths.PATH_GAIT_FEATURES,
        subject_ids=gc.participant_ids.PD_IDS + gc.participant_ids.HC_IDS
    )

    for classifier in classifier_names:
        store_model_output(
            df=df,
            classifier_name=classifier,
            step='gait',
            n_jobs=n_jobs
        )


def store_filtering_gait(classifier_names, n_jobs=-1):
    df = load_dataframes_directory(
        directory_path=gc.paths.PATH_ARM_ACTIVITY_FEATURES,
        subject_ids=gc.participant_ids.PD_IDS,
    )

    for classifier in classifier_names:
        store_model_output(
            df=df,
            classifier_name=classifier,
            step='arm_activity',
            n_jobs=n_jobs
        )


def save_predictions(df_test, subject, path_predictions, classifier_name):
    path_output = os.path.join(path_predictions, classifier_name)
    os.makedirs(path_output, exist_ok=True)  # Ensure output directory exists

    for affected_side in [gc.descriptives.MOST_AFFECTED_SIDE, gc.descriptives.LEAST_AFFECTED_SIDE]:
        df_aff_side = df_test[df_test[gc.columns.AFFECTED_SIDE] == affected_side]
        output_file = os.path.join(path_output, f'{subject}_{affected_side}.parquet')
        df_aff_side.reset_index(drop=True).to_parquet(output_file)


# def save_model(clf, step, classifier_name):
#     """Save the trained classifier as a pickle file."""
#     model_path = os.path.join(gc.paths.PATH_CLASSIFIERS, step, f'{classifier_name}.pkl')
#     os.makedirs(os.path.dirname(model_path), exist_ok=True)
#     with open(model_path, 'wb') as f:
#         pickle.dump(clf, f)


def save_threshold(classification_threshold, subject, classifier_name, step, path_thresholds):
    output_dir = os.path.join(path_thresholds, step)
    os.makedirs(output_dir, exist_ok=True)

    threshold_file = os.path.join(output_dir, f'{classifier_name}_{subject}.txt')
    with open(threshold_file, 'w') as f:
        f.write(str(classification_threshold))


def save_scaler_params(scaler, predictors_scaled, step, classifier_name):
    """Save the scaling parameters for reproducibility."""
    scaler_params = {
        'features': predictors_scaled,
        'mean': scaler.mean_.tolist(),
        'var': scaler.var_.tolist(),
        'scale': scaler.scale_.tolist(),
    }
    scaler_path = os.path.join(gc.paths.PATH_SCALERS, step, f'{classifier_name}_scaler_params.json')
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, 'w') as f:
        json.dump(scaler_params, f)

        
def save_best_params(best_params, best_score, subject, classifier_name, step, path_thresholds):
    best_params['score'] = best_score
    output_dir = os.path.join(path_thresholds, step)
    os.makedirs(output_dir, exist_ok=True)

    params_file = os.path.join(output_dir, f'{classifier_name}_{subject}_params.json')
    with open(params_file, 'w') as f:
        json.dump(best_params, f, indent=4)