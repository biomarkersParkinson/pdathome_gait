import json
import numpy as np
import os
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler

from pdathome.constants import classifiers, columns, descriptives, participant_ids, paths

def cv_train_test_model(subject, df, model, l_predictors, l_predictors_scale, target_column_name, 
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
    X_train = df_train[l_predictors].values
    y_train = df_train[target_column_name].values.astype(int)
    X_test = df_test[l_predictors].values
    y_test = df_test[target_column_name].values.astype(int)

    # Initialize the model
    if model == classifiers.RANDOM_FOREST:
        clf = RandomForestClassifier(
            n_estimators=100, max_features='sqrt', min_samples_split=25, max_depth=15,
            criterion='gini', bootstrap=True, oob_score=True, class_weight=class_weight, random_state=22
        )
    elif model == classifiers.LOGISTIC_REGRESSION:
        clf = LogisticRegression(
            penalty='l1', solver='saga', tol=1e-4, C=1e-2, class_weight=class_weight,
            random_state=22, n_jobs=-1
        )
        
    # Train the model
    clf.fit(X_train, y_train)

    # Predict probabilities on the test set
    df_test['true'] = y_test
    df_test[pred_proba_colname] = clf.predict_proba(X_test)[:,1]
    
    # Threshold determination for 'gait' step
    if step == 'gait':
        X_pop_train = df_train.loc[pd_train_mask, l_predictors].values
        y_pop_train = df_train.loc[pd_train_mask, target_column_name].values.astype(int)
        y_train_pred_proba_pop = clf.predict_proba(X_pop_train)[:,1]

        # set threshold to obtain at least 95% train specificity
        fpr, _, thresholds = roc_curve(y_true=y_pop_train, y_score=y_train_pred_proba_pop, pos_label=1)
        threshold_index = np.argmax(fpr >= 0.05) - 1
        classification_threshold = thresholds[threshold_index]

        df_test[pred_colname] = (df_test[pred_proba_colname] >= classification_threshold).astype(int)
    else:
        df_test[pred_colname] = clf.predict(X_test)
        classification_threshold = df_test.loc[df_test[pred_colname] == 1, pred_proba_colname].min()

    # Feature importance
    if model == classifiers.RANDOM_FOREST:
        importances = pd.Series(clf.feature_importances_, index=l_predictors)
    elif model == classifiers.LOGISTIC_REGRESSION:
        importances = pd.Series(clf.coef_[0], index=l_predictors)
    
    return df_test, classification_threshold, importances


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
        l_explode_cols += pd_specific_cols
        l_merge_ts_cols.append(columns.PRE_OR_POST)
        l_merge_points_cols.append(columns.PRE_OR_POST)
        l_dupl_cols.append(columns.PRE_OR_POST)

    if step == 'gait':
        l_subj_cols += [columns.GAIT_MAJORITY_VOTING, columns.ACTIVITY_LABEL_MAJORITY_VOTING]
        path_features = paths.PATH_GAIT_FEATURES
        l_explode_cols.append(columns.FREE_LIVING_LABEL)
        l_groupby_cols.append(columns.FREE_LIVING_LABEL)
    else:
        l_subj_cols += [columns.OTHER_ARM_ACTIVITY_MAJORITY_VOTING, columns.ARM_LABEL_MAJORITY_VOTING]
        path_features = paths.PATH_ARM_ACTIVITY_FEATURES

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
        
    df_pred_per_point.to_pickle(os.path.join(path_output, f'{subject}.pkl'))


def store_model(df, model, l_predictors, l_predictors_scale, target_column_name, path_scalers, path_classifiers, step):
    print('Storing model ...')

    if step not in ['gait', 'arm_activity']:
        raise ValueError("Step not recognized")
    
    # Determine class weight based on the step
    class_weight = None if step == 'gait' else 'balanced'

    # Standardize features based on the PD subjects    
    scaler = StandardScaler()
    df_pd = df.loc[df[columns.ID].isin(participant_ids.L_PD_IDS), l_predictors_scale]
    df_scaled = df.copy()
    df_scaled[l_predictors_scale] = scaler.fit_transform(df_pd)

    X = df[l_predictors].to_numpy()
    y = df[target_column_name].astype(int).values

    # train on all subjects and store model
    if model == classifiers.RANDOM_FOREST:
        clf = RandomForestClassifier(
            n_estimators=100, max_features='sqrt', min_samples_split=25, max_depth=15,
            criterion='gini', bootstrap=True, oob_score=True, class_weight=class_weight, random_state=22
        )
    elif model == classifiers.LOGISTIC_REGRESSION:
        clf = LogisticRegression(
            penalty='l1', solver='saga', tol=1e-4, C=1e-2, class_weight=class_weight,
            random_state=22, n_jobs=-1
        )

    # Fit the model
    clf.fit(X, y)

    # Save the model as a pickle file
    model_path = os.path.join(path_classifiers, f'clf_{model}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)

    # Save the scaler parameters as JSON
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'var': scaler.var_.tolist(),
        'scale': scaler.scale_.tolist()
    }

    scaler_path = os.path.join(path_scalers, f'scaler_params_{step}.json')
    with open(scaler_path, 'w') as f:
        json.dump(scaler_params, f)

    print(f'Model and scaler stored successfully for {step}.')