import json
import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler

from pdathome.constants import *

def cv_train_test_model(subject, df, model, l_predictors, l_predictors_scale, target_column_name, 
                        pred_proba_colname, pred_colname, step):

    # Check for valid step
    if step not in ['gait', 'arm_activity']:
        raise ValueError("Step not recognized")
    
    # Set class weight based on the step
    class_weight = None if step == 'gait' else 'balanced'

    # Split the data for training and testing
    df_train = df[df[ID_COLNAME] != subject].copy()
    df_test = df[df[ID_COLNAME] == subject].copy()
    
    # Fit a scaler on the pd participants
    pd_train_mask = df_train[ID_COLNAME].isin(L_PD_IDS)
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
    if model == RANDOM_FOREST:
        clf = RandomForestClassifier(
            n_estimators=100, max_features='sqrt', min_samples_split=25, max_depth=15,
            criterion='gini', bootstrap=True, oob_score=True, class_weight=class_weight, random_state=22
        )
    elif model == LOGISTIC_REGRESSION:
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
    if model == RANDOM_FOREST:
        importances = pd.Series(clf.feature_importances_, index=l_predictors)
    elif model == LOGISTIC_REGRESSION:
        importances = pd.Series(clf.coef_[0], index=l_predictors)
    
    return df_test, classification_threshold, importances


def windows_to_timestamps(subject, df, path_output, pred_proba_colname, step):

    # Define base columns
    l_subj_cols = ['side', 'window_nr', pred_proba_colname] + [x for x in df.columns if 'majority_voting' in x]
    l_merge_ts_cols = ['window_nr', 'side']
    l_merge_points_cols = ['time', 'side']
    l_groupby_cols = ['time', 'side']
    l_explode_cols = ['time']
    l_dupl_cols = ['time', 'side']

    # Add PD-specific columns
    if subject in L_PD_IDS:
        pd_specific_cols = ['pre_or_post', 'arm_label']
        l_subj_cols.append('pre_or_post')
        l_groupby_cols += pd_specific_cols
        l_explode_cols += pd_specific_cols
        l_merge_ts_cols.append('pre_or_post')
        l_merge_points_cols.append('pre_or_post')
        l_dupl_cols.append('pre_or_post')

    # Select relevant columns
    df = df[l_subj_cols]

    # Set path and additional columns based on step
    if step == 'gait':
        path_features = PATH_GAIT_FEATURES
        l_explode_cols.append('free_living_label')
        l_groupby_cols.append('free_living_label')
    else:
        path_features = PATH_ARM_ACTIVITY_FEATURES
    
    # Load and combine timestamps data
    df_ts_mas = pd.read_pickle(os.path.join(path_features, f'{subject}_{MOST_AFFECTED_SIDE}_ts.pkl')).assign(side=MOST_AFFECTED_SIDE)
    df_ts_las = pd.read_pickle(os.path.join(path_features, f'{subject}_{LEAST_AFFECTED_SIDE}_ts.pkl')).assign(side=LEAST_AFFECTED_SIDE)
    df_ts = pd.concat([df_ts_mas, df_ts_las], ignore_index=True)

    # Explode timestamp data for merging
    df_ts_exploded = df_ts.explode(l_explode_cols)

    # Merge the exploded data with windowed data
    df_single_points = pd.merge(left=df_ts_exploded, right=df, how='right', on=l_merge_ts_cols)
        
    # Reset index after merging
    df_single_points.reset_index(drop=True, inplace=True)

    # Group by relevant columns and calculate mean prediction probability
    df_pred_per_point = df_single_points.groupby(l_groupby_cols)[pred_proba_colname].mean().reset_index()

    # Save the final result
    output_path = os.path.join(path_output, f'{subject}_ts.pkl')
    df_pred_per_point.to_pickle(output_path)


def store_model(df, model, l_predictors, l_predictors_scale, target_column_name, path_scalers, path_classifiers, step):
    print('Storing model ...')

    if step not in ['gait', 'arm_activity']:
        raise ValueError("Step not recognized")
    
    # Determine class weight based on the step
    class_weight = None if step == 'gait' else 'balanced'

    # Standardize features based on the PD subjects    
    scaler = StandardScaler()
    df_pd = df.loc[df[ID_COLNAME].isin(L_PD_IDS), l_predictors_scale]
    df_scaled = df.copy()
    df_scaled[l_predictors_scale] = scaler.fit_transform(df_pd)

    X = df[l_predictors].to_numpy()
    y = df[target_column_name].astype(int).values

    # train on all subjects and store model
    if model == RANDOM_FOREST:
        clf = RandomForestClassifier(
            n_estimators=100, max_features='sqrt', min_samples_split=25, max_depth=15,
            criterion='gini', bootstrap=True, oob_score=True, class_weight=class_weight, random_state=22
        )
    elif model == LOGISTIC_REGRESSION:
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