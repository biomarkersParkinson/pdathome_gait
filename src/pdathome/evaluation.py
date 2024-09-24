import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pdathome.constants import classifiers, PlotParameters

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
    

def plot_coefs(l_coefs, classifier, color=PlotParameters().COLOR_PALETTE_FIRST_COLOR, figsize=(10,20)):
    if classifier==classifiers.LOGISTIC_REGRESSION:
        coefs = 'coefficient'
    elif classifier==classifiers.RANDOM_FOREST:
        coefs = 'impurity_score'

    coefs_str = [x.replace('[', '').replace(']', '').replace('dtype: float64, ', '') for x in l_coefs.split('\n')]

    l_dfs = []
    for j in range(len(coefs_str)):
        l_strs = [x for x in coefs_str[j].split(' ') if len(x) > 0]
        if l_strs[0] != 'dtype:':
            l_dfs.append(pd.DataFrame([l_strs[0], float(l_strs[1])]).T)

    df_coefs_lr = pd.concat(l_dfs, axis=0).reset_index(drop=True)
    df_coefs_lr.columns = ['feature', coefs]

    sorter = list(df_coefs_lr.groupby('feature')[coefs].mean().sort_values(ascending=False).keys())
    df_coefs_lr = df_coefs_lr.set_index('feature').loc[sorter].reset_index()

    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(data=df_coefs_lr, y='feature', x=coefs, orient='h', color=color)
    ax.set_xlabel(coefs.replace('_', ' ').capitalize())
    ax.set_ylabel("Feature")

    fig.tight_layout()

    plt.show()