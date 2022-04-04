# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 17:46:19 2022

This file is run after Dissertation_Publish_03.py script
This script runs PCA, Does Plotting, logistic regression, random forest models on the training data and 
tests the model performance on testing data

@author: nisha
"""
#%%# Library imports
import seaborn as sns
import pandas as pd
import numpy as np
from numpy import arange
from numpy import argmax
import scipy.stats as stats
from scipy.stats import ks_2samp
import seaborn as sns
import datetime
import calendar
import matplotlib.pyplot as plt
from matplotlib import rcParams
import time
import math
from bokeh.plotting import figure, output_file, save
from bokeh.io import show
from bokeh.models import ColumnDataSource, CDSView, GroupFilter, HoverTool
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix, plot_roc_curve, roc_auc_score,roc_curve, average_precision_score, precision_recall_curve, PrecisionRecallDisplay
from collections import Counter
import pickle
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn.utils.multiclass import unique_labels
from pprint import pprint
from imblearn.ensemble import BalancedRandomForestClassifier
#%%# Load Pickled training data and testing data with features
unpickled_NPI_SUMM_2016 = pd.read_pickle("C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/pickled_list_NPI_Summ_2016.pickle")
unpickled_NPI_SUMM_2017 = pd.read_pickle("C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/pickled_list_NPI_Summ_2017.pickle")
#%%# PICT Case info load
PICT=pd.read_table('file:///C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/PICT DeID Case Info.txt',\
                      sep='\t',\
                      dtype={'Case_Assign_User_Area_Desc':'str','Case_Dtl_Detection_Source_Desc':'str','Case_Dtl_Investigation_Cat_Desc':'str',\
                             'Case_Created_Date':'str','Case_Completed_Date	':'str',\
                             'Case_Dtl_Close_Reason_Desc':'str','Case_Recovery_Est_Recovery_Amt_T':'float64',\
                                    'Case_Recovery_Act_Recovery_Amt_T':'float64','Pharmacy_Type_Indicator':'str','pharmtyp':'str',\
                                           'oigexcludelist':'str','deid_pharmnpi':'str'})
PICT.head()
PICT.columns
# Drop rows with deid pharmnpi missing value
pictF= PICT[PICT['deid_pharmnpi'].notna()]
# Removing cases where no action was taken
exclCase = ['C21- No Action Taken','C3- Insufficient Evidence', 'C27 - No Findings']
pictF=pictF[~pictF['Case_Dtl_Close_Reason_Desc'].isin(exclCase)]
# Replace missing Case_Dtl_Close_Reason_Desc with 'Ongoing' case
pictF['Case_Dtl_Close_Reason_Desc'] = pictF['Case_Dtl_Close_Reason_Desc'].replace(np.nan, 'Case Ongoing', regex=True)
ActPICTCase=pictF['deid_pharmnpi'].value_counts() #221 NPIs have cases in CT, some have more than 1 case
#%%# PICT case descriptive stats
def uniqcnt(x):
    return x.nunique()

aggrCaseClose = {
    'deid_pharmnpi':['count',uniqcnt]
}

pictF.groupby(['Case_Dtl_Close_Reason_Desc']).agg(aggrCaseClose)
pictF.groupby(['Case_Assign_User_Area_Desc']).agg(aggrCaseClose)
#%%# Assign NPI Fraud Confidence (FC) Score depending on case assigned user area
# If Intake, then FC=0.1
# If MPI or DMS, then FC =0.5
# If A/R or Sanctions, then FC = 1

FC1 = ['AR - Accounts Receivable', 'SANCTION - Sanctions']
FC2 = ['DMS - Division of Medical Services', 'MPI - Medicaid Provider Integrity']
FCs = [
    (pictF['Case_Assign_User_Area_Desc'].isin(FC1)),
    (pictF['Case_Assign_User_Area_Desc'].isin(FC2))
    ]

# create a list of the values we want to assign for each condition
values = [1.0,0.5]

pictF['NPI_FrdConf']=np.select(FCs, values, default=0.1)
NPI_FrdConf=pictF.groupby(by=['deid_pharmnpi'])['NPI_FrdConf'].sum()

#%%# Merge Case Tracker data with NPI Summary data - Trg
df_trn=unpickled_NPI_SUMM_2016.join(NPI_FrdConf,how='left')
df_val=unpickled_NPI_SUMM_2017.join(NPI_FrdConf,how='left')
# Checks
df_trn['NPI_FrdConf'].count() # 192 NPIs out of 4982 are fraud
df_val['NPI_FrdConf'].count() # 211 NPIs out of 5007 are fraud
# Fill NANs with 0 and any value greater than 0 with 1 for fraud
df_trn['NPI_FrdBin'] = np.where((df_trn['NPI_FrdConf'] >0),1,0)
df_val['NPI_FrdBin'] = np.where((df_val['NPI_FrdConf'] >0),1,0)
#%%# Logistic regression with principal components - Prepare the dataset
X=df_trn.drop(['NPI_FrdConf','NPI_FrdBin'],1)
y=df_trn['NPI_FrdBin']
X.shape
y.shape

X_val=df_val.drop(['NPI_FrdConf','NPI_FrdBin'],1)
y_val=df_val['NPI_FrdBin']
#%%# Before performing PCA, normalize/scale the features and also look at the features contributing to the first few components
# sns.set(rc={'figure.figsize':(18,10)},font_scale=1.0,style="white")
scaler=MinMaxScaler()
NPI_Summ_Scld=scaler.fit_transform(X)
NPI_Summ_Scld_val=scaler.transform(X_val)

pc=15 # No. of PCs we want
       
pca100 = PCA(n_components=pc)
my100pc=pca100.fit_transform(NPI_Summ_Scld)
my100pc_val=pca100.transform(NPI_Summ_Scld_val)

pca100.explained_variance_ratio_
eigenvalues = pca100.explained_variance_
eigenvec=pca100.components_
percent_variance=np.round(pca100.explained_variance_ratio_*100, decimals=2)
percent_variance.sum() #85.41999% variance explained by 15 principal components

# Create X tick labels
col_nms=[]
for i in range(pc):
        j=i+1
        mystr='PC'+str(j)
        col_nms.append(mystr)
font = {
        'weight' : 'bold',
        'size'   : 16}

plt.rc('font', **font)

plt.figure(figsize=(18,10))
plt.xticks(rotation=0)
plt.bar(x= range(0,pc), height=percent_variance,tick_label=col_nms, color='#98d687')
plt.ylabel('Variance Explained in %', fontsize=16)
plt.xlabel('Principal Components', fontsize=16)
plt.title('15 Component - PCA Scree Plot', fontsize=16)
plt.show()
plt.savefig('C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Model/Plots/PCA_Dissertation1.jpg')
plt.clf()
# Plot of explained variance with all 176 components
plt.plot(np.cumsum(pca100.explained_variance_ratio_)*100)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance in %');
plt.savefig('C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Model/Plots/PCA_Dissertation2.jpg')
plt.rcParams.update(plt.rcParamsDefault)
rcParams.update({'figure.autolayout': True})
#%%# To find feature contributions from the first 15 PCs
def get_float_list(range_max:int, div:int=100) -> list:
    """ To get 0 -> 1, range_max must be same order of mag as div """
    return [float(x)/div for x in range(int(range_max))]

def get_colorcycle(colordict:dict):
    """ Subset cnames with a string match and get a color cycle for plotting """
    return cycle(list(colordict.keys()))

def get_colordict(filter_:str='dark') -> dict:
    """ return dictionary of colornames by filter """
    return dict((k, v) for k, v in cnames.items() if filter_ in k)

def pca_report_interactive(X, scale_X:bool=True, save_plot:bool=False):
    """
    X:          input data matrix
    scale_X:    determine whether to rescale X (StandardScaler) [default: True, X is not prescaled
    save_plot:  save plot to file (html) and not show
    """

    # calculate mean and var
    X_mean, X_var = X.mean(), X.var()
    print('\n*--- PCA Report ---*\n')
    print(f'X mean:\t\t{X_mean:.3f}\nX variance:\t{X_var:.3f}')

    if scale_X:
        # rescale and run PCA
        print("\n...Rescaling data...\n")
        # scaler = StandardScaler()
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_s_mean, X_s_var = X_scaled.mean(), X_scaled.var()
        print(f'X_scaled mean:\t\t{np.round(X_s_mean):.3f}')
        print(f'X_scaled variance:\t{np.round(X_s_var):.3f}\n')
        pca_ = PCA().fit(X_scaled)
        X_pca = PCA().fit_transform(X)
    else:
        # run PCA directly
        print("...Assuming data is properly scaled...")
        pca_ = PCA().fit(X)
        X_pca = PCA().fit_transform(X)
            
    # Get cumulative explained variance for each dimension
    pca_evr = pca_.explained_variance_ratio_
    cumsum_ = np.cumsum(pca_evr)
    
    # Get dimensions where var >= 85% and values for variance at 2D, 3D
    dim_85 = np.argmax(cumsum_ >= 0.85) + 1
    twoD = np.round(cumsum_[1], decimals=3)*100 
    threeD = np.round(cumsum_[2], decimals=3)*100
    instances_, dims_ =  X.shape
    
    # check shape of X
    if dims_ > instances_:
        print("WARNING: number of features greater than number of instances.")
        dimensions = list(range(1, instances_+1))
    else:
        dimensions = list(range(1, dims_+1))
    
    # Print report
    print("\n -- Summary --")
    print(f"I can reduce from {dims_} to {dim_85} dimensions while retaining 85% of variance.")
    print(f"2 principal components explain {twoD:.2f}% of variance.")
    print(f"3 principal components explain {threeD:.2f}% of variance.")
    
    """ - Plotting - """
    # Create custom HoverTool -- we'll name each ROC curve 'ROC' so we only see info on hover there
    hover_ = HoverTool(names=['PCA'], tooltips=[("dimensions", "@x_dim"), 
                                                ("cumulative variance", "@y_cumvar"),
                                                ("explained variance", "@y_var")])
    p_tools = [hover_, 'crosshair', 'zoom_in', 'zoom_out', 'save', 'reset', 'tap', 'box_zoom']

    # insert 0 at beginning for cleaner plotting
    cumsum_plot = np.insert(cumsum_, 0, 0) 
    pca_evr_plot = np.insert(pca_evr, 0, 0)
    dimensions_plot = np.insert(dimensions, 0, 0)

    """
    ColumnDataSource
    - a special type in Bokeh that allows you to store data for plotting
    - store data as dict (key:list)
    - to plot two keys against one another, make sure they're the same length!
    - below:
        x_dim    # of dimensions (length = # of dimensions)
        y_cumvar # cumulative variance (length = # of dimensions)
        var_95   # y = 0.95 (length = # of dimensions)
        zero_one # list of 0 to 1
        twoD     # x = 2 
        threeD   # x = 3 
    """ 
    
    # get sources
    source_PCA = ColumnDataSource(data=dict(x_dim = dimensions_plot,y_cumvar = cumsum_plot, y_var = pca_evr_plot))    
    # source_var95 = ColumnDataSource(data=dict(var95_x = [dim_95]*96, var95_y = get_float_list(96)))
    source_var85 = ColumnDataSource(data=dict(var95_x = [dim_85]*86, var95_y = get_float_list(86)))
    source_twoD = ColumnDataSource(data=dict(twoD_x = [2]*(int(twoD)+1), twoD_y = get_float_list(twoD+1)))
    source_threeD = ColumnDataSource(data=dict(threeD_x = [3]*(int(threeD)+1), threeD_y = get_float_list(threeD+1)))

    """ PLOT """
    # set up figure and add axis labels
    p = figure(title='PCA Analysis', tools=p_tools)
    p.xaxis.axis_label = f'N of {dims_} Principal Components' 
    p.yaxis.axis_label = 'Variance Explained (per PC & Cumulative)'
    
    # add reference lines: y=0.95, x=2, x=3
    p.line('twoD_x', 'twoD_y', line_width=0.5, line_dash='dotted', color='#435363', source=source_twoD) # x=2
    p.line('threeD_x', 'threeD_y', line_width=0.5, line_dash='dotted', color='#435363', source=source_threeD) # x=3
    p.line('var95_x', 'var95_y', line_width=2, line_dash='dotted', color='#435363', source=source_var85) # var = 0.95

    # add bar plot for variance per dimension
    p.vbar(x='x_dim', top='y_var', width=.5, bottom=0, color='#98d687', source=source_PCA, name='PCA')
    
    # add cumulative variance (scatter + line)
    p.line('x_dim', 'y_cumvar', line_width=1, color='#F79737', source=source_PCA)
    p.circle('x_dim', 'y_cumvar', size=7, color='#FF4C00', source=source_PCA, name='PCA')

    # change gridlines
    p.ygrid.grid_line_alpha = 0.25
    p.xgrid.grid_line_alpha = 0.25

    # change axis bounds and grid
    p.xaxis.bounds = (0, dims_)
    p.yaxis.bounds = (0, 1)
    p.grid.bounds = (0, dims_)

    # save and show p
    if save_plot:
        output_file('PCA_analysis_dissertation.html')
    show(p)
        
    # output PCA info as a dataframe
    df_PCA = pd.DataFrame({'dimension': dimensions, 'variance_cumulative': cumsum_, 'variance': pca_evr}).set_index(['dimension'])
        
    return df_PCA, X_pca, pca_evr


def pca_feature_correlation(X, X_pca, explained_var, features:list=None, fig_dpi:int=250, save_plot:bool=False):
    """
    1. Get dot product of X and X_pca
    2. Run normalizations of X*X_pca
    3. Retrieve df/matrices
    X:               data (numpy matrix)
    X_pca:           PCA
    explained_var:   explained variance matrix
    features:        list of feature names
    fig_dpi:         dpi to use for heatmaps
    save_plot:       save plot to file (html) and not show
    """
    
    # Add zeroes for data where features > instances
    outer_diff = X.T.shape[0] - X_pca.shape[1]
    if outer_diff > 0: # outer dims must match to get sq matrix
        Z = np.zeros([X_pca.shape[0], outer_diff])
        X_pca = np.c_[X_pca, Z]
        explained_var = np.append(explained_var, np.zeros(outer_diff))
    
    # Get correlation between original features (X) and PCs (X_pca)
    dot_matrix = np.dot(X.T, X_pca)
    print(f"X*X_pca: {X.T.shape} * {X_pca.shape} = {dot_matrix.shape}")
    
    # Correlation matrix -> df
    df_dotproduct = pd.DataFrame(dot_matrix)
    df_dotproduct.columns = [''.join(['PC', f'{i+1}']) for i in range(dot_matrix.shape[0])]
    if any(features): df_dotproduct.index = features    
    
    # Normalize & Sort
    df_n, df_na, df_nabv = normalize_dataframe(df_dotproduct, explained_var, plot_opt=True, save_plot=save_plot)
    
    return df_dotproduct, df_n, df_na, df_nabv


def normalize_dataframe(df, explained_var=None, fig_dpi:int=250, plot_opt:bool=True, save_plot:bool=False):
    """
    1. Get z-normalized df (normalized to µ=0, σ=1)
    2. Get absolute value of z-normalized df
    3. If explained_variance matrix provided, dot it w/ (2)
    """
    # Normalize, Reindex, & Sort
    df_norm = (df.copy()-df.mean())/df.std()
    df_norm = df_norm.sort_values(list(df_norm.columns), ascending=False)
    
    # Absolute value of normalized (& sort)
    df_abs = df_norm.copy().abs().set_index(df_norm.index)
    df_abs = df_abs.sort_values(by=list(df_abs.columns), ascending=False)
    
    # Plot
    if plot_opt:
        # Z-normalized corr matrix
        plt.figure(dpi=fig_dpi)
        ax_normal = sns.heatmap(df_norm, cmap="RdBu")
        ax_normal.set_title("Z-Normalized Data")
        if save_plot:
            plt.savefig('Z_normalized_corr_matrix.png')
        else:
            plt.show()

        # |Z-normalized corr matrix|  
        plt.figure(dpi=fig_dpi)
        ax_abs = sns.heatmap(df_abs, cmap="Purples")
        ax_abs.set_title("|Z-Normalized|")
        if save_plot:
            plt.savefig('Z_normalized_corr_matrix_Abs.png')
        else:
            plt.show()
        
    # Re-normalize by explained var (& sort)
    if explained_var.any():
        df_byvar = df_abs.copy()*explained_var
        df_byvar = df_byvar.sort_values(by=list(df_norm.columns), ascending=False)
        if plot_opt:
            plt.figure(dpi=fig_dpi)
            ax_relative = sns.heatmap(df_byvar, cmap="Purples")
            ax_relative.set_title("|Z-Normalized|*Explained_Variance")
            if save_plot:
                plt.savefig('Normalized_corr_matrix.png')
            else:
                plt.show()
    else:
        df_byvar = None
    return df_norm, df_abs, df_byvar


def pca_rank_features(df_nabv, verbose:bool=True):
    """
    Given a dataframe df_nabv with dimensions [f, p], where:
        f = features (sorted)
        p = principal components
        df_nabv.values are |Z-normalized X|*pca_.explained_variance_ratio_
        
    1. Create column of sum of each row, sort by it 'score_'
    3. Set index as 'rank'
    """
    df_rank = df_nabv.copy().assign(score_ = df_nabv.sum(axis=1)).sort_values('score_', ascending=False)
    df_rank['feature_'] = df_rank.index
    df_rank.index = range(1, len(df_rank)+1)
    df_rank.drop(df_nabv.columns, axis=1, inplace=True)
    df_rank.index.rename('rank', inplace=True)
    if verbose: print(df_rank)
    return df_rank


def pca_full_report(X, features_:list=None, fig_dpi:int=150, save_plot:bool=False):
    """
    Run complete PCA workflow:
        1. pca_report_interactive()
        2. pca_feature_correlation()
        3. pca_rank_features()
        
    X:            data (numpy array)
    features_:    list of feature names
    fig_dpi:      image resolution
    
    """
    # Retrieve the interactive report
    df_pca, X_pca, pca_evr = pca_report_interactive(X, save_plot=save_plot)
    # Get feature-PC correlation matrices
    df_corr, df_n, df_na, df_nabv = pca_feature_correlation(X, X_pca, pca_evr, features_, fig_dpi, save_plot) 
    # Get rank for each feature
    df_rank = pca_rank_features(df_nabv)
    return (df_pca, X_pca, pca_evr, df_corr, df_n, df_na, df_nabv, df_rank)


if __name__ == '__main__':
    """ DISSERTATION DATA """

    diss_data=X.to_numpy()
    diss_feats=unpickled_NPI_SUMM_2016.columns.tolist()
    outputs_diss = pca_full_report(X=diss_data, features_=diss_feats, save_plot=True)
#%%# Logistic Regression Models - Training and Testing
# =========================================================# =========================================================# ===================
#                                                              TRAINING LR MODEL #  

# ========================================================= With all 176 features =========================================================
logmodl176=LogisticRegression(random_state=4, max_iter=10000).fit(NPI_Summ_Scld,y)
pred_176=logmodl176.predict(NPI_Summ_Scld)

print('Training data accuracy score with all 176 features : ', logmodl176.score(NPI_Summ_Scld,y))
cm176=confusion_matrix(y,logmodl176.predict(NPI_Summ_Scld))
print(classification_report(y,logmodl176.predict(NPI_Summ_Scld)))

#======================= TEST VARYING THRESHOLDS ==================================
# apply threshold to positive probabilities to create labels
# search thresholds for imbalanced classification
def to_labels(pos_probs, threshold):
	return (pos_probs >= threshold).astype('int')
# define thresholds
thresholds = arange(0, 1, 0.01)
# evaluate each threshold
scores = [f1_score(y, to_labels(logmodl176.predict_proba(NPI_Summ_Scld)[:,1], t)) for t in thresholds]
# get best threshold
ix = argmax(scores)
thresholds[ix]

f1_score(y, logmodl176.predict(NPI_Summ_Scld))
f1_score(y, (logmodl176.predict_proba(NPI_Summ_Scld)[:,1]>=0.09).astype('int'))
#======================= TEST VARYING THRESHOLDS ENDs here =======================

# predict probabilities
lr_probs = logmodl176.predict_proba(NPI_Summ_Scld)
# keep probabilities for the positive outcome only (fraud)
lr_probs = lr_probs[:, 1]
area_prc=average_precision_score(y_true=y,y_score=lr_probs) 
print("Area under PR - curve", area_prc)
area_roc=roc_auc_score(y_true=y,y_score=lr_probs) 
print("Area under ROC - curve", area_roc)

def my_pr_curve(my_mdl,df_X,true_y,pr_name):
       """ Plot a precision-recall curve"""
       display = PrecisionRecallDisplay.from_estimator(my_mdl, df_X, true_y, name=pr_name)
       _ = display.ax_.set_title("Fraud and No-Fraud class Precision-Recall curve")
       plt.show()
       
def my_roc_curve(my_mdl,df_X,true_y,pr_name):
       """ Plot AUC """
       display = plot_roc_curve(my_mdl, df_X, true_y, name=pr_name)
       _ = display.ax_.set_title("Fraud and No-Fraud class Area under ROC curve")
       plt.show()       
       
my_pr_curve(logmodl176, NPI_Summ_Scld, y, pr_name="Logistic regression with 176 features")      
my_roc_curve(logmodl176, NPI_Summ_Scld, y, pr_name="Logistic regression with 176 features")      

#%%# With 15 principal components
logmodl=LogisticRegression(random_state=4, max_iter=10000).fit(my100pc,y)
pred=logmodl.predict(my100pc)

print('Training score with 15 components : ', logmodl.score(my100pc,y))
cm15=confusion_matrix(y,logmodl.predict(my100pc))
print(classification_report(y,logmodl.predict(my100pc)))

my_pr_curve(logmodl, my100pc, y, pr_name="Logistic regression with 15 principal components")      
my_roc_curve(logmodl, my100pc, y, pr_name="Logistic regression with 15 principal components")  

# predict probabilities
lr_probs = logmodl.predict_proba(my100pc)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
area_prc=average_precision_score(y_true=y,y_score=lr_probs) 
print("Area under PR - curve", area_prc)
area_roc=roc_auc_score(y_true=y,y_score=lr_probs) 
print("Area under ROC - curve", area_roc)    
#%%# SMOTE resampling
###################################### With all 176 features - smote ######################################
print(imblearn.__version__)
sm=SMOTE(random_state=2,sampling_strategy=1.0)
X_SMOTE,y_SMOTE=sm.fit_resample(NPI_Summ_Scld,y)

counter=Counter(y)
print(counter)
counterSMOTE = Counter(y_SMOTE)
print(counterSMOTE)

logmodl176_smote=LogisticRegression(random_state=4, max_iter=10000).fit(X_SMOTE,y_SMOTE)
pred_176_smote=logmodl176_smote.predict(X_SMOTE)

print('Training score with all 176 features - SMOTE: ', logmodl176_smote.score(X_SMOTE,y_SMOTE))
cm176_smote=confusion_matrix(y_SMOTE,logmodl176_smote.predict(X_SMOTE))
print(classification_report(y_SMOTE ,logmodl176_smote.predict(X_SMOTE)))

my_pr_curve(logmodl176_smote, X_SMOTE, y_SMOTE, pr_name="Logistic regression with 176 features + SMOTE")      
my_roc_curve(logmodl176_smote, X_SMOTE, y_SMOTE, pr_name="Logistic regression with 176 features + SMOTE")  

# predict probabilities
lr_probs = logmodl176_smote.predict_proba(X_SMOTE)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
area_prc=average_precision_score(y_true=y_SMOTE,y_score=lr_probs) 
print("Area under PR - curve", area_prc)
area_roc=roc_auc_score(y_true=y_SMOTE,y_score=lr_probs) 
print("Area under ROC - curve", area_roc)   

###################################### With 15 components - smote ######################################
sm=SMOTE(random_state=2,sampling_strategy=1.0)
X_PC_SMOTE,y_pc_SMOTE=sm.fit_resample(my100pc,y)
logmodl_smote=LogisticRegression(random_state=4, max_iter=10000).fit(X_PC_SMOTE,y_pc_SMOTE)
pred_smote=logmodl_smote.predict(X_PC_SMOTE)

print('Training score with 15 components : ', logmodl_smote.score(X_PC_SMOTE,y_pc_SMOTE))
cm15_smote=confusion_matrix(y,logmodl_smote.predict(my100pc))
print(classification_report(y,logmodl_smote.predict(my100pc))) #On original dataset

my_pr_curve(logmodl_smote, my100pc,y, pr_name="Logistic regression with 15 principal components + SMOTE")      
my_roc_curve(logmodl_smote, my100pc,y, pr_name="Logistic regression with 15 principal components + SMOTE")  

# predict probabilities
lr_probs = logmodl_smote.predict_proba(my100pc)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
area_prc=average_precision_score(y_true=y,y_score=lr_probs) 
print("Area under PR - curve", area_prc)
area_roc=roc_auc_score(y_true=y,y_score=lr_probs) 
print("Area under ROC - curve", area_roc)  
#%%# Functions define - Confusion Matrix and ROC curve
def my_plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Greens):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """

  if not title:
    if normalize:
      title = 'Normalized confusion matrix'
    else:
      title = 'Confusion matrix, without normalization'   
                    
  # Compute confusion matrix
  #Convert y_true and y_pred to labels Fraud and Not fraud
  cl_y_true=list()
  for i in range(y_true.shape[0]):
    #print(i)                                     
    cl_y_true.insert(i,"Fraud") if y_true[i]==1 else cl_y_true.insert(i,"Not Fraud")
  cl_y_trueF=pd.Series(cl_y_true, index=range(y_true.shape[0]))
  #print('cl_y_true[4960]:',cl_y_true[4960])
  #print (type(cl_y_trueF))
  
  cl_y_pred=list() 

  for i in range(y_pred.shape[0]):
    #print(i)
    cl_y_pred.insert(i,"Fraud") if y_pred[i]==1 else cl_y_pred.insert(i,"Not Fraud") 
  #print('cl_y_pred[4960]:',cl_y_pred[4960])
  cl_y_predF=pd.Series(cl_y_pred, index=range(y_true.shape[0]))
  #print('cl_y_predF[4960',cl_y_predF[4960]=='Fraud')
  #print (type(cl_y_predF))
  cm = confusion_matrix(cl_y_trueF, cl_y_predF)
  # Only use the labels that appear in the data
  #classes = classes[unique_labels(y_true, y_pred)]
  classes = np.unique(cl_y_trueF)  

  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')

  print(cm)

  fig, ax = plt.subplots()
  im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
  ax.figure.colorbar(im, ax=ax)
  # We want to show all ticks...
  ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes, yticklabels=classes,
        title=title,
        ylabel='True label',
        xlabel='Predicted label')

  # Rotate the tick labels and set their alignment.
  plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
          rotation_mode="anchor")

  # Loop over data dimensions and create text annotations.
  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i in range(cm.shape[0]):
      for j in range(cm.shape[1]):
          ax.text(j, i, format(cm[i, j], fmt),
                  ha="center", va="center",
                  color="white" if cm[i, j] > thresh else "black")
  fig.tight_layout()
  return ax

np.set_printoptions(precision=2)
class_names = np.unique(y)
# Plot non-normalized confusion matrix - LR w 176 features (No resampling)
# my_plot_confusion_matrix(y,logmodl176.predict(NPI_Summ_Scld), classes=class_names,
#                       title='Confusion matrix, without normalization')

# Plot normalized confusion matrix - LR w 176 features (No resampling)
my_plot_confusion_matrix(y,logmodl176.predict(NPI_Summ_Scld), classes=class_names, 
                      normalize=True,
                      title='Normalized confusion matrix - Logistic regression with 176 features')

# Plot normalized confusion matrix - LR w 15 PCs (No resampling)
my_plot_confusion_matrix(y,logmodl.predict(my100pc), classes=class_names, 
                      normalize=True,
                      title='Normalized confusion matrix - Logistic regression with 15 principal components')

# Plot normalized confusion matrix - LR 176 feature + SMOTE
my_plot_confusion_matrix(y_SMOTE ,logmodl176_smote.predict(X_SMOTE), classes=class_names, 
                      normalize=True,
                      title='Normalized confusion matrix - Logistic regression with 176 features + SMOTE')

# Plot normalized confusion matrix - LR 15 PCs + SMOTE
my_plot_confusion_matrix(y,logmodl_smote.predict(my100pc), classes=class_names, 
                      normalize=True,
                      title='Normalized confusion matrix - Logistic regression with 15 principal components + SMOTE')

# Plot normalized confusion matrix - LR w 176 features (No resampling)
my_plot_confusion_matrix(y,logmodl176.predict(NPI_Summ_Scld), classes=class_names, 
                      normalize=False,
                      title='Confusion matrix - Logistic regression with 176 features')

# Plot non-normalized confusion matrix - LR w 15 PCs (No resampling)
# my_plot_confusion_matrix(y,logmodl_smote.predict(my100pc), classes=class_names,
#                       title='Confusion matrix, without normalization')

# Plot normalized confusion matrix - LR w 15 PCs (No resampling)
my_plot_confusion_matrix(y,logmodl.predict(my100pc), classes=class_names, 
                      normalize=False,
                      title='Confusion matrix - Logistic regression with 15 principal components')

# Plot normalized confusion matrix - LR 176 feature + SMOTE
my_plot_confusion_matrix(y_SMOTE ,logmodl176_smote.predict(X_SMOTE), classes=class_names, 
                      normalize=False,
                      title='Confusion matrix - Logistic regression with 176 features + SMOTE')

# Plot normalized confusion matrix - LR 15 PCs + SMOTE
my_plot_confusion_matrix(y,logmodl_smote.predict(my100pc), classes=class_names, 
                      normalize=False,
                      title='Confusion matrix - Logistic regression with 15 principal components + SMOTE')

plt.show()
#%%# Save Log Reg model as pickled file
filename = 'LR_176Features.sav'
pickle.dump(logmodl176, open(filename, 'wb'))

filename = 'LR_15Features.sav'
pickle.dump(logmodl, open(filename, 'wb'))

# With Smote
filename = 'LR_176FeaturesSMOTE.sav'
pickle.dump(logmodl176_smote, open(filename, 'wb'))

filename = 'LR_15FeaturesSMOTE.sav'
pickle.dump(logmodl_smote, open(filename, 'wb'))

# Write min-max scaler object and PC object as pickled objects 
with open('pickled_list_Scaler.pickle', 'wb') as f:
    pickle.dump(scaler, f)    
    
with open('pickled_list_pca100.pickle', 'wb') as f:
    pickle.dump(pca100, f)
    
# =========================================================# =========================================================# ===================
#                                                              TESTING LR MODEL #  

NPI_Summ_Scld_val=scaler.transform(X_val)
my100pc_val=pca100.transform(NPI_Summ_Scld_val)
print('Testing mean accuracy with all 176 features : ', logmodl176.score(NPI_Summ_Scld_val,y_val))
print('Testing report with all 176 features: \n',classification_report(y_val,logmodl176.predict(NPI_Summ_Scld_val)))
my_pr_curve(logmodl176, NPI_Summ_Scld_val, y_val, pr_name="Logistic regression with 176 features - Testing")      
my_roc_curve(logmodl176, NPI_Summ_Scld_val, y_val, pr_name="Logistic regression with 176 features - Testing")      

print('Testing mean accuracy with all 15 components : ', logmodl.score(my100pc_val,y_val))
print('Testing report with all 15 components tst: \n',classification_report(y_val,logmodl.predict(my100pc_val)))
my_pr_curve(logmodl,my100pc_val,y_val, pr_name="Logistic regression with all 15 components - Testing")      
my_roc_curve(logmodl,my100pc_val,y_val, pr_name="Logistic regression with all 15 components - Testing")      

print('Testing mean accuracy with all 176 features + SMOTE: ', logmodl176_smote.score(NPI_Summ_Scld_val,y_val))
print('Testing report with all 176 features + SMOTE: \n', classification_report(y_val ,logmodl176_smote.predict(NPI_Summ_Scld_val)))
my_pr_curve(logmodl176_smote,NPI_Summ_Scld_val,y_val, pr_name="Logistic regression with 176 features + SMOTE - Testing")      
my_roc_curve(logmodl176_smote,NPI_Summ_Scld_val,y_val, pr_name="Logistic regression with 176 features + SMOTE - Testing")      

print('Testing mean accuracy with all 15 components + SMOTE: ', logmodl_smote.score(my100pc_val,y_val))
print('Testing report with all 15 components + SMOTE: \n', classification_report(y_val,logmodl_smote.predict(my100pc_val))) 
my_pr_curve(logmodl_smote,my100pc_val,y_val, pr_name="Logistic regression with 15 components + SMOTE - Testing")      
my_roc_curve(logmodl_smote,my100pc_val,y_val, pr_name="Logistic regression with 15 components + SMOTE - Testing")      

#%%# Random Forest Models
X_RF_TRN=df_trn.drop(['NPI_FrdConf','NPI_FrdBin'],1)
y_rf_trn=df_trn['NPI_FrdBin']
X_RF_TRN.shape
y_rf_trn.shape

X_RF_VAL=df_val.drop(['NPI_FrdConf','NPI_FrdBin'],1)
y_rf_val=df_val['NPI_FrdBin']
X_RF_VAL.shape
y_rf_val.shape

print(f'Number of columns ={X_RF_TRN.shape[1]}')
# scaler=MinMaxScaler().fit(X_RF_TRN)
X_RF_Scld_trn=scaler.transform(X_RF_TRN)
X_RF_Scld_val=scaler.transform(X_RF_VAL)

def create_model(model, X_train, Y_train):
       model=model.fit(X_train, Y_train)
       return model

model=create_model(RandomForestClassifier(n_estimators=200, random_state=100),X_RF_Scld_trn,y_rf_trn)
# RF with balanced class weight
model_bal1=create_model(RandomForestClassifier(n_estimators=200, random_state=100,class_weight='balanced'),X_RF_Scld_trn,y_rf_trn) 
# RF with bootstrap class weighting
model_bal2=create_model(RandomForestClassifier(n_estimators=200, random_state=100,class_weight='balanced_subsample'),X_RF_Scld_trn,y_rf_trn)
# RF with random undersampling
# model_bal3=create_model(BalancedRandomForestClassifier(n_estimators=200, random_state=100),X_RF_Scld_trn,y_rf_trn)
#%%# RF Metrics Train & Test
print('=========================================')
y_predict_trn=model.predict(X_RF_Scld_trn)
print('Training score with standard random forest estimator : \n',classification_report(y_rf_trn,y_predict_trn))

print('=========================================')
y_predict_val=model.predict(X_RF_Scld_val)
print('Testing score with standard random forest estimator : \n',classification_report(y_rf_val,y_predict_val))

print('=========================================')
y_predict_trn_bal1RF=model_bal1.predict(X_RF_Scld_trn)
print('Training score with balanced random forest estimator : \n',classification_report(y_rf_trn,y_predict_trn_bal1RF))

print('=========================================')
y_predict_val_bal1RF=model_bal1.predict(X_RF_Scld_val)
print('Testing score with balanced random forest estimator : \n',classification_report(y_rf_val,y_predict_val_bal1RF))

print('=========================================')
y_predict_trn_bal2RF=model_bal2.predict(X_RF_Scld_trn)
print('Training score with balanced sub sample random forest estimator : \n',classification_report(y_rf_trn,y_predict_trn_bal2RF))

print('=========================================')
y_predict_val_bal2RF=model_bal2.predict(X_RF_Scld_val)
print('Testing score with balanced sub sample random forest estimator : \n',classification_report(y_rf_val,y_predict_val_bal2RF))
# print('=========================================')

# Look at parameters used by our current forest
print('Parameters currently in use standard RF:\n')
pprint(model.get_params())

print('Parameters currently in use balanced RF:\n')
pprint(model_bal1.get_params())

print('Parameters currently in use balanced sub sample RF:\n')
pprint(model_bal2.get_params())
#%%# RF AUC
y_proba_rf=model.predict_proba(X_RF_Scld_trn)[:,1]
print("Roc AUC Standard TRG data:", roc_auc_score(y_rf_trn, y_proba_rf,average='macro'))

y_proba_rf=model_bal1.predict_proba(X_RF_Scld_trn)[:,1]
print("Roc AUC Balanced TRG data:", roc_auc_score(y_rf_trn, y_proba_rf,average='macro'))

y_proba_rf=model_bal2.predict_proba(X_RF_Scld_trn)[:,1]
print("Roc AUC Balanced subsample TRG data:", roc_auc_score(y_rf_trn, y_proba_rf,average='macro'))

# Validation data
y_proba_rf=model.predict_proba(X_RF_Scld_val)[:,1]
print("Roc AUC Standard TST data:", roc_auc_score(y_rf_val, y_proba_rf,average='macro'))

y_proba_rf=model_bal1.predict_proba(X_RF_Scld_val)[:,1]
print("Roc AUC Balanced TST data:", roc_auc_score(y_rf_val, y_proba_rf,average='macro'))

y_proba_rf=model_bal2.predict_proba(X_RF_Scld_val)[:,1]
print("Roc AUC Balanced subsample TST data:", roc_auc_score(y_rf_val, y_proba_rf,average='macro'))
#%%# Optimization trials on RF models

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 220, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

rf=RandomForestClassifier()
# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations, and use all available cores
start_time = time.time()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 5, verbose=2, random_state=100, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_RF_Scld_trn,y_rf_trn)
print("Program took %s seconds" % (time.time() - start_time))#Time to run the RF optimization on training data
pprint(rf_random.best_params_)
best_random = rf_random.best_estimator_
#%%# To determine if random search yielded better model - compare to base RF
print('=========================================')
print('Training score with random forest random search best estimator : \n',classification_report(y_rf_trn,y_predict_trn))
y_proba_rf=best_random.predict_proba(X_RF_Scld_trn)[:,1]
y_predict_trn=best_random.predict(X_RF_Scld_trn)
print("Roc Trg AUC RF Randomized Search CV based params:", roc_auc_score(y_rf_trn, y_proba_rf,average='macro'))

print('=========================================')
y_predict_val=best_random.predict(X_RF_Scld_val)
print('Testing score with random forest random search best estimator : \n',classification_report(y_rf_val,y_predict_val))

y_proba_rf_rscv=best_random.predict_proba(X_RF_Scld_val)[:,1]
print("Roc Tst AUC RF Randomized Search CV based params:", roc_auc_score(y_rf_val, y_proba_rf_rscv,average='macro'))
#%%# Logistic regression with 176 features interpretability - Best Model 3rd model in LR data experiment trials = logmodl176_smote.sav
logmodl176_smote.intercept_[0]# Intercept
wts=logmodl176_smote.coef_[0] # Coefficient weights array

feat_nms=unpickled_NPI_SUMM_2016.columns.tolist()

# Interpretability
feature_importance = pd.DataFrame(feat_nms, columns = ["feature"])
feature_importance["importance"] = pow(math.e, wts)
feature_importance = feature_importance.sort_values(by = ["importance"], ascending=False)
#%%# To determine feature Imp from RF even though it is not the best model
# get importance
rf_importance = best_random.feature_importances_
# summarize feature importance
for i,v in enumerate(rf_importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(rf_importance))], rf_importance)
plt.show()


       

