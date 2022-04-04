# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 21:47:33 2021

@author: nisha

This file is run after Dissertation_Publish_03.py script
This script creates 176 features from 2017 raw claims data (Testing dataset), creates a data testing matrix,  performs PCA on the data matrix 
and saves as pickled files for consumption in Dissertation_Publish_04.py script

"""
#%%# Library imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
#%%# Read Data for year 2017 - for not compound drug claims
#CombAll=pd.read_csv('C:/Users/Acer/Desktop/Nisha/Nisha Career/PhD/Research/Nisha Rsch/Raw Data/Comball',dtype={'deid_pharmnpi': str})
start_time = time.time()
CombAllRxT=pd.read_csv('C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/ComballRX_2017',\
                     dtype={'Source':'str','REFILLNBR':'uint8','RXNBR':'str',\
                             'DAYSUPPLY':'uint16','AUTHREFILL':'uint8','SEQNBR':'str','ICN':'str','NDC':'str',\
                             'LABELNAME':'str','AHFS':'str','RXQTY':'str','DISPENSEFEE':'float64','PAIDAMT':'float64',\
                             'NDCSRC':'str','NDCSRCDESC':'str','DEACD':'str','COMPOUNDCD':'str','DAWCD':'str','THERACLASS':'str',\
                             'THERADESC':'str', 'MAINT_DRUG_IND':'str','DEA_CODE':'str',\
                            'clientcat':'str','mcoprog':'str','deid_pcn':'str',\
                            'deid_pharmnpi':'str','deid_prescribenpi':'str','RX_filldt':'str','RX_dt':'str','Tot_Pd':'float64'})
print("CombAllRxT took %s seconds" % (time.time() - start_time))#Time to load the combined dataset

#%%# Checks on 2017 data based on prescription dates 
CombAllRxT['RX_filldt_Yr']=pd.DatetimeIndex(CombAllRxT['RX_filldt']).year
CombAllRxT['RX_filldt_Mon']=pd.DatetimeIndex(CombAllRxT['RX_filldt']).month
CombAllRxT['RX_filldt_Yr'].value_counts() # RX fill date should be all 2017

CombAllRxT['deid_pcn'].nunique() #3,116,270 for 2017

CombAllRxT['deid_pharmnpi'].nunique() # 5007 for 2017

CombAllRxT.columns
CombAllRxT.head()
dataTypeSeries = CombAllRxT.dtypes
print('Data type of each column of Dataframe :')
print(dataTypeSeries)

#%%# Feature creation based on a pharmacy id (de-identified NPI here) --- FEATURE ENGINEERING
# Features of Testing data
# Features 1 to 63
# 1. No. of unique PCNs per NPI
NPI_PCNs = CombAllRxT.groupby('deid_pharmnpi')['deid_pcn'].nunique().to_frame().rename(columns={'deid_pcn':'NPI_PCNCnt'}) 

def q3(x):
    return x.quantile(0.75)

# 2. Paid amount, DF, tot paid amounts - count of claims, sum of $$, mean, std, median and q3
# Total of 18 features each for compound and 18 features each for non-compound drugs 

aggrAllDoll = {
    'PAIDAMT':['count', 'sum', 'mean', 'std', 'median', q3],
    'DISPENSEFEE':['sum', 'mean', 'std', 'median', q3],
    'Tot_Pd':['sum', 'mean', 'std', 'median', q3]
}

aggNPI_tmp=CombAllRxT.groupby(['deid_pharmnpi','COMPOUNDCD']).agg(aggrAllDoll)
aggNPI_tmp.columns = aggNPI_tmp.columns.map('_'.join)
NPI_RxtypeDoll=aggNPI_tmp.reset_index().pivot(index='deid_pharmnpi', columns='COMPOUNDCD')
NPI_RxtypeDoll.columns = NPI_RxtypeDoll.columns.map('_'.join)

# 3. Paid amount, DF, tot paid amounts - count of claims, sum of $$, mean, std, median and q3
# Total of 18 features each for MCO categories Dental, STAR, FFS, STAR KIDS and STAR+PLUS 
# Define the aggregation procedure outside of the groupby operation
aggrTotDoll = {
    'Tot_Pd':['count', 'sum', 'mean', 'std', 'median', q3]
}

CombAllRxT.dtypes
myMCO = ['Dental', 'STAR', 'FFS','STAR Kids','STAR+PLUS']

aggNPI_tmp=CombAllRxT[CombAllRxT['mcoprog'].isin(myMCO)].groupby(['deid_pharmnpi','mcoprog']).agg(aggrTotDoll)
aggNPI_tmp.columns = aggNPI_tmp.columns.map('_'.join)
NPI_MCOtypeDoll=aggNPI_tmp.reset_index().pivot(index='deid_pharmnpi', columns='mcoprog').rename(columns={'Dental':'Dent','STAR':'STR', 'STAR+PLUS':'STRP','FFS':'FFS','STAR Kids':'STRK'})
NPI_MCOtypeDoll.columns = NPI_MCOtypeDoll.columns.map('_'.join)
#%%# Features continued...
# Features 64 to 87
# 4. No. of unique prescribing providers per NPI 
NPI_PrescCnt=CombAllRxT.groupby('deid_pharmnpi')['deid_prescribenpi'].nunique().to_frame().rename(columns={'deid_prescribenpi':'UniqPrescribrs'})

# 5. DAW code value of 1 i.e., Physician DAW - count of claims, sum of $$, mean, std, median and q3
NPI_DAW1_Doll=CombAllRxT[CombAllRxT['DAWCD']=='1'].groupby(['deid_pharmnpi']).agg(aggrTotDoll)
NPI_DAW1_Doll.columns = NPI_DAW1_Doll.columns.map('_DAW1_'.join)

# 6. Collapse age categories to 3 final categories, 0 to 10, 10 - 20 and 20  above
# These categories were chosen based on the ## of claims in each bracket so as to have equal distribution within each category.
# create a list of our conditions
myCat1 = ['0-2', '2-4','4-6', '6-8', '8-10']
myCat2 = ['10-12', '12-14','14-16', '16-18', '18-20']
conditions = [
    (CombAllRxT['clientcat'].isin(myCat1)),
    (CombAllRxT['clientcat'].isin(myCat2))
    ]

# create a list of the values we want to assign for each condition
values = ['Cat1', 'Cat2']

CombAllRxT['clientcatR']=np.select(conditions, values, default='Cat3')

aggNPI_tmp = CombAllRxT.groupby(['clientcatR']).agg(aggrTotDoll)
aggNPI_tmp.columns = aggNPI_tmp.columns.map('_'.join)
#sns.barplot(x='clientcatR', y='Tot_Pd_count', data=aggNPI_tmp.reset_index())
#sns.barplot(x='clientcatR', y='Tot_Pd_mean', data=aggNPI_tmp.reset_index())
#sns.barplot(x='clientcatR', y='Tot_Pd_sum', data=aggNPI_tmp.reset_index())
#sns.barplot(x='clientcatR', y='Tot_Pd_median', data=aggNPI_tmp.reset_index())

aggNPI_tmp = CombAllRxT.groupby(['deid_pharmnpi','clientcatR']).agg(aggrTotDoll) 
aggNPI_tmp.columns = aggNPI_tmp.columns.map('_'.join)
NPI_ClientCatDoll=aggNPI_tmp.reset_index().pivot(index='deid_pharmnpi', columns='clientcatR')
NPI_ClientCatDoll.columns = NPI_ClientCatDoll.columns.map('_'.join)
#%%# 7. Prepare PMPMexposure matrix to calculate feature PMPMexposure (Per member per month exposure)
# Calculating PCN table for determining member month exposures
MembExp1=CombAllRxT.groupby(['deid_pcn','RX_filldt_Yr','RX_filldt_Mon']).size().reset_index().rename(columns={0:'Active'}) #claim counts
MembExp1['bool']=(MembExp1['Active']>0).astype(int)
MembExpF=MembExp1.groupby('deid_pcn')['bool'].sum().reset_index()
MembExpF.head()

NPI_PCN_TPD=CombAllRxT.groupby(['deid_pharmnpi','deid_pcn'])[['Tot_Pd']].apply(sum).reset_index()
NPI_PCN_TPD.head()
NPI_PCN_mat=pd.merge(left=NPI_PCN_TPD, right=MembExpF, left_on=['deid_pcn'], right_on=['deid_pcn'])
NPI_PCN_mat.head()
PCN_NPI_Ovlap=NPI_PCN_TPD.groupby(['deid_pcn'])['deid_pharmnpi'].nunique().to_frame()
PCN_NPI_Ovlap['deid_pharmnpi'].max() # No. of PCNs who have maximum no. of NPI overlaps i.e., the PCN is shared by max number of NPIs
PCN_NPI_Ovlap['deid_pharmnpi'].median()
PCN_NPI_Ovlap['deid_pharmnpi'].mean()
PCN_WMT1NPI=PCN_NPI_Ovlap[PCN_NPI_Ovlap['deid_pharmnpi']>30]
PCN_WMT1NPI.shape
PCN_WMT1NPI.head()
NPI_PCN_mat[NPI_PCN_mat['deid_pcn']=='853632738']
PMPM_exp=NPI_PCN_mat.groupby(['deid_pharmnpi'])[['Tot_Pd','bool']].apply(sum)
PMPM_exp['PMPMexp']=PMPM_exp['Tot_Pd']/PMPM_exp['bool']
#%%# Load thresholds from pickled files
with open('C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/pickled_list_myTC.pickle', 'rb') as f:
    myTC = pickle.load(f)
    
with open('C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/pickled_list_myAHFS.pickle', 'rb') as f:
    myAHFS = pickle.load(f)
    
with open('C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/pickled_list_thsh.pickle', 'rb') as f:
    thsh = pickle.load(f)       
    
with open('C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/pickled_list_thsh1.pickle', 'rb') as f:
    thsh1 = pickle.load(f)
    
#%%# Features 89 - 176
# 8. To apply the top 5  thera class from trg data to test data
aggNPI_tmp=CombAllRxT[CombAllRxT['THERACLASS'].isin(myTC)].groupby(['deid_pharmnpi','THERACLASS']).agg(aggrTotDoll)
aggNPI_tmp.columns = aggNPI_tmp.columns.map('_'.join)
NPI_Top5TCDoll=aggNPI_tmp.reset_index().pivot(index='deid_pharmnpi', columns='THERACLASS')
NPI_Top5TCDoll.columns = NPI_Top5TCDoll.columns.map('_'.join)

# 9. To apply the top 5 AHFS drug categories from trg data to test data
aggNPI_tmp=CombAllRxT[CombAllRxT['AHFS'].isin(myAHFS)].groupby(['deid_pharmnpi' ,'AHFS']).agg(aggrTotDoll)
aggNPI_tmp.columns = aggNPI_tmp.columns.map('_'.join)
NPI_Top5AHFSDoll=aggNPI_tmp.reset_index().pivot(index='deid_pharmnpi', columns='AHFS')
NPI_Top5AHFSDoll.columns = NPI_Top5AHFSDoll.columns.map('_'.join)

#%%# # Concatenate to form the feature matrix for each Pharm NPI
dfconcats=[NPI_PCNs, NPI_PrescCnt, NPI_DAW1_Doll, NPI_RxtypeDoll, NPI_MCOtypeDoll, NPI_ClientCatDoll, PMPM_exp['PMPMexp'],NPI_Top5AHFSDoll,NPI_Top5TCDoll]
NPI_Summ1=pd.concat(dfconcats,axis=1) 
NPI_Summ1.columns
NPI_Summ1.fillna(0, inplace=True)
NPI_Summ1.columns
#%%# 10. PPR relation establish
# Define a function that counts unique values in a passed vector
def uniqcnt(x):
    return x.nunique()

PPR = {
    'Tot_Pd':['count', 'sum', 'mean', 'median'],
    'deid_pcn':[uniqcnt],
}

aggNPI_tmp = CombAllRxT.groupby(['deid_pharmnpi', 'deid_prescribenpi']).agg(PPR)
aggNPI_tmp.columns = aggNPI_tmp.columns.map('_'.join)

# Prescription count by NPI and prescriber
PresCntbyNPI=(NPI_Summ1['PAIDAMT_count_H']+NPI_Summ1['PAIDAMT_count_N']).to_frame().rename(columns = {0:'PresCnt'}) 
PresCntbyPresc=CombAllRxT.groupby(['deid_prescribenpi']).agg(aggrTotDoll)
PresCntbyPresc.columns = PresCntbyPresc.columns.map('_'.join)
PresCntbyPresc=PresCntbyPresc.drop(index='.') # Drop missing prescriber NPI rows
PresCntbyPrescCnt=PresCntbyPresc['Tot_Pd_count'].to_frame().rename(columns = {'Tot_Pd_count':'PresCntbyPresc'})

aggNPI_tmp=aggNPI_tmp.join(PresCntbyNPI,on='deid_pharmnpi')
aggNPI_tmp['PCNT_Presc']=(aggNPI_tmp['Tot_Pd_count']/aggNPI_tmp['PresCnt']).multiply(100)
aggNPI_tmp['PCNT_Presc'].max()
aggNPI_tmp['PCNT_Presc'].quantile(0.75)


aggNPI_tmp=aggNPI_tmp.join(PresCntbyPrescCnt,on='deid_prescribenpi')
aggNPI_tmp['PCNT_Pharm']=(aggNPI_tmp['Tot_Pd_count']/aggNPI_tmp['PresCntbyPresc']).multiply(100)
aggNPI_tmp['PCNT_Pharm'].max()
aggNPI_tmp['PCNT_Pharm'].quantile(0.75)

aggNPI_tmp['PCNT_Tot']=aggNPI_tmp['PCNT_Pharm']+aggNPI_tmp['PCNT_Presc']
#thsh=aggNPI_tmp['deid_pcn_uniqcnt'].quantile(0.75)
aggNPI_tmp[aggNPI_tmp['deid_pcn_uniqcnt']>thsh]['PCNT_Tot'].quantile(0.75)
aggNPI_tmp['PCNT_Tot'].max()

aggNPI_tmp['PCNT_Tot'].quantile(0.75)

# If a NPI and prescriber relation total is more than 75 and the NPI, prec combination had atleast 'thsh' no. of PCNs we flag as collusion
#thsh1=NPI_Summ1['NPI_PCNCnt'].quantile(0.25)
aggNPI_PrescReln=aggNPI_tmp[(aggNPI_tmp['PCNT_Tot']>75) & (aggNPI_tmp['deid_pcn_uniqcnt']>thsh1)]
aggNPI_PrescReln=aggNPI_PrescReln.reset_index()
aggNPI_P_Rln=aggNPI_PrescReln.groupby(['deid_pharmnpi'])['deid_prescribenpi'].count().reset_index().rename(columns = {'deid_prescribenpi':'PresCnt'})
aggNPI_P_Rln.set_index('deid_pharmnpi', inplace=True)
#%%# 11. Days of supply counts
DYS30_Cnts=CombAllRxT[(CombAllRxT['COMPOUNDCD']=='N') & (CombAllRxT['DAYSUPPLY']==30)].groupby(['deid_pharmnpi']).agg(PPR)
DYS30_Cnts.columns = DYS30_Cnts.columns.map('_30DS_'.join)
DYS30_Cnts.index.name

#%%# 12. Brand Generic prescription counts for each pharmacy NPI
# We look at Single Source, Innovator and Branded drug non-compound claims only for each provider.
NDCSRC_Cnts=CombAllRxT[(CombAllRxT['COMPOUNDCD']=='N') & (CombAllRxT['NDCSRC']!='G')].groupby(['deid_pharmnpi','NDCSRC']).agg(PPR)
NDCSRC_Cnts.columns = NDCSRC_Cnts.columns.map('_'.join)
NDCSRC_type=NDCSRC_Cnts.reset_index().pivot(index='deid_pharmnpi', columns='NDCSRC')
NDCSRC_type.columns = NDCSRC_type.columns.map('_NDCsrc_'.join)
NDCSRC_type.index.name

#%%# 13. DEA code morphine drug counts for each pharmacy NPI
DEACD2=CombAllRxT[(CombAllRxT['COMPOUNDCD']=='N')  & (CombAllRxT['DEA_CODE']=='2')].groupby(['deid_pharmnpi']).agg(aggrTotDoll)
DEACD2.columns = DEACD2.columns.map('_DEACD2_'.join)
DEACD2.index.name
#%%#
dfconcats=[]
dfconcats=[NPI_PCNs, NPI_PrescCnt, NPI_DAW1_Doll, NPI_RxtypeDoll, NPI_MCOtypeDoll, NPI_ClientCatDoll, PMPM_exp['PMPMexp'],NPI_Top5AHFSDoll,NPI_Top5TCDoll,aggNPI_P_Rln,DYS30_Cnts,NDCSRC_type,DEACD2]
NPI_Summ1=pd.concat(dfconcats,axis=1) 
NPI_Summ1.columns
NPI_Summ1.fillna(0, inplace=True)
NPI_Summ1.columns
# NPI_Summ1.to_csv('C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/NPI_SummT_2017.csv',index=True)
# with open('C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/pickled_list_NPI_Summ_2017.pickle', 'wb') as f:
#     pickle.dump(NPI_Summ1, f)  