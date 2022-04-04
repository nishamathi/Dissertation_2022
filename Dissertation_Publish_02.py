# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 18:36:38 2020

This file is run after Dissertation_Publish_01.py script
This script creates 176 features from 2016 raw claims data (Training dataset), creates a data matrix, performs PCA on the data matrix 
and saves as pickled files for consumption in Dissertation_Publish_03.py script

@author: nisha
"""
#%%# Library imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import imblearn

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import pickle
from imblearn.over_sampling import SMOTE
#%%# Read Data for year 2016 - for not compound drug claims
start_time = time.time()
CombAllRx=pd.read_csv('C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/ComballRX_2016',\
                     dtype={'Source':'str','REFILLNBR':'uint8','RXNBR':'str',\
                             'DAYSUPPLY':'uint16','AUTHREFILL':'uint8','SEQNBR':'str','ICN':'str','NDC':'str',\
                             'LABELNAME':'str','AHFS':'str','RXQTY':'str','DISPENSEFEE':'float64','PAIDAMT':'float64',\
                             'NDCSRC':'str','NDCSRCDESC':'str','DEACD':'str','COMPOUNDCD':'str','DAWCD':'str','THERACLASS':'str',\
                             'THERADESC':'str', 'MAINT_DRUG_IND':'str','DEA_CODE':'str',\
                            'clientcat':'str','mcoprog':'str','deid_pcn':'str',\
                            'deid_pharmnpi':'str','deid_prescribenpi':'str','RX_filldt':'str','RX_dt':'str','Tot_Pd':'float64'})
print("CombAllRx took %s seconds" % (time.time() - start_time))#Time to load the combined dataset
#%%#
print("Unique PCNs: ",CombAllRx['deid_pcn'].nunique()) 
#3,113,182 for 2016
#3,116,270 for 2017
print("Unique Pharmacy provider: ",CombAllRx['deid_pharmnpi'].nunique()) 
# 4982 for 2016
# 5007 for 2017
#%%# Checks
# Data checks
#msctrx=CombAllRx[CombAllRx['AHFS']=='92000044'] # Checking Misc Therpeutic drug category NDCs
#deacd5=CombAllRx[CombAllRx['DEACD']=='5'] # Checking DEA code 5 NDCs
#brnd=CombAllRx[CombAllRx['NDCSRC']=='B'].head(10) # Checking branded generic
CombAllRx.columns
CombAllRx.head()
dataTypeSeries = CombAllRx.dtypes
print('Data type of each column of Dataframe :')
print(dataTypeSeries)
NPI_ClmCnts=CombAllRx['deid_pharmnpi'].value_counts().to_frame()# No. of claim details as df
sns.distplot(NPI_ClmCnts,kde=False,color='blue',bins=20)
#%%# Feature creation based on a pharmacy id (de-identified NPI here) --- FEATURE ENGINEERING

#Features 1 to 87
# 1. No. of unique PCNs per NPI
NPI_PCNs = CombAllRx.groupby('deid_pharmnpi')['deid_pcn'].nunique().to_frame().rename(columns={'deid_pcn':'NPI_PCNCnt'}) 
# There are ~200 providers with <= 10 PCNs
sns.distplot(NPI_PCNs,kde=False,color='blue',bins=20)

# Define the aggregation procedure outside of the groupby operation - Claim=focused features
# Finds q3, count, sum, mean, std, median for a group by condition.
def q3(x):
    return x.quantile(0.75)

# 2. Paid amount, DF, tot paid amounts - count of claims, sum of $$, mean, std, median and q3
# Total of 32 features - 16 each for compound and 16 each for non-compound drugs 

aggrAllDoll = {
    'PAIDAMT':['count', 'sum', 'mean', 'std', 'median', q3],
    'DISPENSEFEE':['sum', 'mean', 'std', 'median', q3],
    'Tot_Pd':['sum', 'mean', 'std', 'median', q3]
}

aggNPI_tmp=CombAllRx.groupby(['deid_pharmnpi','COMPOUNDCD']).agg(aggrAllDoll)
aggNPI_tmp.columns = aggNPI_tmp.columns.map('_'.join)
NPI_RxtypeDoll=aggNPI_tmp.reset_index().pivot(index='deid_pharmnpi', columns='COMPOUNDCD')
NPI_RxtypeDoll.columns = NPI_RxtypeDoll.columns.map('_'.join)

# 3. Tot paid amounts - count of claims, sum of $$, mean, std, median and q3 - 30 features
# Total of 30 features (5 categorie * 6 measures) each for MCO categories Dental, STAR, FFS, STAR KIDS and STAR+PLUS 
# Define the aggregation procedure outside of the groupby operation
aggrTotDoll = {
    'Tot_Pd':['count', 'sum', 'mean', 'std', 'median', q3]
}

CombAllRx.dtypes
myMCO = ['Dental', 'STAR', 'FFS','STAR Kids','STAR+PLUS']

aggNPI_tmp=CombAllRx[CombAllRx['mcoprog'].isin(myMCO)].groupby(['deid_pharmnpi','mcoprog']).agg(aggrTotDoll)
aggNPI_tmp.columns = aggNPI_tmp.columns.map('_'.join)
NPI_MCOtypeDoll=aggNPI_tmp.reset_index().pivot(index='deid_pharmnpi', columns='mcoprog').rename(columns={'Dental':'Dent','STAR':'STR', 'STAR+PLUS':'STRP','FFS':'FFS','STAR Kids':'STRK'})
NPI_MCOtypeDoll.columns = NPI_MCOtypeDoll.columns.map('_'.join)

# 4. No. of unique prescribing providers per NPI - 1 feature
NPI_PrescCnt=CombAllRx.groupby('deid_pharmnpi')['deid_prescribenpi'].nunique().to_frame().rename(columns={'deid_prescribenpi':'UniqPrescribrs'})

# 5. DAW code value of 1 i.e., Physician DAW - count of claims, sum of $$, mean, std, median and q3 - 6 features
NPI_DAW1_Doll=CombAllRx[CombAllRx['DAWCD']=='1'].groupby(['deid_pharmnpi']).agg(aggrTotDoll)
NPI_DAW1_Doll.columns = NPI_DAW1_Doll.columns.map('_DAW1_'.join)

# 6. Collapse age categories to 3 final categories, 0 to 10, 10 - 20 and 20  above - 18 features
# These categories were chosen based on the ## of claims in each bracket so as to have equal distribution within each category.
# create a list of our conditions
myCat1 = ['0-2', '2-4','4-6', '6-8', '8-10']
myCat2 = ['10-12', '12-14','14-16', '16-18', '18-20']
conditions = [
    (CombAllRx['clientcat'].isin(myCat1)),
    (CombAllRx['clientcat'].isin(myCat2))
    ]

# create a list of the values we want to assign for each condition
values = ['Cat1', 'Cat2']

CombAllRx['clientcatR']=np.select(conditions, values, default='Cat3')

aggNPI_tmp = CombAllRx.groupby(['clientcatR']).agg(aggrTotDoll)
aggNPI_tmp.columns = aggNPI_tmp.columns.map('_'.join)
sns.barplot(x='clientcatR', y='Tot_Pd_count', data=aggNPI_tmp.reset_index())
sns.barplot(x='clientcatR', y='Tot_Pd_mean', data=aggNPI_tmp.reset_index())
sns.barplot(x='clientcatR', y='Tot_Pd_sum', data=aggNPI_tmp.reset_index())
sns.barplot(x='clientcatR', y='Tot_Pd_median', data=aggNPI_tmp.reset_index())

aggNPI_tmp = CombAllRx.groupby(['deid_pharmnpi','clientcatR']).agg(aggrTotDoll) 
aggNPI_tmp.columns = aggNPI_tmp.columns.map('_'.join)
NPI_ClientCatDoll=aggNPI_tmp.reset_index().pivot(index='deid_pharmnpi', columns='clientcatR')
NPI_ClientCatDoll.columns = NPI_ClientCatDoll.columns.map('_'.join)

#Creating a heat map for average $$ by NPI
NPI_heat=NPI_ClientCatDoll.iloc[:,[6,7,8]]
cols = list(NPI_heat.columns.values)
NPI_heat1=NPI_heat[['Tot_Pd_mean_Cat1', 'Tot_Pd_mean_Cat2', 'Tot_Pd_mean_Cat3']]
fig, ax = plt.subplots(figsize=(20,10))
sns.heatmap(NPI_heat.transpose(), cmap="YlGnBu", robust=True, ax=ax)

#Creating a heat map for median $$ by NPI
NPI_heat=NPI_ClientCatDoll.iloc[:,[12,13,14]]
cols = list(NPI_heat.columns.values)
NPI_heat1=NPI_heat[cols]
fig, ax = plt.subplots(figsize=(20,10))
sns.heatmap(NPI_heat.transpose(), cmap="YlGnBu", robust=True, ax=ax)
#%%# Features 88
# 7. Prepare PMPMexposure matrix to calculate feature PMPMexposure (Per member per month exposure)
CombAllRx['RX_filldt_Yr']=pd.DatetimeIndex(CombAllRx['RX_filldt']).year
CombAllRx['RX_filldt_Mon']=pd.DatetimeIndex(CombAllRx['RX_filldt']).month
CombAllRx['RX_filldt_Yr'].value_counts() # RX fill date should be all 2016
CombAllRx['RX_filldt_Yr'].value_counts()

# Calculating PCN table for determining member month exposures
MembExp1=CombAllRx.groupby(['deid_pcn','RX_filldt_Yr','RX_filldt_Mon']).size().reset_index().rename(columns={0:'Active'}) #claim counts
MembExp1['bool']=(MembExp1['Active']>0).astype(int)
MembExpF=MembExp1.groupby('deid_pcn')['bool'].sum().reset_index()
MembExpF.head()

NPI_PCN_TPD=CombAllRx.groupby(['deid_pharmnpi','deid_pcn'])[['Tot_Pd']].apply(sum).reset_index()
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
#%%# Features 89 - 118
#8. To find the top 5 Thera class drug categories that have a high difference between the % of No. of claims for the Thera class 
# and the % of DOllars utilized by the Thera class. That is if 5% of the claims are the composition for a Theraclass
# but this class occupies 10% of the total pais dollars amongst all Theraclasses, then we choose this Theraclass, 
# We choose top 5 theraclasses this way.

# I only look at non-compound claims counts here

# Theraclass count of claims in data
TCls_Cnts=CombAllRx[CombAllRx['COMPOUNDCD']=='N'].groupby(['THERACLASS','THERADESC']).size().reset_index().rename(columns={0:'TC_NUMOFCLMS'})
TCls_Cnts.set_index('THERACLASS',inplace=True)
TCls_Cnts.index.name
#TCls_Cnts.index.name='THERACLASS' # Rename index column name
#TCls_Cnts.index.name
# No. of unique prescribing NPIs for each thera class
TCls_PrescNPI=CombAllRx.groupby('THERACLASS')['deid_prescribenpi'].nunique().to_frame().rename(columns={'deid_prescribenpi':'TC_CntPrescNPI'})
# No. of unique Pharmacy NPIs who dispensed within each thera class
TCls_NPI=CombAllRx.groupby('THERACLASS')['deid_pharmnpi'].nunique().to_frame().rename(columns={'deid_prescribenpi':'TC_CntPrescNPI'})
# No. of unique PCNs who fell under each thera class category prescriptions
TCls_PCNs=CombAllRx.groupby('THERACLASS')['deid_pcn'].nunique().to_frame()
# Theraclass total paid, DF etc.,
TCls_Pd = CombAllRx.groupby(['THERACLASS'])[['PAIDAMT','DISPENSEFEE','Tot_Pd']].apply(sum).rename(columns={'PAIDAMT':'TC_Pd','DISPENSEFEE':'TC_DF','Tot_Pd':'TC_TotPd'}) # Tot, df, paid amt of cmpd RXs and not cmpd RXs per NPI

TClsdfconcats=[TCls_Cnts, TCls_Pd, TCls_PrescNPI, TCls_NPI, TCls_PCNs]
RefFile_TC=pd.concat(TClsdfconcats,axis=1) 
RefFile_TC.columns

# Average claims per thera class
RefFile_TC['AvgPerClm']=RefFile_TC['TC_Pd']/RefFile_TC['TC_NUMOFCLMS']
# Percent composition of claims a thera class constitutes
RefFile_TC['%CompPerTC']=RefFile_TC['TC_NUMOFCLMS']/RefFile_TC['TC_NUMOFCLMS'].sum()
# Percent composition of total paid $$ a thera class constitutes
RefFile_TC['%PdCompPerTC']=RefFile_TC['TC_Pd']/RefFile_TC['TC_Pd'].sum()
# Percent difference between total paid $$ per TC and % of TC composition claims wise. 
# If the diff is +ve it means, the TC pd % is higher than the composition of the claims it composes.
RefFile_TC['%DiffBtwn']=RefFile_TC['%PdCompPerTC']-RefFile_TC['%CompPerTC']
#sns.boxplot(x=RefFile_TC['%DiffBtwn'])
Top5TC=RefFile_TC.nlargest(5,'%DiffBtwn').reset_index()
# RefFile_TC.to_csv('C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/RefFile_TC_NC_2016.csv',index=True)


myTC=Top5TC['THERACLASS']
aggNPI_tmp=CombAllRx[CombAllRx['THERACLASS'].isin(myTC)].groupby(['deid_pharmnpi','THERACLASS']).agg(aggrTotDoll)
aggNPI_tmp.columns = aggNPI_tmp.columns.map('_'.join)
NPI_Top5TCDoll=aggNPI_tmp.reset_index().pivot(index='deid_pharmnpi', columns='THERACLASS')
NPI_Top5TCDoll.columns = NPI_Top5TCDoll.columns.map('_'.join)

#%%# Features 119 - 148
# 9. To find the top 5 AHFS drug categories that have a high difference between the % of No. of claims for the AHFS code 
# and the % of DOllars utilized by the AHFS. That is if 5% of the claims are the composition for a AHFS code
# but this code occupies 10% of the total pais dollars amongst all AHFS codes, then we choose this AHFS, 
# We choose top 5 AHFS codes this way.

# We only look at non-compound claims counts here

# AHFS count of claims in data
AHFS_Cnts=CombAllRx[CombAllRx['COMPOUNDCD']=='N'].groupby(['AHFS']).size().reset_index().rename(columns={0:'AHFS_NUMOFCLMS'})
AHFS_Cnts.set_index('AHFS',inplace=True)
AHFS_Cnts.index.name
#AHFS_Cnts.index.name='AHFS' # Rename index column name
#AHFS_Cnts.index.name
# No. of unique prescribing NPIs for each thera class
AHFS_PrescNPI=CombAllRx[CombAllRx['COMPOUNDCD']=='N'].groupby('AHFS')['deid_prescribenpi'].nunique().to_frame().rename(columns={'deid_prescribenpi':'AHFS_CntPrescNPI'})
# No. of unique Pharmacy NPIs who dispensed within each thera class
AHFS_NPI=CombAllRx[CombAllRx['COMPOUNDCD']=='N'].groupby('AHFS')['deid_pharmnpi'].nunique().to_frame().rename(columns={'deid_prescribenpi':'AHFS_CntPrescNPI'})
# No. of unique PCNs who fell under each thera class category prescriptions
AHFS_PCNs=CombAllRx[CombAllRx['COMPOUNDCD']=='N'].groupby('AHFS')['deid_pcn'].nunique().to_frame()
# AHFS total paid, DF eAHFS.,
AHFS_Pd = CombAllRx[CombAllRx['COMPOUNDCD']=='N'].groupby(['AHFS'])[['PAIDAMT','DISPENSEFEE','Tot_Pd']].apply(sum).rename(columns={'PAIDAMT':'AHFS_Pd','DISPENSEFEE':'AHFS_DF','Tot_Pd':'AHFS_TotPd'}) # Tot, df, paid amt of cmpd RXs and not cmpd RXs per NPI

AHFSdfconcats=[AHFS_Cnts, AHFS_Pd, AHFS_PrescNPI, AHFS_NPI, AHFS_PCNs]
RefFile_AHFS=pd.concat(AHFSdfconcats,axis=1) 
RefFile_AHFS.columns

# Average claims per thera class
RefFile_AHFS['AvgPerClm']=RefFile_AHFS['AHFS_Pd']/RefFile_AHFS['AHFS_NUMOFCLMS']
# Percent composition of claims a thera class constitutes
RefFile_AHFS['%CompPerAHFS']=RefFile_AHFS['AHFS_NUMOFCLMS']/RefFile_AHFS['AHFS_NUMOFCLMS'].sum()
# Percent composition of total paid $$ a thera class constitutes
RefFile_AHFS['%PdCompPerAHFS']=RefFile_AHFS['AHFS_Pd']/RefFile_AHFS['AHFS_Pd'].sum()
# Percent difference between total paid $$ per AHFS and % of AHFS composition claims wise. 
# If the diff is +ve it means, the AHFS pd % is higher than the composition of the claims it composes.
RefFile_AHFS['%DiffBtwn']=RefFile_AHFS['%PdCompPerAHFS']-RefFile_AHFS['%CompPerAHFS']
#RefFile_AHFS.to_csv('C:/Users/Acer/Desktop/Nisha/Nisha Career/PhD/Research/Nisha Rsch/Raw Data/RefFile_AHFS_NC_2016.csv',index=True)

#sns.boxplot(x=RefFile_AHFS['%DiffBtwn'])
#Top5AHFS=RefFile_AHFS.nlargest(5,'%DiffBtwn').reset_index()
Top12AHFS=RefFile_AHFS.nlargest(12,'%DiffBtwn').reset_index()

# There is overlap between AHFS drug class and Therapeutic class, so we find those AHFS classes that do not overlap with Theraclass
# The top 12 AHFS classes are:
# 1. Pituitary drug class 
# 2. Antipsychotics drug class
# 3. Blood formation, coagulation and thrombosis drug class
# 4. Amphetamines drug class                                          - select
# 5. Insulins (Antidiabetic Agents )
# 6. Antipsychotics drug class
# 7. Antipsychotics drug class
# 8. Miscellaneous Therapeutic agents                                 - select
# 9. Miscellaneous GI drugs                                           - select
# 10. Muscle relaxants                                                - select
# 11. Insulins (Antidiabetic Agents )
# 12. Anti-inflammatory agents                                        - select


#myAHFS=Top5AHFS['index']

myAHFS=Top12AHFS.iloc[[3,7,8,9,11]]['AHFS'].reset_index(drop=True)
aggNPI_tmp=CombAllRx[CombAllRx['AHFS'].isin(myAHFS)].groupby(['deid_pharmnpi' ,'AHFS']).agg(aggrTotDoll)
aggNPI_tmp.columns = aggNPI_tmp.columns.map('_'.join)
NPI_Top5AHFSDoll=aggNPI_tmp.reset_index().pivot(index='deid_pharmnpi', columns='AHFS')
NPI_Top5AHFSDoll.columns = NPI_Top5AHFSDoll.columns.map('_'.join)

#%%# # Concatenate to form the feature matrix for each Pharm NPI
dfconcats=[NPI_PCNs, NPI_PrescCnt, NPI_DAW1_Doll, NPI_RxtypeDoll, NPI_MCOtypeDoll, NPI_ClientCatDoll, PMPM_exp['PMPMexp'],NPI_Top5AHFSDoll,NPI_Top5TCDoll]
NPI_Summ=pd.concat(dfconcats,axis=1) 
NPI_Summ.columns
NPI_Summ.fillna(0, inplace=True)
NPI_Summ.columns
#%%# Features 149
# 10. PPR relation establish
# Define a function that counts unique values in a passed vector
def uniqcnt(x):
    return x.nunique()

PPR = {
    'Tot_Pd':['count', 'sum', 'mean', 'median'],
    'deid_pcn':[uniqcnt],
}

aggNPI_tmp = CombAllRx.groupby(['deid_pharmnpi', 'deid_prescribenpi']).agg(PPR)
aggNPI_tmp.columns = aggNPI_tmp.columns.map('_'.join)
#Checks
test=aggNPI_tmp.head(200)
aggNPI_tmp['deid_pcn_uniqcnt'].max()
aggNPI_tmp['deid_pcn_uniqcnt'].quantile(0.75)
#sns.distplot(aggNPI_tmp['deid_pcn_uniqcnt'])
#plt.hist(aggNPI_tmp['deid_pcn_uniqcnt'], bins=2)

aggNPI_tmp['Tot_Pd_count'].max()
aggNPI_tmp['Tot_Pd_count'].quantile(0.75)
# Prescription count by NPI and prescriber
PresCntbyNPI=(NPI_Summ['PAIDAMT_count_H']+NPI_Summ['PAIDAMT_count_N']).to_frame().rename(columns = {0:'PresCnt'}) 
PresCntbyPresc=CombAllRx.groupby(['deid_prescribenpi']).agg(aggrTotDoll)
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
thsh=aggNPI_tmp['deid_pcn_uniqcnt'].quantile(0.75)
aggNPI_tmp[aggNPI_tmp['deid_pcn_uniqcnt']>thsh]['PCNT_Tot'].quantile(0.75)
aggNPI_tmp['PCNT_Tot'].max()

aggNPI_tmp['PCNT_Tot'].quantile(0.75)

# If a NPI and prescriber relation total is more than 75 and the NPI, prec combination had atleast 'thsh' no. of PCNs we flag as collusion
thsh1=NPI_Summ['NPI_PCNCnt'].quantile(0.25)
aggNPI_PrescReln=aggNPI_tmp[(aggNPI_tmp['PCNT_Tot']>75) & (aggNPI_tmp['deid_pcn_uniqcnt']>thsh1)]
aggNPI_PrescReln=aggNPI_PrescReln.reset_index()
aggNPI_P_Rln=aggNPI_PrescReln.groupby(['deid_pharmnpi'])['deid_prescribenpi'].count().reset_index().rename(columns = {'deid_prescribenpi':'PresCnt'})
aggNPI_P_Rln.set_index('deid_pharmnpi', inplace=True)
#%%# Features 150 -154
# 11. Days of supply counts
DYS_Cnts=CombAllRx[CombAllRx['COMPOUNDCD']=='N'].groupby(['DAYSUPPLY']).size().reset_index().rename(columns={0:'DYSSPLY_NUMOFCLMS'})
DYS_Cnts.set_index('DAYSUPPLY',inplace=True)
DYS_Cnts.index.name

myplt=DYS_Cnts.reset_index()
myplt['PCT']=myplt['DYSSPLY_NUMOFCLMS']/sum(myplt['DYSSPLY_NUMOFCLMS'])
#myplt.to_csv('C:/Users/Acer/Desktop/Nisha/Nisha Career/PhD/Research/Nisha Rsch/Raw Data/DYS_CNT.csv',index=False)
fig = plt.figure(figsize = (10, 5)) 
  
# creating the bar plot 
plt.bar(myplt['DAYSUPPLY'], myplt['DYSSPLY_NUMOFCLMS'], color ='maroon',  
        width = 1.0) 
  
plt.xlabel("Days Supply") 
plt.ylabel("No. of claims within each value of days supply") 
plt.title("Distribution of claims based on the days supply") 
plt.show() 

plt.bar(myplt['DAYSUPPLY'], myplt['PCT'], color ='maroon',  
        width = 1.0) 

# 30 day supply is the most often used in the claims. 
# So we look at each provider's 30 day supply claim counts from all their clients, 
# no. of unique clients who fell the 30 day supply claims
DYS30_Cnts=CombAllRx[(CombAllRx['COMPOUNDCD']=='N') & (CombAllRx['DAYSUPPLY']==30)].groupby(['deid_pharmnpi']).agg(PPR)
DYS30_Cnts.columns = DYS30_Cnts.columns.map('_30DS_'.join)
DYS30_Cnts.index.name
#%%#  Features 155 - 170
# 12. Brand Generic prescription counts for each pharmacy NPI
NDCSRC=CombAllRx[CombAllRx['COMPOUNDCD']=='N'].groupby(['NDCSRC']).size().reset_index().rename(columns={0:'NDCSRC_NOFCLMS'})
NDCSRC.set_index('NDCSRC',inplace=True)
NDCSRC.index.name

myplt=NDCSRC.reset_index()
fig = plt.figure(figsize = (10, 5)) 
  
# creating the bar plot 
plt.bar(myplt['NDCSRC'], myplt['NDCSRC_NOFCLMS'], color ='#98d687',  #Tableau green color
        width = 0.5) 
  
plt.xlabel("NDC source") 
plt.ylabel("No. of claims within each value of NDC source") 
plt.title("Distribution of claims based on NDC source") 
plt.show() 

# We look at Single Source, Innovator and Branded drug non-compound claims only for each provider.
NDCSRC_Cnts=CombAllRx[(CombAllRx['COMPOUNDCD']=='N') & (CombAllRx['NDCSRC']!='G')].groupby(['deid_pharmnpi','NDCSRC']).agg(PPR)
NDCSRC_Cnts.columns = NDCSRC_Cnts.columns.map('_'.join)
NDCSRC_type=NDCSRC_Cnts.reset_index().pivot(index='deid_pharmnpi', columns='NDCSRC')
NDCSRC_type.columns = NDCSRC_type.columns.map('_NDCsrc_'.join)
NDCSRC_type.index.name
#%%# Features 171 - 176
# 13. DEA code morphine drug counts for each pharmacy NPI
DEACD2=CombAllRx[(CombAllRx['COMPOUNDCD']=='N')  & (CombAllRx['DEA_CODE']=='2')].groupby(['deid_pharmnpi']).agg(aggrTotDoll)
DEACD2.columns = DEACD2.columns.map('_DEACD2_'.join)
DEACD2.index.name
#%%#
dfconcats=[]
dfconcats=[NPI_PCNs, NPI_PrescCnt, NPI_DAW1_Doll, NPI_RxtypeDoll, NPI_MCOtypeDoll, NPI_ClientCatDoll, PMPM_exp['PMPMexp'],NPI_Top5AHFSDoll,NPI_Top5TCDoll,aggNPI_P_Rln,DYS30_Cnts,NDCSRC_type,DEACD2]
NPI_Summ=pd.concat(dfconcats,axis=1) 
NPI_Summ.columns
NPI_Summ.fillna(0, inplace=True)
NPI_Summ.columns
#NPI_Summ.to_csv('C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/NPI_Summ_2016.csv',index=True)
#%%# 
corr=NPI_Summ.corr()#Default is pearson correlation
#corrp=NPI_Summ.corr(method='pearson')
# plot the heatmap
sns.set(rc={'figure.figsize':(19,15)},font_scale=0.8, style="white")
sns.heatmap(corr, center=0, xticklabels=2, yticklabels=2,cmap='coolwarm',vmin=-1, vmax=1,linewidths=0.2,square=True)
#sns.heatmap(corr, center=0, xticklabels=corr.columns, yticklabels=corr.columns,cmap='coolwarm')
#sns.heatmap(corr, center=0, xticklabels=corr.columns, yticklabels=corr.columns,annot=True, fmt='.2f' ,cmap='coolwarm')
shp=NPI_Summ.shape[0]
matrix = np.tril(np.ones(corr.shape)).astype(np.bool)
NPI_Summ_lt=corr.where(matrix)
sns.set(rc={'figure.figsize':(19,15)},font_scale=0.8, style="white")
sns.heatmap(NPI_Summ_lt, annot=False, center=0, xticklabels=False, yticklabels=False,cmap='coolwarm',vmin=-1, vmax=1,linewidths=0.2)

# Corr matrix for columns > 0.8 values
tst=corr[corr>0.8]
tst_cnt=corr>0.8
Pairs=tst_cnt.sum()-1
Pairs.sum()
matrix2 = np.tril(np.ones(tst.shape)).astype(np.bool)
tst_lt=tst.where(matrix2)
# plot the heatmap
sns.set(rc={'figure.figsize':(19,15)},font_scale=0.8, style="white")
sns.heatmap(tst_lt, center=0, xticklabels=False, yticklabels=False,cmap='coolwarm')
#%%# Principal component Analysis - Prepare the dataset
# Before performing PCA, normalize the features
sns.set(rc={'figure.figsize':(18,10)},font_scale=1.0,style="white")
scaler=MinMaxScaler()
NPI_Summ_Scld=scaler.fit_transform(NPI_Summ)
pc=15 # No. of PCs we want

# Create X tick labels
col_nms=[]
for i in range(pc):
       j=i+1
       mystr='PC'+str(j)
       col_nms.append(mystr)
       
pca100 = PCA(n_components=pc)
my100pc=pca100.fit_transform(NPI_Summ_Scld)
pca100.explained_variance_ratio_

percent_variance=np.round(pca100.explained_variance_ratio_*100, decimals=2)
percent_variance.sum() #85.42%
  
font = {
        'weight' : 'bold',
        'size'   : 40}

plt.rc('font', **font)

plt.figure(figsize=(18,10))
plt.xticks(rotation=0)
plt.bar(x= range(0,pc), height=percent_variance,tick_label=col_nms, color='#98d687')
plt.ylabel('Variance Explained in %', fontsize=16)
plt.xlabel('Principal Components', fontsize=16)
plt.title('15 Component - PCA Scree Plot', fontsize=16)
plt.show()
# plt.savefig('C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Model/Plots/PCA1.jpg')
#%%#
# Plot of explained variance with all 176 components
pca = PCA().fit(NPI_Summ_Scld)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
# plt.savefig('C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Model/Plots/PCA2.jpg')
#%%#  Write features such as myTC, myAHFS as pickled objects to perform same action on 2017 data
with open('C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/pickled_list_myTC.pickle', 'wb') as f:
    pickle.dump(myTC, f)

with open('C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/pickled_list_myAHFS.pickle', 'wb') as f:
    pickle.dump(myAHFS, f)
    
with open('C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/pickled_list_thsh.pickle', 'wb') as f:
    pickle.dump(thsh, f)
    
with open('C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/pickled_list_thsh1.pickle', 'wb') as f:
    pickle.dump(thsh1, f)
    
with open('C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/pickled_list_NPI_Summ_2016.pickle', 'wb') as f:
    pickle.dump(NPI_Summ, f)  

with open('C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/pickled_list_NPI_Summ_Scld_2016.pickle', 'wb') as f:
    pickle.dump(NPI_Summ_Scld, f)  
    
with open('C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/pickled_list_Scaler.pickle', 'wb') as f:
    pickle.dump(scaler, f)    
    
with open('C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/pickled_list_pca100.pickle', 'wb') as f:
    pickle.dump(pca100, f)

# ---------------------------------------------------- 176 FEATURES CREATED AND 15 COMPONENTS (by PCA)---------------------------------------------------------#