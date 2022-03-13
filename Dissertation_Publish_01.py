# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 15:13:30 2020
Read Raw data and create combined files for data preprocessing
@author: nisha
"""
#%%# Library imports
import pandas as pd
import pickle

import datetime
from datetime import date

import time
import psutil
import multiprocessing as mp
#%%#
print("This kernel has ",mp.cpu_count(),"cores and the information on the memory usage currently executing this code is:",psutil.virtual_memory())
#%%# Read Raw Data by Calendar quarters for year 2016
# Jan 2016 to March 2016
Rawdat1=pd.read_table("file:///C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/Pharm DeID 010116 to 0033116.txt",\
                      sep='\t',\
                      dtype={'Source':'str','RX_FILL_DT':'str','RX_DT':'str','RXNBR':'str','REFILLNBR':'uint8',\
                             'DAYSUPPLY':'uint16','AUTHREFILL':'uint8','HDRSTATUS':'str','SEQNBR':'str','ICN':'str','NDC':'str',\
                             'LABELNAME':'str','AHFS':'str','RXQTY':'str','DAWCD':'str','DISPENSEFEE':'float64','PAIDAMT':'float64',\
                             'NDCSRC':'str','NDCSRCDESC':'str','DEACD':'str','COMPOUNDCD':'str','DAWCD':'str','THERACLASS':'str',\
                             'THERADESC':'str', 'MAINT_DRUG_IND':'str','DEA_CODE':'str',\
                            'clientcat':'str','mcoprog':'str','deid_pcn':'str',\
                            'deid_pharmnpi':'str','deid_prescribenpi':'str'})
#Rawdat1.dtypes
# Convert RX fill date and RX date to datetime objects and drop the original columns
Rawdat1['RX_filldt']=Rawdat1['RX_FILL_DT'].str[:9].apply(lambda x: datetime.datetime.strptime(x,'%d%b%Y'))
Rawdat1['RX_dt']=Rawdat1['RX_DT'].str[:9].apply(lambda x: datetime.datetime.strptime(x,'%d%b%Y'))
Rawdat1['Tot_Pd']=Rawdat1.apply(lambda x: x['PAIDAMT']+x['DISPENSEFEE'],axis=1)
Rawdat1.drop(['RX_FILL_DT', 'RX_DT', 'HDRSTATUS'], axis=1,inplace=True)
Rawdat1['mcoprog'].fillna(value='FFS',inplace=True)
Rawdat1['MAINT_DRUG_IND'].fillna(value='0',inplace=True)
# April 2016 to June 2016
Rawdat2=pd.read_table('file:///C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/Pharm DeID 040116 to 063016.txt',\
                      sep='\t',\
                      dtype={'Source':'str','RX_FILL_DT':'str','RX_DT':'str','RXNBR':'str','REFILLNBR':'uint8',\
                             'DAYSUPPLY':'uint16','AUTHREFILL':'uint8','HDRSTATUS':'str','SEQNBR':'str','ICN':'str','NDC':'str',\
                             'LABELNAME':'str','AHFS':'str','RXQTY':'str','DAWCD':'str','DISPENSEFEE':'float64','PAIDAMT':'float64',\
                             'NDCSRC':'str','NDCSRCDESC':'str','DEACD':'str','COMPOUNDCD':'str','DAWCD':'str','THERACLASS':'str',\
                             'THERADESC':'str', 'MAINT_DRUG_IND':'str','DEA_CODE':'str',\
                            'clientcat':'str','mcoprog':'str','deid_pcn':'str',\
                            'deid_pharmnpi':'str','deid_prescribenpi':'str'})
#Rawdat2.dtypes
# Convert RX fill date and RX date to datetime objects and drop the original columns
Rawdat2['RX_filldt']=Rawdat2['RX_FILL_DT'].str[:9].apply(lambda x: datetime.datetime.strptime(x,'%d%b%Y'))
Rawdat2['RX_dt']=Rawdat2['RX_DT'].str[:9].apply(lambda x: datetime.datetime.strptime(x,'%d%b%Y'))
Rawdat2['Tot_Pd']=Rawdat2.apply(lambda x: x['PAIDAMT']+x['DISPENSEFEE'],axis=1)
Rawdat2.drop(['RX_FILL_DT', 'RX_DT', 'HDRSTATUS'], axis=1,inplace=True)
Rawdat2['mcoprog'].fillna(value='FFS',inplace=True)
Rawdat2['MAINT_DRUG_IND'].fillna(value='0',inplace=True)
# July 2016 to Sep 2016
Rawdat3=pd.read_table('file:///C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/Pharm DeID 070116 to 093016.txt',\
                      sep='\t',\
                      dtype={'Source':'str','RX_FILL_DT':'str','RX_DT':'str','RXNBR':'str','REFILLNBR':'uint8',\
                             'DAYSUPPLY':'uint16','AUTHREFILL':'uint8','HDRSTATUS':'str','SEQNBR':'str','ICN':'str','NDC':'str',\
                             'LABELNAME':'str','AHFS':'str','RXQTY':'str','DAWCD':'str','DISPENSEFEE':'float64','PAIDAMT':'float64',\
                             'NDCSRC':'str','NDCSRCDESC':'str','DEACD':'str','COMPOUNDCD':'str','DAWCD':'str','THERACLASS':'str',\
                             'THERADESC':'str', 'MAINT_DRUG_IND':'str','DEA_CODE':'str',\
                            'clientcat':'str','mcoprog':'str','deid_pcn':'str',\
                            'deid_pharmnpi':'str','deid_prescribenpi':'str'})
#Rawdat3.dtypes
# Convert RX fill date and RX date to datetime objects and drop the original columns
Rawdat3['RX_filldt']=Rawdat3['RX_FILL_DT'].str[:9].apply(lambda x: datetime.datetime.strptime(x,'%d%b%Y'))
Rawdat3['RX_dt']=Rawdat3['RX_DT'].str[:9].apply(lambda x: datetime.datetime.strptime(x,'%d%b%Y'))
Rawdat3['Tot_Pd']=Rawdat3.apply(lambda x: x['PAIDAMT']+x['DISPENSEFEE'],axis=1)
Rawdat3.drop(['RX_FILL_DT', 'RX_DT', 'HDRSTATUS'], axis=1,inplace=True)
Rawdat3['mcoprog'].fillna(value='FFS',inplace=True)
Rawdat3['MAINT_DRUG_IND'].fillna(value='0',inplace=True)
# Oct 2016 to Dec 2016
Rawdat4=pd.read_table('file:///C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/Pharm DeID 100116 to 123116.txt',\
                      sep='\t',\
                      dtype={'Source':'str','RX_FILL_DT':'str','RX_DT':'str','RXNBR':'str','REFILLNBR':'uint8',\
                             'DAYSUPPLY':'uint16','AUTHREFILL':'uint8','HDRSTATUS':'str','SEQNBR':'str','ICN':'str','NDC':'str',\
                             'LABELNAME':'str','AHFS':'str','RXQTY':'str','DAWCD':'str','DISPENSEFEE':'float64','PAIDAMT':'float64',\
                             'NDCSRC':'str','NDCSRCDESC':'str','DEACD':'str','COMPOUNDCD':'str','DAWCD':'str','THERACLASS':'str',\
                             'THERADESC':'str', 'MAINT_DRUG_IND':'str','DEA_CODE':'str',\
                            'clientcat':'str','mcoprog':'str','deid_pcn':'str',\
                            'deid_pharmnpi':'str','deid_prescribenpi':'str'})
#Rawdat4.dtypes
# Convert RX fill date and RX date to datetime objects and drop the original columns
Rawdat4['RX_filldt']=Rawdat4['RX_FILL_DT'].str[:9].apply(lambda x: datetime.datetime.strptime(x,'%d%b%Y'))
Rawdat4['RX_dt']=Rawdat4['RX_DT'].str[:9].apply(lambda x: datetime.datetime.strptime(x,'%d%b%Y'))
Rawdat4['Tot_Pd']=Rawdat4.apply(lambda x: x['PAIDAMT']+x['DISPENSEFEE'],axis=1)
Rawdat4.drop(['RX_FILL_DT', 'RX_DT', 'HDRSTATUS'], axis=1,inplace=True)
Rawdat4['mcoprog'].fillna(value='FFS',inplace=True)
Rawdat4['MAINT_DRUG_IND'].fillna(value='0',inplace=True)
#%%# Combine Raw Data
Comb2016=pd.concat([Rawdat1,Rawdat2,Rawdat3,Rawdat4], ignore_index=True)
# Compound RX - without ingredients + Non compound RXs dataset
Comb2016[Comb2016['COMPOUNDCD']!='C'].to_csv('C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/ComballRX_2016',index=False)
#%%# Delete rawdat 1 to 4 dataframes
del Rawdat1
del Rawdat2
del Rawdat3
del Rawdat4
#%%#
print("This kernel has ",mp.cpu_count(),"cores and the information on the memory usage currently executing this code is:",psutil.virtual_memory())
#%%# Read Raw Data by Calendar quarters for year 2017 
# Jan 2017 to March 2017
Rawdat5=pd.read_table('file:///C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/Pharm DeID 010117 to 033117.txt',\
                      sep='\t',\
                      dtype={'Source':'str','RX_FILL_DT':'str','RX_DT':'str','RXNBR':'str','REFILLNBR':'uint8',\
                             'DAYSUPPLY':'uint16','AUTHREFILL':'uint8','HDRSTATUS':'str','SEQNBR':'str','ICN':'str','NDC':'str',\
                             'LABELNAME':'str','AHFS':'str','RXQTY':'str','DAWCD':'str','DISPENSEFEE':'float64','PAIDAMT':'float64',\
                             'NDCSRC':'str','NDCSRCDESC':'str','DEACD':'str','COMPOUNDCD':'str','DAWCD':'str','THERACLASS':'str',\
                             'THERADESC':'str', 'MAINT_DRUG_IND':'str','DEA_CODE':'str',\
                            'clientcat':'str','mcoprog':'str','deid_pcn':'str',\
                            'deid_pharmnpi':'str','deid_prescribenpi':'str'})
#Rawdat5.dtypes
# Convert RX fill date and RX date to datetime objects and drop the original columns
Rawdat5['RX_filldt']=Rawdat5['RX_FILL_DT'].str[:9].apply(lambda x: datetime.datetime.strptime(x,'%d%b%Y'))
Rawdat5['RX_dt']=Rawdat5['RX_DT'].str[:9].apply(lambda x: datetime.datetime.strptime(x,'%d%b%Y'))
Rawdat5['Tot_Pd']=Rawdat5.apply(lambda x: x['PAIDAMT']+x['DISPENSEFEE'],axis=1)
Rawdat5.drop(['RX_FILL_DT', 'RX_DT', 'HDRSTATUS'], axis=1,inplace=True)
Rawdat5['mcoprog'].fillna(value='FFS',inplace=True)
Rawdat5['MAINT_DRUG_IND'].fillna(value='0',inplace=True)
Rawdat5.columns# View columns of the dataframe
# April 2017 to June 2017
Rawdat6=pd.read_table('file:///C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/Pharm DeID 040117 to 063017.txt',\
                      sep='\t',\
                      dtype={'Source':'str','RX_FILL_DT':'str','RX_DT':'str','RXNBR':'str','REFILLNBR':'uint8',\
                             'DAYSUPPLY':'uint16','AUTHREFILL':'uint8','HDRSTATUS':'str','SEQNBR':'str','ICN':'str','NDC':'str',\
                             'LABELNAME':'str','AHFS':'str','RXQTY':'str','DAWCD':'str','DISPENSEFEE':'float64','PAIDAMT':'float64',\
                             'NDCSRC':'str','NDCSRCDESC':'str','DEACD':'str','COMPOUNDCD':'str','DAWCD':'str','THERACLASS':'str',\
                             'THERADESC':'str', 'MAINT_DRUG_IND':'str','DEA_CODE':'str',\
                            'clientcat':'str','mcoprog':'str','deid_pcn':'str',\
                            'deid_pharmnpi':'str','deid_prescribenpi':'str'})
#Rawdat6.dtypes
# Convert RX fill date and RX date to datetime objects and drop the original columns
Rawdat6['RX_filldt']=Rawdat6['RX_FILL_DT'].str[:9].apply(lambda x: datetime.datetime.strptime(x,'%d%b%Y'))
Rawdat6['RX_dt']=Rawdat6['RX_DT'].str[:9].apply(lambda x: datetime.datetime.strptime(x,'%d%b%Y'))
Rawdat6['Tot_Pd']=Rawdat6.apply(lambda x: x['PAIDAMT']+x['DISPENSEFEE'],axis=1)
Rawdat6.drop(['RX_FILL_DT', 'RX_DT', 'HDRSTATUS'], axis=1,inplace=True)
Rawdat6['mcoprog'].fillna(value='FFS',inplace=True)
Rawdat6['MAINT_DRUG_IND'].fillna(value='0',inplace=True)
# July 2017 to Sep 2017
Rawdat7=pd.read_table('file:///C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/Pharm DeID 070117 to 093017.txt',\
                      sep='\t',\
                      dtype={'Source':'str','RX_FILL_DT':'str','RX_DT':'str','RXNBR':'str','REFILLNBR':'uint8',\
                             'DAYSUPPLY':'uint16','AUTHREFILL':'uint8','HDRSTATUS':'str','SEQNBR':'str','ICN':'str','NDC':'str',\
                             'LABELNAME':'str','AHFS':'str','RXQTY':'str','DAWCD':'str','DISPENSEFEE':'float64','PAIDAMT':'float64',\
                             'NDCSRC':'str','NDCSRCDESC':'str','DEACD':'str','COMPOUNDCD':'str','DAWCD':'str','THERACLASS':'str',\
                             'THERADESC':'str', 'MAINT_DRUG_IND':'str','DEA_CODE':'str',\
                            'clientcat':'str','mcoprog':'str','deid_pcn':'str',\
                            'deid_pharmnpi':'str','deid_prescribenpi':'str'})
#Rawdat7.dtypes
# Convert RX fill date and RX date to datetime objects and drop the original columns
Rawdat7['RX_filldt']=Rawdat7['RX_FILL_DT'].str[:9].apply(lambda x: datetime.datetime.strptime(x,'%d%b%Y'))
Rawdat7['RX_dt']=Rawdat7['RX_DT'].str[:9].apply(lambda x: datetime.datetime.strptime(x,'%d%b%Y'))
Rawdat7['Tot_Pd']=Rawdat7.apply(lambda x: x['PAIDAMT']+x['DISPENSEFEE'],axis=1)
Rawdat7.drop(['RX_FILL_DT', 'RX_DT', 'HDRSTATUS'], axis=1,inplace=True)
Rawdat7['mcoprog'].fillna(value='FFS',inplace=True)
Rawdat7['MAINT_DRUG_IND'].fillna(value='0',inplace=True)
# Oct 2017 to Dec 2017
Rawdat8=pd.read_table('file:///C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/Pharm DeID 100117 to 123117.txt',\
                      sep='\t',\
                      dtype={'Source':'str','RX_FILL_DT':'str','RX_DT':'str','RXNBR':'str','REFILLNBR':'uint8',\
                             'DAYSUPPLY':'uint16','AUTHREFILL':'uint8','HDRSTATUS':'str','SEQNBR':'str','ICN':'str','NDC':'str',\
                             'LABELNAME':'str','AHFS':'str','RXQTY':'str','DAWCD':'str','DISPENSEFEE':'float64','PAIDAMT':'float64',\
                             'NDCSRC':'str','NDCSRCDESC':'str','DEACD':'str','COMPOUNDCD':'str','DAWCD':'str','THERACLASS':'str',\
                             'THERADESC':'str', 'MAINT_DRUG_IND':'str','DEA_CODE':'str',\
                            'clientcat':'str','mcoprog':'str','deid_pcn':'str',\
                            'deid_pharmnpi':'str','deid_prescribenpi':'str'})
#Rawdat8.dtypes
# Convert RX fill date and RX date to datetime objects and drop the original columns
Rawdat8['RX_filldt']=Rawdat8['RX_FILL_DT'].str[:9].apply(lambda x: datetime.datetime.strptime(x,'%d%b%Y'))
Rawdat8['RX_dt']=Rawdat8['RX_DT'].str[:9].apply(lambda x: datetime.datetime.strptime(x,'%d%b%Y'))
Rawdat8['Tot_Pd']=Rawdat8.apply(lambda x: x['PAIDAMT']+x['DISPENSEFEE'],axis=1)
Rawdat8.drop(['RX_FILL_DT', 'RX_DT', 'HDRSTATUS'], axis=1,inplace=True)
Rawdat8['mcoprog'].fillna(value='FFS',inplace=True)
Rawdat8['MAINT_DRUG_IND'].fillna(value='0',inplace=True)
#%%# Combine Raw Data
Comb2017=pd.concat([Rawdat5,Rawdat6,Rawdat7,Rawdat8], ignore_index=True)
# Compound RX - without ingredients + Non compound RXs dataset
Comb2017[Comb2017['COMPOUNDCD']!='C'].to_csv('C:/Users/Acer/Desktop/Nisha/UT Box/Box Sync/Nisha FA Research/Nisha Rsch/Raw Data/ComballRX_2017',index=False)
#%%# Delete rawdat 5 to 8 dataframes
del Rawdat5
del Rawdat6
del Rawdat7
del Rawdat8
#%%# Raw Data info for Yr 2016 and Yr 2017
Comb2016.info()
Comb2017.info()
#%%  Data Checks
Comb2016['DEACD'].value_counts()
Comb2016['DEA_CODE'].value_counts()
Comb2016['COMPOUNDCD'].value_counts()
CmpdRX=Comb2016[Comb2016['COMPOUNDCD']!='N']
CombNC=Comb2016[Comb2016['COMPOUNDCD']=='N']
Comb2016.columns#Lists all columns in our final dataframe
#%%# Find datatypes of each column before export
# Get a Series object containing the data type objects of each column of Dataframe.
# Index of series is column name.
dataTypeSeries = Comb2016.dtypes
print('Data type of each column of Dataframe :')
print(dataTypeSeries)


