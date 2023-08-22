'''
The objective of this file is to merge Refinitiv data and WRDS (compustat) data.

Keywords: Merge
'''

import pandas as pd
import numpy as np

#######################
# 1. firm variable data
#######################

###########################
# 1.1 WRDS (compustat) data
###########################
df_tot = pd.read_csv('data/Compustat data.csv')


###########################
# 1.2 Refinitiv P MV ESG ESGC data
###########################

# active (listed) in NYSE, AMEX, NASDAQ
df_active = pd.read_excel('data/data_active.xlsx', sheet_name=None)
# dict_keys(['nyse_active', 'amex_active', 'nasdaq_active'])

# setting the index of df
def indexized(df):
    '''setting the index'''
    df = df.rename(columns={'Unnamed: 0':'Month'})
    df.set_index('Month', inplace=True)
    return df

df_nyse_active = indexized(df_active['nyse_active'])
df_amex_active = indexized(df_active['amex_active']) 
df_nasdaq_active = indexized(df_active['nasdaq_active']) 

# dead (delisted) in NYSE, AMEX, NASDAQ
df_nyse_dead = pd.read_excel('data/nyse_dead.xlsx', sheet_name=None)
df_amex_dead = pd.read_excel('data/amex_dead.xlsx')
df_nasdaq_dead = pd.read_excel('data/nasdaq_dead.xlsx', sheet_name=None)

df_nyse_dead_t = pd.concat([ indexized(df_nyse_dead['dead 1']), indexized(df_nyse_dead['dead 2']), \
                            indexized(df_nyse_dead['dead 3']), indexized(df_nyse_dead['dead 4']) ], axis=1)
df_amex_dead = indexized(df_amex_dead)
df_nasdaq_dead_t = pd.concat([ indexized(df_nasdaq_dead['nasdaq dead 1']), indexized(df_nasdaq_dead['nasdaq dead 2'])], axis=1)     

# concatenate all df
df_Ref_tot = pd.concat([df_nyse_active, df_amex_active, df_nasdaq_active, df_nyse_dead_t, df_amex_dead, df_nasdaq_dead_t], axis=1) # (253, 106744)
# df_Ref_tot.to_csv('data/df_Ref_tot.csv') # saving total dataframe
df_Ref_tot.shape[1] % 4 == 0 # True: all firm has "P - mv - ESG - ESGC" formats

'''
Recall df_Ref_tot
'''
df_Ref_tot = pd.read_csv('data/df_Ref_tot.csv', dtype='str')
df_Ref_tot.set_index('Month')



# # 분석에 사용할 수 없는 기업 제거
# for i in range(int(df.shape[1]/4)):           
#     # p, mv, esg 및 esgc 데이터가 없는 경우 해당 기업 데이터 삭제
#     if (df['{} - {}'.format(i,criterion)].isnull().sum(axis=0) == df.shape[0]) or (df['{} - {}'.format(i, 'p')].isnull().sum(axis=0) == df.shape[0]) or (df['{} - {}'.format(i, 'mv')].isnull().sum(axis=0) == df.shape[0]) :
#         df.drop(columns=['{} - p'.format(i) , '{} - mv'.format(i), '{} - esg'.format(i), '{} - esgc'.format(i)], inplace=True )



'''
data format:
firmname | firmname - MARKET VALUE | firmname - ESG Score | firmname - ESG Controversies Score
'''

###########################
# 1.3 Refinitiv code data
###########################
'''
The CODE_... files consists of two sheets:
sheet 1 - info (code(symbol) - cusip)
sheet 2 - code (name - code(symbol))
'''
CODE_NASDAQ_act = pd.read_excel('data/CODE_NASDAQ_act.xlsx', sheet_name=None) 
CODE_NASDAQ_dead = pd.read_excel('data/CODE_NASDAQ_dead.xlsx', sheet_name=None) 
CODE_NYSE_act = pd.read_excel('data/CODE_NYSE_act.xlsx', sheet_name=None) 
CODE_NYSE_dead_major_equity = pd.read_excel('data/CODE_NYSE_dead_major_equity.xlsx', sheet_name=None) 
CODE_NYSE_dead_major_nonequity = pd.read_excel('data/CODE_NYSE_dead_major_nonequity.xlsx', sheet_name=None) 
CODE_NYSE_dead_minor = pd.read_excel('data/CODE_NYSE_dead_minor.xlsx', sheet_name=None) 
CODE_NYSEmkt = pd.read_excel('data/CODE_NYSEmkt.xlsx', sheet_name=None) 

# For info sheet, convert nan into Name -> set column Name to index
# For code sheet, convert NaN into Symbol_2
CODE_NASDAQ_act_info = CODE_NASDAQ_act['info']
CODE_NASDAQ_act_code = CODE_NASDAQ_act['code'].rename(columns={np.NaN:'Name'}).set_index(keys='Name') 
CODE_NASDAQ_dead_info = CODE_NASDAQ_dead['info']
CODE_NASDAQ_dead_code = CODE_NASDAQ_dead['code'].rename(columns={np.NaN:'Name'}).set_index(keys='Name') 
CODE_NYSE_act_info = CODE_NYSE_act['info']
CODE_NYSE_act_code = CODE_NYSE_act['code'].rename(columns={np.NaN:'Name'}).set_index(keys='Name') 
CODE_NYSE_dead_major_equity_info = CODE_NYSE_dead_major_equity['info']
CODE_NYSE_dead_major_equity_code = CODE_NYSE_dead_major_equity['code'].rename(columns={np.NaN:'Name'}).set_index(keys='Name') 
CODE_NYSE_dead_major_nonequity_info = CODE_NYSE_dead_major_nonequity['info']
CODE_NYSE_dead_major_nonequity_code = CODE_NYSE_dead_major_nonequity['code'].rename(columns={np.NaN:'Name'}).set_index(keys='Name') 
CODE_NYSE_dead_minor_info = CODE_NYSE_dead_minor['info']
CODE_NYSE_dead_minor_code = CODE_NYSE_dead_minor['code'].rename(columns={np.NaN:'Name'}).set_index(keys='Name') 
CODE_NYSEmkt_info = CODE_NYSEmkt['info']
CODE_NYSEmkt_code = CODE_NYSEmkt['code'].rename(columns={np.NaN:'Name'}).set_index(keys='Name') 

# merge 
df_info = pd.concat([CODE_NASDAQ_act_info, CODE_NASDAQ_dead_info, CODE_NYSE_act_info, CODE_NYSE_dead_major_equity_info, CODE_NYSE_dead_major_nonequity_info, \
    CODE_NYSE_dead_minor_info, CODE_NYSEmkt_info], axis=0, join='inner').reset_index(drop=True)
df_code = pd.concat([CODE_NASDAQ_act_code, CODE_NASDAQ_dead_code, CODE_NYSE_act_code, CODE_NYSE_dead_major_equity_code, CODE_NYSE_dead_major_nonequity_code, \
    CODE_NYSE_dead_minor_code, CODE_NYSEmkt_code], axis=1, join='outer')
df_info.columns

# Drop columns with names containing "#ERROR"
def drop_error_col(df):
    columns_to_keep = ~df.columns.str.contains('#ERROR')
    return df.loc[:, columns_to_keep]
df_code = drop_error_col(df_code) 
df_code.shape # (20, 27599)

# eliminate (P) in 'Code' -> convert to dict
dict_name_code = df_code.loc['Code'][:].replace(to_replace=r'\(P\)',value='',regex=True).to_dict() 
len(dict_name_code) # 27529

# in info sheet, convert to dict (code-cusip)
df_info_temp = df_info.dropna(subset='CUSIP') # drop nan, reduce significantly no of row 
df_info_temp.loc[:, 'CUSIP'] = df_info_temp['CUSIP'].astype('str') # unify int/str as str
dict_code_cusip = df_info_temp.set_index('Symbol')['CUSIP'].to_dict()
len(dict_code_cusip) # 17242

# if name does not match, the corresponding value is '-1'
dict_name_cusip = {}
for key, value in dict_name_code.items():
    if value in dict_code_cusip: # match
        dict_name_cusip[key] = dict_code_cusip[value]
    else: # not match
        dict_name_cusip[key] = '-1'
len(dict_name_cusip)
list(dict_name_cusip.items())[:5]

dict_name_cusip



'''
임시

이것을 가지고 무엇을 할지 고민하기
info에서 code - cusip dict 생성 (중복 있는지 확인)
code에서 name - code cusip dict 생성
둘을 이용해서 name - cusip dict 생성 

'''




'''
If you want to divide the values of the overlapping columns between two DataFrames, 
you can do so by matching the columns and performing element-wise division. 
You can use the .div() method in Pandas to achieve this. Here's how you can do it:
'''
# Sample DataFrames
data1 = {'a': [10, 20, 30], 'b': [5, 10, 15]}
data2 = {'a': [2, 4, 6], 'b': [1, 2, 3], 'c': [0.1, 0.2, 0.3]}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Divide overlapping columns
overlapping_columns = df1.columns.intersection(df2.columns)  # Find common columns

result_df = df1[overlapping_columns].div(df2[overlapping_columns])

print(result_df)
