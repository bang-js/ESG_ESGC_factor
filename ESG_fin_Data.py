'''
The objective of this file is to merge Refinitiv data and WRDS (compustat) data.

Keywords: Merge
'''

import pandas as pd
import numpy as np
import sys

###########################
###########################
# 1. firm variable data
###########################
###########################

###########################
# 1.1 WRDS (compustat) data
###########################

# load compustat data
df_compustat = pd.read_csv('data/Compustat data.csv', index_col=[0,1]) # multiindex

# Extract existant cusip sequence from df_compustat 
# NOTION: use in section 1.3 Merge
cusip_compustat = df_compustat['cusip'].unique() 
len(cusip_compustat) # 25452

# Filter the DataFrame to keep only rows with NYSE (11), AMEX (american SE, 12), NASDAQ (14) in the specified column
df_compustat_sel = df_compustat[df_compustat['exchg'].isin([11.0, 12.0, 14.0])] # [132673 rows x 23 columns]

def pivot_per_var(df_tot, variable_):
    '''
    To make a new dataframe whose columns are firm names and rows are fiscal years.
    Parameter:
        variable_: target variable to extract from df_tot
    '''
    # Create a copy to avoid modifying the original DataFrame
    df_tot_copy = df_tot.copy()

    # Drop rows without 'fyear'
    df_tot_copy.dropna(axis=0, subset=['fyear'], inplace=True)
    
    # Convert 'fyear' to datetime and extract year
    df_tot_copy.loc[:, 'fyear'] = pd.to_datetime(df_tot_copy['fyear'], format='%Y').dt.year 

    # Create a pivot table
    pivot_table = df_tot_copy.pivot_table(index='fyear', columns='cusip', values=variable_)

    return pivot_table

# Make a new dataframe whose columns are firm names and rows are fiscal years.
at_df = pivot_per_var(df_compustat_sel, 'at') # total asset                               
sale_df = pivot_per_var(df_compustat_sel, 'sale') # sales
ni_df = pivot_per_var(df_compustat_sel, 'ni') # net income
ch_df = pivot_per_var(df_compustat_sel, 'ch') # cash (flow)
che_df = pivot_per_var(df_compustat_sel, 'che') # Cash and Short-Term Investments         
oancf_df = pivot_per_var(df_compustat_sel, 'oancf') # Operating Activities Net Cash Flow 
lt_df = pivot_per_var(df_compustat_sel, 'lt') # liabilities
dvc_df = pivot_per_var(df_compustat_sel, 'dvc') # Dividends Common/Ordinary 
xad_df = pivot_per_var(df_compustat_sel, 'xad') # Ads expenditure                          
 
# Print out the shape of each df
df_lst_name = ['at_df', ' sale_df', ' ni_df', ' ch_df', ' che_df', ' oancf_df', ' lt_df', ' dvc_df', ' xad_df']
df_lst = [at_df, sale_df, ni_df, ch_df, che_df, oancf_df, lt_df, dvc_df, xad_df]
[print(f'{df_lst_name[_]:10}{df_lst[_].shape}') for _ in range(len(df_lst_name))]
'''
 at_df     (19, 10721)
 sale_df  (19, 10608)
 ni_df    (19, 10607)
 ch_df    (19, 10695)
 che_df   (19, 10720)
 oancf_df (19, 10488)
 lt_df    (19, 10717)
 dvc_df   (19, 10076)
 xad_df   (19, 4614)'''

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
df_Ref_tot = df_Ref_tot.set_index('Month')

# Convert strings to numbers where possible, replace non-numeric strings with NaN
# eliminate '$$ER: 9898,NO DATA AVAILABLE' etc
df_Ref_tot = df_Ref_tot.apply(pd.to_numeric, errors='coerce')

# Drop columns with names containing '#ERROR'
df_Ref_tot = df_Ref_tot.drop(columns=[col for col in df_Ref_tot.columns if '#ERROR' in col])    # [253 rows x 55120 columns]

# Seperate to four dataframes
df_MV = df_Ref_tot.loc[:,df_Ref_tot.columns.str.contains(' - MARKET VALUE')]                    # [253 rows x 20160 columns]
df_ESG = df_Ref_tot.loc[:,df_Ref_tot.columns.str.contains(' - ESG Score')]                      # [253 rows x 6517 columns]
df_ESGC = df_Ref_tot.loc[:,df_Ref_tot.columns.str.contains(' - ESG Controversies Score')]       # [253 rows x 6517 columns]
df_P = df_Ref_tot.loc[:, ~df_Ref_tot.columns.str.contains(' - MARKET VALUE| - ESG Score| - ESG Controversies Score')] # [253 rows x 21926 columns]

def elimword_colname(df, word):
    '''
    To eliminate the given word from the names of columns
    Parameter
        df: the objective of operation
        word: word to eliminate
    '''
    # Create a dictionary to rename columns
    rename_dict = {col: col.replace(word, '') for col in df.columns}
    # Rename columns using the dictionary
    df = df.rename(columns=rename_dict)
    return df
df_MV = elimword_colname(df_MV, ' - MARKET VALUE')
df_ESG = elimword_colname(df_ESG, ' - ESG Score')
df_ESGC = elimword_colname(df_ESGC, ' - ESG Controversies Score')

def erase_notchg_value(df):
    '''
    if price and market value does not change during more than 2 months, convert float to NaN after the second element.
    ex) 12.69 12.69 12.69 -> 12.69 NaN NaN
    '''
    df = df.apply(lambda col: np.where(col == col.shift(), np.nan, col))
    return df
df_P = erase_notchg_value(df_P)
df_MV = erase_notchg_value(df_MV)


###########################
# 1.3 Refinitiv name -> cusip
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
# df_info.to_csv('data/df_info.csv')
df_info = pd.read_csv('data/df_info.csv', index_col=0)


#########VER1########
'''
Indirectly connect name and cusip using code sheet and info sheet
1. in code sheet, match name (format of retrieved data) and code
=> CAUTION: the name format of info sheet and that of code sheet are DIFFERENT! 
=> Thus the name in code sheet MUST be used to transform the firmname cols of (old) Reftv data to CUSIP
2. 
'''
# DEAD - DELIST 사이의 띄어쓰기...
# name -> code -> dict
# eliminate (P) in 'Code' -> convert to dict
dict_name_code = df_code.loc['Code'][:].replace(to_replace=r'\(P\)',value='',regex=True).to_dict() 
len(dict_name_code) # 30648
list(dict_name_code.items())[-10:] 
# 

# in info sheet, convert to dict (code-cusip)
df_info_temp = df_info.dropna(subset='CUSIP') # drop nan, reduce significantly no of row 
df_info_temp.loc[:, 'CUSIP'] = df_info_temp['CUSIP'].astype('str') # unify int/str as str
dict_code_cusip = df_info_temp.set_index('Symbol')['CUSIP'].to_dict()
len(dict_code_cusip) # 17242

########VER2#########
'''
Directly connect name - cusip in info sheet
CAUTION: FULFILL for !! 9 digit !! to merge with compustat data (0x9)
'''
df_info_temp = df_info.dropna(subset='CUSIP') # drop nan, reduce significantly no of row 
df_info_temp.loc[:,'CUSIP'] = df_info_temp['CUSIP'].astype('str').apply(lambda x: x.zfill(9)) # unify int/str as str 
dict_name_cusip = df_info_temp.set_index('Name')['CUSIP'].to_dict() # convert to dict (code-cusip)
len(dict_name_cusip) # 17125
list(dict_name_cusip.items())[-100:] # [('APPLE', '037833100'), ('MICROSOFT', '594918104'), ('AMAZON.COM', '023135106')] # The cusip codes must be 9 digits with 0

def col_match_name_cusip(df):
    '''
    To convert the columns of the dataframe from name to cusip
    '''
    # lst_cusip: cusip sequence that is matched between Reftv_firmname (columns) and dict_name_cusip
    lst_cusip = []
    for name in df.columns:
        if name in dict_name_cusip:
            lst_cusip.append(dict_name_cusip[name]) 

    # Rename dataframe from firmaname to cusip by matching coumns and dict_name_cusipl
    df_ = df.copy()
    df_.rename(columns=dict_name_cusip, inplace=True) # use dict to perfectly match 
    
    # remain only cusip 
    return df_.loc[:,list(set(cusip_compustat) & set(lst_cusip))]

df_P_match = col_match_name_cusip(df_P)             # [253 rows x 5173 columns]
df_MV_match = col_match_name_cusip(df_MV)
df_ESG_match = col_match_name_cusip(df_ESG)
df_ESGC_match = col_match_name_cusip(df_ESGC)
 # 253 rows x 3945 columns


##############
# 1.4 Merge Reftv and Compustat
##############


df_lst = [at_df, sale_df, ni_df, ch_df, che_df, oancf_df, lt_df, dvc_df, xad_df]

df




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


# ### ver 1. list로 덮어씌우기
# # df_ESG_col_new: Reftv firm name을 dict_name_cusip으로 연결하여 얻은 cusip sequence, 대응되지 않는 요소는 기존의 name을 사용
# df_ESG_col_new = []
# for name in df_ESG.columns:
#     if name in dict_name_cusip:
#         df_ESG_col_new.append(dict_name_cusip[name])
#     else:
#         df_ESG_col_new.append(name)

# # df_ESG_copy의 column 이름을 df_ESG_col_new로 변경
# df_ESG_new = df_ESG.copy()
# df_ESG_new.columns = df_ESG_col_new
# df_ESG_new.loc[:,lst_cusip] # 253 rows x 4565 columns

# lst_cusip_comp = []
# for c in cusip_compustat:
#     if c in df_ESG_col_new:
#         lst_cusip_comp.append(c)
# df_ESG_new.loc[:,lst_cusip_comp] # 253 rows x 3945 columns


######
# df_info.columns
# df_P.columns[-30:]
# check_lst = ['DEAD', 'DELIST']
# for _ in df_info.loc[-100:,'Name']:
#     for c in check_lst:
#         if c in _: 
#             print(_)
#     # 
# ######