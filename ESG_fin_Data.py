'''
The objective of this file is to merge Refinitiv data and WRDS (compustat) data.

Keywords: Merge
'''

import pandas as pd
import numpy as np

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

# firms with ZERO and NEGATIVE assets/sales
at_df.loc[:, at_df.columns[at_df[at_df <= 0.0].any()]]
sale_df.loc[:, sale_df.columns[sale_df[sale_df <= 0.0].any()]]
che_df.loc[:,che_df.columns[che_df[che_df <= 0.0].any()]] # cash and short term invest negative?? 
xad_df.loc[:,xad_df.columns[xad_df[xad_df <= 0.0].any()]] 

# zero and negative assets -> NaN
at_df = at_df.applymap(lambda x: np.NaN if x <= 0.0 else x)
sale_df = sale_df.applymap(lambda x: np.NaN if x <= 0.0 else x)
che_df = che_df.applymap(lambda x: np.NaN if x <= 0.0 else x)
xad_df = xad_df.applymap(lambda x: np.NaN if x <= 0.0 else x)

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
df_Ref_tot = pd.read_csv('data/df_Ref_tot.csv', dtype='str') # 30seconds
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
# df_info = pd.read_csv('data/df_info.csv', index_col=0)


#########VER1########
'''
Indirectly connect name and cusip using code sheet and info sheet
1. in code sheet, match name (format of retrieved data) and code
=> CAUTION: the name format of info sheet and that of code sheet are DIFFERENT! 
=> ex) space b/t "DEAD - DELIST"
=> Thus the name in code sheet MUST be used to transform the firmname cols of (old) Reftv data to CUSIP
2. in info sheet, 
'''
# dict: name - code
# eliminate (P) in 'Code' -> convert to dict
dict_name_code = df_code.loc['Code'][:].replace(to_replace=r'\(P\)',value='',regex=True).to_dict() 
len(dict_name_code) # 30648
list(dict_name_code.items())[-10:] 

# dict: code - name
# in info sheet, convert to dict (code-cusip)
df_info_temp = df_info.dropna(subset='CUSIP') # drop nan, reduce significantly no of row 
df_info_temp.loc[:,'CUSIP'] = df_info_temp['CUSIP'].astype('str').apply(lambda x: x.zfill(9)) # unify int/str as str and fill ZERO for 9-digit
dict_code_cusip = df_info_temp.set_index('Symbol')['CUSIP'].to_dict()
len(dict_code_cusip) # 17242
list(dict_code_cusip.items())[:5] # CAUTION: check whether cusip is 9-digit format. ex) ('@AMZN', '023135106')

# dict: name - cusip
# store only matched name (with cusip)
dict_name_cusip = {}
for key, value in dict_name_code.items():
    if value in dict_code_cusip:
        dict_name_cusip[key] = dict_code_cusip[value]
len(dict_name_cusip)
list(dict_name_cusip.items())[-10:]

########VER2#########
'''
Directly connect name - cusip in info sheet
CAUTION: FULFILL for !! 9 digit !! to merge with compustat data (0x9)
'''
# df_info_temp = df_info.dropna(subset='CUSIP') # drop nan, reduce significantly no of row 
# df_info_temp.loc[:,'CUSIP'] = df_info_temp['CUSIP'].astype('str').apply(lambda x: x.zfill(9)) # unify int/str as str and fill ZERO for 9-digit
# dict_name_cusip_2 = df_info_temp.set_index('Name')['CUSIP'].to_dict() # convert to dict (code-cusip)
# len(dict_name_cusip_2) # 17125
# list(dict_name_cusip_2.items())[-10:] # [('APPLE', '037833100'), ('MICROSOFT', '594918104'), ('AMAZON.COM', '023135106')] 
# # CAUTION: check whether cusip is 9-digit format. ex) ('@AMZN', '023135106')

####################
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

# Match name - cusip for four dataframes
df_P_match = col_match_name_cusip(df_P)             # [253 rows x 8246 columns] <- [253 rows x 21926 columns]
df_MV_match = col_match_name_cusip(df_MV)           # [253 rows x 8242 columns]
df_ESG_match = col_match_name_cusip(df_ESG)         # [253 rows x 4322 columns]
df_ESGC_match = col_match_name_cusip(df_ESGC)       # [253 rows x 4322 columns]

# Convert index format from object to datetime
df_P_match.index = pd.to_datetime(df_P_match.index)
df_MV_match.index = pd.to_datetime(df_MV_match.index)
df_ESG_match.index = pd.to_datetime(df_ESG_match.index)
df_ESGC_match.index = pd.to_datetime(df_ESGC_match.index)

def conv_m2y(df):
    '''
    To convert the index frequency from month to year
    Return:
        df with index that has the format of 'YEAR' (not Y-m-D)
    '''
    df_temp = df.resample(rule='Y').last()
    df_temp.index = df_temp.index.year
    df_temp.index.name = 'year'
    return df_temp

# Convert the index frequency of ESG and ESGC from month to year
df_ESG_match = conv_m2y(df_ESG_match)
df_ESGC_match = conv_m2y(df_ESGC_match)

# cut after 2020
df_ESG_match = df_ESG_match[df_ESG_match.index < 2021]
df_ESGC_match = df_ESGC_match[df_ESGC_match.index < 2021]

# # save
# df_P_match.to_csv('df_P.csv')
# df_MV_match.to_csv('df_MV.csv')

# describe
df_ESG_match.apply(pd.DataFrame.describe, axis=1).round(2) # by year
df_ESG_match.stack().describe().round(2) # all
df_ESGC_match.stack().describe().round(2) # all


### ESG Controversy change ###
df_ESGC_Chg_match = df_ESGC_match.diff()
df_ESGC_Chg_match.stack().describe().round(2)
df_ESGC_Chg_match.stack().quantile(np.arange(0,0.25,0.01)).round(2) # after 0.17, go zero
df_ESGC_Chg_match.stack().quantile(np.arange(0.75,1,0.01)).round(2) # before 0.86, go zero

######################################################################
# 1.4 Calculate CONTROLLED ESG dataframe: Merge Reftv and Compustat
######################################################################
'''
If you want to divide the values of the overlapping columns between two DataFrames, 
you can do so by matching the columns and performing element-wise division. 
You can use the .div() method in Pandas to achieve this. Here's how you can do it:
'''
def cal_two_df(df1, df2, mode):
    '''
    To calculate two dataframes 
    Parameters:
        df1: (first) dataframe which is inserted into the former one in calculation method (such as div, mul)
        df2: (second) dataframe which is inserted into the letter one in calculation method (such as div, mul)
        mode: calculation mode (div or mul)
    return:
        df1/df2 or df1*df2
    '''
    overlapping_columns = df1.columns.intersection(df2.columns)
    if mode == 'div':
        df_result = df1[overlapping_columns].div(df2[overlapping_columns])
    elif mode == 'mul':
        df_result = df1[overlapping_columns].mul(df2[overlapping_columns])
    else:
        print('no support type')
    return df_result

# financial variables divided by asset
df_liq_to_at = cal_two_df(df1=che_df, df2=at_df, mode='div') *100         # Cash and Short-Term Investments scaled by assets (liquidity)
df_booklev_to_at = cal_two_df(df1=lt_df, df2=at_df, mode='div') *100       # liabilities scaled by assets (book leverage)
df_oancf_to_at = cal_two_df(df1=oancf_df, df2=at_df, mode='div') *100     # Operating Activities Net Cash Flow scaled by assets
df_ad_to_at = cal_two_df(df1=xad_df, df2=at_df, mode='div') *100           # Ads expenditure scaled by assets
df_ad_to_sale = cal_two_df(df1=xad_df, df2=sale_df, mode='div') *100           # Ads expenditure scaled by assets

# summary statistics
at_df.stack().describe().round(2)
sale_df.stack().describe().round(2)
df_liq_to_at.stack().describe().round(4) # [0, inf)
df_booklev_to_at.stack().describe().round(4) # [0, inf)
df_oancf_to_at.stack().describe().round(4) # (-inf, inf)
df_oancf_to_at.stack().quantile(np.arange(0,1,0.1)).round(4)
df_ad_to_at.stack().describe().round(4) # [0,inf])
df_ad_to_at.stack().quantile(np.arange(0.9,1,0.01)).round(4)
df_ad_to_sale.stack().describe().round(4) # [0,inf])

'''
Exponential transformation:
for the variable x with the range of (-inf, inf), ESG/x is NOT a strictly increse function.
x_new = exp(x)
but extremely small number make computational error (imprecision)
so scale-down is demanded (it is not problematic because the ORDER is only consideration)
'''
# # zero check
# df_oancf_to_at.loc[:,df_oancf_to_at.columns[df_oancf_to_at[df_oancf_to_at==0.0].any()]]
# # scale down
# df_oancf_to_at = df_oancf_to_at/10
# # exponentialize 
# df_oancf_to_at = np.exp(df_oancf_to_at)

#############
# Min max scaling with log
#############
def apply_log(x):
    # seperate two part: posi and nega
    if x >= 0:
        return np.log(1 + x)
    else: # symmetric with x>=0
        return -np.log(1 - x)

def minmax_scale(df, log_scale=False):
    '''min max scale
    replace 0 with min value among not 0 (avoid error from dividing by zero)'''
    # minmax scale
    if log_scale == True: 
        df_copy = df.copy().applymap(apply_log).transpose()
    else: # default
        df_copy = df.copy().transpose()
    df_copy = (df_copy - df_copy.min()) / (df_copy.max() - df_copy.min())
    df_copy = df_copy.transpose() * 100

    # replace zero -> second min
    min_value = df_copy.min().min()
    second_min_value = df_copy[df_copy > min_value].min().min()
    df_copy[df_copy == min_value] = second_min_value

    return df_copy

# assets and sales are needed to be taken the logarithm
minmax_scale(at_df, log_scale=True).stack().describe().round(2)
minmax_scale(sale_df, log_scale=True).stack().describe().round(2)

# others are needed to check (shape)
minmax_scale(df_liq_to_at).stack().describe().round(4)
minmax_scale(df_liq_to_at, log_scale=True).stack().describe().round(4)
minmax_scale(df_oancf_to_at).stack().describe().round(4)
minmax_scale(df_oancf_to_at, log_scale=True).stack().describe().round(4)
minmax_scale(df_booklev_to_at).stack().describe().round(4)
minmax_scale(df_booklev_to_at, log_scale=True).stack().describe().round(4)
minmax_scale(df_ad_to_at).stack().describe().round(4)
minmax_scale(df_ad_to_at, log_scale=True).stack().describe().round(4)
minmax_scale(df_ad_to_sale).stack().describe().round(4)
minmax_scale(df_ad_to_sale, log_scale=True).stack().describe().round(4)

'''
ESG
'''
''' w/o scaling '''
# Calculate CONTROLLED ESG dataframe
# ESG/Size
df_ESG_to_at = cal_two_df(df1=df_ESG_match, df2=at_df, mode='div')          # ESG scaled by asset
df_ESG_to_sale = cal_two_df(df1=df_ESG_match, df2=sale_df, mode='div')      # ESG scaled by sales
# ESG/agencycost
# ESG risk = 1/(ESG score*liquidity) or 1/(ESG score/book leverage)
df_ESG_dot_liq = cal_two_df(df1=df_ESG_match, df2=df_liq_to_at, mode='mul')
df_ESG_dot_oancf = cal_two_df(df1=df_ESG_match, df2=df_oancf_to_at, mode='mul') # alt proxy of liq
df_ESG_to_booklev = cal_two_df(df1=df_ESG_match, df2=df_booklev_to_at, mode='div')
# anti: 1/(ESG score/liquidity) or 1/(ESG score*book leverage)
df_ESG_to_liq = cal_two_df(df1=df_ESG_match, df2=df_liq_to_at, mode='div')
df_ESG_to_oancf = cal_two_df(df1=df_ESG_match, df2=df_oancf_to_at, mode='div') # alt proxy of liq
df_ESG_dot_booklev = cal_two_df(df1=df_ESG_match, df2=df_booklev_to_at, mode='mul')
# ESG/ad
df_ESG_to_ad = cal_two_df(df1=df_ESG_match, df2=df_ad_to_at, mode='div')
df_ESG_to_ad_sale = cal_two_df(df1=df_ESG_match, df2=df_ad_to_sale, mode='div')
# anti ESG*ad
df_ESG_dot_ad = cal_two_df(df1=df_ESG_match, df2=df_ad_to_at, mode='mul')
df_ESG_dot_ad_sale = cal_two_df(df1=df_ESG_match, df2=df_ad_to_sale, mode='mul')


# describe
df_ESG_to_at.stack().describe().round(4) # [0, inf) # [19 rows x 4303 columns]
df_ESG_to_sale.stack().describe().round(4) # [0, inf) # [19 rows x 4303 columns]
# df_ESG_to_sale.stack().quantile(np.arange(0.9,1.005,0.005)).round(2)

df_ESG_dot_liq.stack().describe().round(4) # [0, inf) # [19 rows x 4303 columns]
df_ESG_dot_oancf.stack().describe().round(4) # (-inf, inf) # [19 rows x 4303 columns]
df_ESG_to_booklev.stack().describe().round(4) # [0, inf]) # [19 rows x 4303 columns]
df_ESG_to_liq.stack().describe().round(4) # [0, inf) # [19 rows x 4303 columns]
df_ESG_to_oancf.stack().describe().round(4) # (-inf, inf) # [19 rows x 4303 columns]
df_ESG_dot_booklev.stack().describe().round(4) # [0, inf]) # [19 rows x 4303 columns]

df_ESG_to_ad.stack().describe().round(4) # [0, inf]) # [19 rows x 2110 columns]
df_ESG_to_ad_sale.stack().describe().round(4) # [0, inf]) # [19 rows x 2110 columns]
df_ESG_dot_ad.stack().describe().round(4) # [0, inf]) # [19 rows x 2110 columns]
df_ESG_dot_ad_sale.stack().describe().round(4) # [0, inf]) # [19 rows x 2110 columns]


# save the dataframes
df_ESG_to_at.to_csv('data/df_ESG_to_at.csv')
df_ESG_to_sale.to_csv('data/df_ESG_to_sale.csv')

df_ESG_dot_liq.to_csv('data/df_ESG_dot_liq.csv')
df_ESG_dot_oancf.to_csv('data/df_ESG_dot_oancf.csv')
df_ESG_to_booklev.to_csv('data/df_ESG_to_booklev.csv')
df_ESG_to_liq.to_csv('data/df_ESG_to_liq.csv')
# df_ESG_to_oancf.to_csv('data/df_ESG_to_oancf.csv')
df_ESG_dot_booklev.to_csv('data/df_ESG_dot_booklev.csv')

df_ESG_to_ad.to_csv('data/df_ESG_to_ad.csv')
df_ESG_to_ad_sale.to_csv('data/df_ESG_to_ad_sale.csv')
df_ESG_dot_ad.to_csv('data/df_ESG_dot_ad.csv')
df_ESG_dot_ad_sale.to_csv('data/df_ESG_dot_ad_sale.csv')

'''w/ scaling'''
# ESG/Size
df_ESG_to_at_mmlog = cal_two_df(df1=df_ESG_match, df2=minmax_scale(at_df, log_scale=False), mode='div')          # ESG scaled by asset
df_ESG_to_sale_mmlog = cal_two_df(df1=df_ESG_match, df2=minmax_scale(sale_df, log_scale=False) , mode='div')      # ESG scaled by sales
# ESG/agencycost
# ESG risk = 1/(ESG score*liquidity) or 1/(ESG score/book leverage)
df_ESG_dot_liq_mmlog = cal_two_df(df1=df_ESG_match, df2=minmax_scale(df_liq_to_at, log_scale=False), mode='mul')
df_ESG_dot_oancf_mmlog = cal_two_df(df1=df_ESG_match, df2=minmax_scale(df_oancf_to_at, log_scale=False), mode='mul') # alt proxy of liq
df_ESG_to_booklev_mmlog = cal_two_df(df1=df_ESG_match, df2=minmax_scale(df_booklev_to_at, log_scale=False), mode='div')
# ESG/ad
df_ESG_to_ad_mmlog = cal_two_df(df1=df_ESG_match, df2=minmax_scale(df_ad_to_at, log_scale=False), mode='div')
df_ESG_to_ad_sale_mmlog = cal_two_df(df1=df_ESG_match, df2=minmax_scale(df_ad_to_sale, log_scale=False), mode='div')

# describe
df_ESG_to_at_mmlog.stack().describe().round(4) # [0, inf) # [19 rows x 4303 columns]
df_ESG_to_sale_mmlog.stack().describe().round(4) # [0, inf) # [19 rows x 4303 columns]
# df_ESG_to_sale_mmlog.stack().quantile(np.arange(0.9,1.005,0.005)).round(2)
df_ESG_dot_liq_mmlog.stack().describe().round(4) # [0, inf) # [19 rows x 4303 columns]
df_ESG_dot_oancf_mmlog.stack().describe().round(4) # (-inf, inf) # [19 rows x 4303 columns]
df_ESG_to_booklev_mmlog.stack().describe().round(4) # [0, inf]) # [19 rows x 4303 columns]
# df_ESG_to_booklev_mmlog.stack().quantile(np.arange(0.9,1.005,0.005)).round(2)
df_ESG_to_ad_mmlog.stack().describe().round(4) # [0, inf]) # [19 rows x 2110 columns]
df_ESG_to_ad_mmlog.stack().quantile(np.arange(0.9,1.005,0.005)).round(2)
df_ESG_to_ad_sale_mmlog.stack().describe().round(2) # [0, inf]) # [19 rows x 2110 columns]
df_ESG_to_ad_sale_mmlog.stack().quantile(np.arange(0.9,1.005,0.005)).round(2)

# save the dataframes
df_ESG_to_at_mmlog.to_csv('data/df_ESG_to_at_mmlog.csv')
df_ESG_to_sale_mmlog.to_csv('data/df_ESG_to_sale_mmlog.csv')
df_ESG_dot_liq_mmlog.to_csv('data/df_ESG_dot_liq_mmlog.csv')
df_ESG_dot_oancf_mmlog.to_csv('data/df_ESG_dot_oancf_mmlog.csv')
df_ESG_to_booklev_mmlog.to_csv('data/df_ESG_to_booklev_mmlog.csv')
df_ESG_to_ad_mmlog.to_csv('data/df_ESG_to_ad_mmlog.csv')
df_ESG_to_ad_sale_mmlog.to_csv('data/df_ESG_to_ad_sale_mmlog.csv')

'''
ESG, controlled by Lagged
'''
'''w/ scaling'''
lag = 1
# ESG/Size
df_ESG_to_l_at_mmlog = cal_two_df(df1=df_ESG_match, df2=minmax_scale(at_df.shift(lag), log_scale=True), mode='div')          # ESG scaled by asset
df_ESG_to_l_sale_mmlog = cal_two_df(df1=df_ESG_match, df2=minmax_scale(sale_df.shift(lag), log_scale=True) , mode='div')      # ESG scaled by sales
# ESG/agencycost
# ESG risk = 1/(ESG score*liquidity) or 1/(ESG score/book leverage)
df_ESG_dot_l_liq_mmlog = cal_two_df(df1=df_ESG_match, df2=minmax_scale(df_liq_to_at.shift(lag), log_scale=True), mode='mul')
df_ESG_dot_l_oancf_mmlog = cal_two_df(df1=df_ESG_match, df2=minmax_scale(df_oancf_to_at.shift(lag), log_scale=True), mode='mul') # alt proxy of liq
df_ESG_to_l_booklev_mmlog = cal_two_df(df1=df_ESG_match, df2=minmax_scale(df_booklev_to_at.shift(lag), log_scale=True), mode='div')
# ESG/ad
df_ESG_to_l_ad_mmlog = cal_two_df(df1=df_ESG_match, df2=minmax_scale(df_ad_to_at.shift(lag), log_scale=True), mode='div')

# describe
df_ESG_to_l_at_mmlog.stack().describe().round(4) # [0, inf) # [19 rows x 4303 columns]
df_ESG_to_l_sale_mmlog.stack().describe().round(4) # [0, inf) # [19 rows x 4303 columns]
# df_ESG_to_sale_mmlog.stack().quantile(np.arange(0.9,1.005,0.005)).round(2)
df_ESG_dot_l_liq_mmlog.stack().describe().round(4) # [0, inf) # [19 rows x 4303 columns]
df_ESG_dot_l_oancf_mmlog.stack().describe().round(4) # (-inf, inf) # [19 rows x 4303 columns]
df_ESG_to_l_booklev_mmlog.stack().describe().round(4) # [0, inf]) # [19 rows x 4303 columns]
# df_ESG_to_booklev_mmlog.stack().quantile(np.arange(0.9,1.005,0.005)).round(2)
df_ESG_to_l_ad_mmlog.stack().describe().round(4) # [0, inf]) # [19 rows x 2110 columns]
# df_ESG_to_l_ad_mmlog.stack().quantile(np.arange(0.9,1.005,0.005)).round(2)

# save the dataframes
df_ESG_to_l_at_mmlog.to_csv('data/df_ESG_to_l_at_mmlog.csv')
df_ESG_to_l_sale_mmlog.to_csv('data/df_ESG_to_l_sale_mmlog.csv')
df_ESG_dot_l_liq_mmlog.to_csv('data/df_ESG_dot_l_liq_mmlog.csv')
df_ESG_dot_l_oancf_mmlog.to_csv('data/df_ESG_dot_l_oancf_mmlog.csv')
df_ESG_to_l_booklev_mmlog.to_csv('data/df_ESG_to_l_booklev_mmlog.csv')
df_ESG_to_l_ad_mmlog.to_csv('data/df_ESG_to_l_ad_mmlog.csv')


'''
ESG Controversy
'''
# ESG/Size
df_ESGC_to_at = cal_two_df(df1=df_ESGC_match, df2=at_df, mode='div')          # ESGC scaled by asset
df_ESGC_to_sale = cal_two_df(df1=df_ESGC_match, df2=sale_df, mode='div')      # ESGC scaled by sales
# ESG/agencycost
# ESG risk = 1/(ESG score*liquidity) or 1/(ESG score/book leverage)
df_ESGC_dot_liq = cal_two_df(df1=df_ESGC_match, df2=df_liq_to_at, mode='mul')
df_ESGC_dot_oancf = cal_two_df(df1=df_ESGC_match, df2=df_oancf_to_at, mode='mul') # alt proxy of liq
df_ESGC_to_booklev = cal_two_df(df1=df_ESGC_match, df2=df_booklev_to_at, mode='div')
# ESG/ad
df_ESGC_to_ad = cal_two_df(df1=df_ESGC_match, df2=df_ad_to_at, mode='div')

# save the dataframes
df_ESGC_to_at.to_csv('data/df_ESGC_to_at.csv')
df_ESGC_to_sale.to_csv('data/df_ESGC_to_sale.csv')
df_ESGC_dot_liq.to_csv('data/df_ESGC_dot_liq.csv')
df_ESGC_dot_oancf.to_csv('data/df_ESGC_dot_oancf.csv')
df_ESGC_to_booklev.to_csv('data/df_ESGC_to_booklev.csv')
df_ESGC_to_ad.to_csv('data/df_ESGC_to_ad.csv')

'''
Greenwashing or ESG betrayal
'''
df_GW = df_ESGC_match * df_ESG_match.shift(1)
df_GW.stack().describe().round(2)
df_GW.stack().quantile(np.arange(0,1.1,0.1)).round(2)
df_GW.apply(pd.DataFrame.describe, axis=1).round(2) # by year
df_GW.transpose().quantile(np.arange(0,1.1,0.1)).round(2) # by year
df_GW.to_csv('data/df_GW.csv')

# ########
# # check 
# '''df_ESG: firmname with monthly
# df_ESG_match: cusip with yearly'''
# # Randomly select 10 columns from df_ESG
# df_ESG_temp = df_ESG.sample(n=3, axis=1) 
# # Preprocess index (to yearly)
# df_ESG_temp.index = pd.to_datetime(df_ESG_temp.index)
# df_ESG_temp = df_ESG_temp.resample('Y').last()
# # match name-cusip
# df_ESG_temp.columns
# # df_ESG_match.loc[:,['12571T100','92342Y109','05070R104']] # manually collect from SEC.gov
