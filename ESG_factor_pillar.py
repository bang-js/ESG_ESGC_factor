import pandas as pd
import numpy as np

###########################
# 1. Refinitiv ESG pillar data
###########################
df_NASDAQ_act_pillar = pd.read_excel('data/NASDAQ_act_pillar.xlsx', sheet_name='data').rename(columns={'Unnamed: 0':'Name'}).set_index(keys='Name') # [22 rows x 11760 columns]
df_NASDAQ_dead_pillar = pd.read_excel('data/NASDAQ_dead_pillar.xlsx', sheet_name='data').rename(columns={'Unnamed: 0':'Name'}).set_index(keys='Name') # [22 rows x 13140 columns]
df_NYSE_mkt_pillar = pd.read_excel('data/NYSE_mkt_pillar.xlsx', sheet_name='data').rename(columns={'Unnamed: 0':'Name'}).set_index(keys='Name') # [22 rows x 3651 columns]
df_NYSE_act_pillar = pd.read_excel('data/NYSE_act_pillar.xlsx', sheet_name='data').rename(columns={'Unnamed: 0':'Name'}).set_index(keys='Name') # [22 rows x 6573 columns]
df_NYSE_dead_minor_pillar = pd.read_excel('data/NYSE_dead_minor_pillar.xlsx', sheet_name='data').rename(columns={'Unnamed: 0':'Name'}).set_index(keys='Name') # [22 rows x 14943 columns]

# merge NYSE_dead_major_pillar
df_NYSE_dead_major_pillar = pd.read_excel('data/NYSE_dead_major_pillar.xlsx', sheet_name=None)
df_NYSE_dead_major_pillar_1 = df_NYSE_dead_major_pillar['-2000'].rename(columns={'Unnamed: 0':'Name'}).set_index(keys='Name')
df_NYSE_dead_major_pillar_2 = df_NYSE_dead_major_pillar['2001-6000'].rename(columns={'Unnamed: 0':'Name'}).set_index(keys='Name')
df_NYSE_dead_major_pillar_3 = df_NYSE_dead_major_pillar['6001-'].rename(columns={'Unnamed: 0':'Name'}).set_index(keys='Name')
df_NYSE_dead_major_pillar = pd.concat([df_NYSE_dead_major_pillar_1, df_NYSE_dead_major_pillar_2, df_NYSE_dead_major_pillar_3],axis=1) # [22 rows x 26472 columns]

# concatenate all df
df_Ref_pillar_tot = pd.concat([df_NASDAQ_act_pillar, df_NASDAQ_dead_pillar, df_NYSE_mkt_pillar, df_NYSE_act_pillar, df_NYSE_dead_major_pillar, df_NYSE_dead_minor_pillar], axis=1)
# [22 rows x 76539 columns]
# df_Ref_pillar_tot.to_csv('data/df_Ref_pillar_tot.csv') # saving total dataframe
print('check the length of column (devided by 3 = E S G):', df_Ref_pillar_tot.shape[1] %3 == 0 )

'Read the df_Ref_pillar_tot'
# df_Ref_pillar_tot = pd.read_csv('data/df_Ref_pillar_tot.csv', index_col=0) 

# Seperate to three dataframes
df_Env = df_Ref_pillar_tot.loc[:,df_Ref_pillar_tot.columns.str.contains(' - Environment Pillar Score')]         # [22 rows x 5676 columns]
df_Soc = df_Ref_pillar_tot.loc[:,df_Ref_pillar_tot.columns.str.contains(' - Social Pillar Score')]              # [22 rows x 5676 columns]
df_Gov = df_Ref_pillar_tot.loc[:,df_Ref_pillar_tot.columns.str.contains(' - Governance Pillar Score')]          # [22 rows x 5676 columns]

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
# eliminate index in columns
df_Env = elimword_colname(df_Env, ' - Environment Pillar Score')
df_Soc = elimword_colname(df_Soc, ' - Social Pillar Score')
df_Gov = elimword_colname(df_Gov, ' - Governance Pillar Score')

# convert the columns of the dataframe from name to code
df_Env.columns = df_Env.loc['Code'][:].replace(to_replace=r'\(ENSCORE\)',value='',regex=True)
df_Soc.columns = df_Soc.loc['Code'][:].replace(to_replace=r'\(SOSCORE\)',value='',regex=True)
df_Gov.columns = df_Gov.loc['Code'][:].replace(to_replace=r'\(CGSCORE\)',value='',regex=True)

# drop row of 'Code'
df_Env.drop(df_Env.index[0], inplace=True)
df_Soc.drop(df_Soc.index[0], inplace=True)
df_Gov.drop(df_Gov.index[0], inplace=True)

# Convert object to numbers
df_Env = df_Env.apply(pd.to_numeric, errors='coerce')
df_Soc = df_Soc.apply(pd.to_numeric, errors='coerce')
df_Gov = df_Gov.apply(pd.to_numeric, errors='coerce')

'''중복되는 것을 지우면 둘 중 어느 것을 선택해야 하는지에 대한 문제가 발생'''
'''아래는 그냥 늦게 나오는 컬럼을 삭제하는 code'''
# drop same firms
df_Env_match = df_Env.T.drop_duplicates(keep='first').T # [21 rows x 3296 columns]
df_Soc_match = df_Soc.T.drop_duplicates(keep='first').T # [21 rows x 4295 columns]
df_Gov_match = df_Gov.T.drop_duplicates(keep='first').T # [21 rows x 4294 columns]

# cut after 2020
df_Env_match = df_Env_match[df_Env_match.index < 2021]
df_Soc_match = df_Soc_match[df_Soc_match.index < 2021]
df_Gov_match = df_Gov_match[df_Gov_match.index < 2021]

# zero -> nan
'''Env의 경우 0이 너무 많아서 0을 nan으로 처리하여 분석진행'''
df_Env_match_zero = df_Env_match.replace(0.0, np.nan) 
df_Env_match_zero = df_Env_match_zero.dropna(axis=1, how='all') # [19 rows x 2811 columns]

# # Save the dataframes
# df_Env_match.to_csv('data/df_Env_match.csv')
# df_Env_match_zero.to_csv('data/df_Env_match_zero.csv')
# df_Soc_match.to_csv('data/df_Soc_match.csv')
# df_Gov_match.to_csv('data/df_Gov_match.csv')

###########################
# 2. Refinitiv P MV ESG ESGC data
###########################

'''
Recall df_Ref_tot
See ESG_fin_Data.py
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
df_ESG.index = pd.to_datetime(df_ESG.index)
df_ESG = conv_m2y(df_ESG)

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

# dict: name - code

## ver 1. using only df_code
# eliminate (P) in 'Code' -> convert to dict
dict_name_code = df_code.loc['Code'][:].replace(to_replace=r'\(P\)',value='',regex=True).to_dict()
len(dict_name_code) # 30648
list(dict_name_code.items())[:10]

# # Match name - code for four dataframes
# df_P = df_P.rename(columns=dict_name_code) # [253 rows x 21926 columns]
# df_MV = df_MV.rename(columns=dict_name_code) # [253 rows x 20160 columns]

# # Convert index format from object to datetime
# df_P.index = pd.to_datetime(df_P.index)
# df_MV.index = pd.to_datetime(df_MV.index)

# # save
# df_P.to_csv('df_P_Rftcode.csv')
# df_MV.to_csv('df_MV_Rftcode.csv')


'''왜 차이가 심한지?? missing이 의미하는게 뭔지
-> 애초에 retrieve를 못한것
-> df_info를 사용하여 메꾸기'''
## ver 2. from df_info
df_info_temp = df_info[['Name','Symbol']]
df_info_temp.set_index('Name', inplace=True)
dict_name_code_2 = df_info_temp['Symbol'].to_dict() # only df_info
len(dict_name_code_2) # 36532
list(dict_name_code_2.items())[:10]

# combine df_info and df_code
dict_name_code_2.update(dict_name_code)
len(dict_name_code_2) # 53274
list(dict_name_code_2.items())[:10]

# match using extended dict
df_P = df_P.rename(columns=dict_name_code_2)
df_MV = df_MV.rename(columns=dict_name_code_2)
df_ESG = df_ESG.rename(columns=dict_name_code_2)

# drop same firms
df_ESG = df_ESG.T.drop_duplicates(keep='first').T # [19 rows x 4780 columns]

# check the missing
# len([_ for _ in df_P.columns.astype('str') if len(_)>7])
len([_ for _ in df_P.columns.astype('str') if len(_)>7]) # 717

######################################################################
# 3. Factor construction
######################################################################

# pre-processing P and MV df
df_P.index = pd.to_datetime(df_P.index)
df_P.index = df_P.asfreq(freq='M').index # set frequency to 'M' for factor construnction  

df_MV.index = pd.to_datetime(df_MV.index)
df_MV.index = df_MV.asfreq(freq='M').index 

# pre-processing ESG df 
df_ESG = df_ESG[df_ESG.index < 2021] # period cut

# convert data frequency from month to year for df_MV
df_MV_year = df_MV.resample(rule='1Y').last() # monthly data -> yearly data, data: Dec 
df_MV_year.index = df_MV_year.index.year # re-indexing
df_MV_year.index.name = 'year'
df_MV_year = df_MV_year.loc[df_Env_match.index,:] # match YEARs with df_ESG_...


######
# ESG match with ENV and SOC
######
df_ESG_match_SOC = df_ESG[df_ESG.columns.intersection(df_Soc_match.columns)] # [19 rows x 4037 columns]

######
# Summary stat
######

print('duplicated columns b/t df_P and df_Env:', len(set(df_P.columns) & set(df_Env_match.columns))) # 3153
print('duplicated columns b/t df_P and df_Soc:', len(set(df_P.columns) & set(df_Soc_match.columns))) # 4112
print('duplicated columns b/t df_P and df_Gov:', len(set(df_P.columns) & set(df_Gov_match.columns))) # 4112

print('duplicated columns b/t df_P and df_Env:', len(set(df_MV.columns) & set(df_Env_match.columns))) # 3137
print('duplicated columns b/t df_P and df_Soc:', len(set(df_MV.columns) & set(df_Soc_match.columns))) # 4095
print('duplicated columns b/t df_P and df_Gov:', len(set(df_MV.columns) & set(df_Gov_match.columns))) # 4095

print('duplicated columns b/t df_P and df_ESG:', len(set(df_P.columns) & set(df_ESG.columns))) # 4863
print('duplicated columns b/t df_ESG and df_SOC:', len(set(df_ESG.columns) & set(df_Soc_match.columns))) # 4037

def summary_over(df_temp):
    ov_col = df_P.columns.intersection(df_MV.columns).intersection(df_temp.columns)
    print(df_temp[ov_col].shape)
    print(df_temp[ov_col].apply(pd.DataFrame.describe, axis=1).round(2))
    print(df_temp[ov_col].T.quantile(np.arange(0,1,0.1)).round(2).T)
    print(df_temp[ov_col].stack().describe().round(2))

summary_over(df_ESG) 
summary_over(df_ESG_match_SOC) 
summary_over(df_Env_match)
summary_over(df_Env_match_zero)
summary_over(df_Soc_match)
summary_over(df_Gov_match)

#########
# ONE-WAY sort portfolio 
#########
def no_stock(x,y):
    '''
    return the number of non-NaN elements
    '''
    return min(x[~np.isnan(x)].shape[0], y[~np.isnan(y)].shape[0])
# |       | Low  | ... | High |
# |-------|------|-----|------|
# | pf no.| 1    |     |q(>1) |
#           --- INCREASE --->
def factor_cal(quantile, weight_mode, df_temp, df_temp_p=df_P, df_temp_mv=df_MV, start_criterion=2002, end_criterion=2020):
    '''
    The purpose of this function is to calculat the sorted portfolio returns using pandas qcut method.

    Parameters
        quantile: quantile no.
        weight mode \in {value, equal, log-value}
        df_temp: df_ESG, df_ESGC, ... (YEARLY)
        df_temp_p: price dateframe (MONTHLY)
        df_temp_mv: market value dataframe (MONTHLY)
        start_criterion, end_criterion: the start year and end year of the criterion (e.g. ESG)

    Return
        quantile_return_df: one-way portfolio returns (monthly) by the given quantile
    '''
    # drop firm without ESG or price or mv data
    ov_col = df_temp_p.columns.intersection(df_temp_mv.columns).intersection(df_temp.columns)
    df_temp = df_temp[ov_col]
    df_temp_p = df_temp_p[ov_col]
    df_temp_mv = df_temp_mv[ov_col]

    # the start year of factor construction (the next year of the sort variable's year)
    start = int(start_criterion + 1)  
    end = int(end_criterion + 1) 

    # generate stock return dataframe
    df_temp_r = df_temp_p/df_temp_p.shift(1)

    # array for storing return values
    quantile_return = np.empty((int((end_criterion-start_criterion+1)*12),quantile))

    # slice target data for the given period
    df_temp = df_temp[(start_criterion <= df_temp.index) & (df_temp.index <= end_criterion)] # from Jul 2003 to Jun 2022

    # calcualte factor returns
    for y, year in enumerate(df_temp.index):
        # row of year y (series) -> dataframe (shape=(n,1)) 
        df_per_year_cr = df_temp.iloc[y][:].to_frame('first') 
        '''note. due to to_frame() the df is transposed '''
        
        # eliminate firms that have NaN data
        df_per_year_cr.dropna(inplace=True)

        # generate quantile no of each firm 
        df_per_year_cr['quantile'] = pd.qcut(df_per_year_cr['first'], quantile, labels=False)

        # partition arrays for each quantile
        partition = []
        for _q in range(quantile):
            # Find the indices of rows with the target value
            temp_q = df_per_year_cr.index[df_per_year_cr['quantile'] == _q].to_numpy()
            partition.append(temp_q)

        # set time range : from July of year t+1 to June of year t+2
        t_rg = pd.date_range(str(year+1)+'-06', freq='M', periods=13)  # get July of year t+2 to calculate r_t+1

        ### calculate factor returns ###
        # weight (mv): one month ahead / return: the given month
        ## value weight
        if weight_mode == 'value':
            for q, firms_qt in enumerate(partition):
                # print('quantil no',q+1)
                for _, t in enumerate(t_rg[:-1]):
                    r_t1 = np.array(df_temp_r.loc[t_rg[_+1],firms_qt].values) # r_t+1 vector             
                    w_t = np.array(df_temp_mv.loc[t,firms_qt].values/df_temp_mv.loc[t,firms_qt].sum(skipna=True)) # w_t vector for value-weight portfolio
                    # print(t_rg[_+1], np.nansum(r_t1*w_t), no_stock(r_t1, w_t)) # date - factor return - no. of stock            
                    quantile_return[y*12+_,q] = np.nansum(r_t1*w_t) # store

        ## equal weight
        elif weight_mode == 'equal':
            for q, firms_qt in enumerate(partition):
                # print('quantil no',q+1)
                for _, t in enumerate(t_rg[:-1]):
                    r_t1 = np.array(df_temp_r.loc[t_rg[_+1],firms_qt].values) # r_t+1 vector
                    # print(t_rg[_+1], np.nanmean(r_t1), no_stock(r_t1, r_t1))            
                    quantile_return[y*12+_,q] = np.nanmean(r_t1) # store
    # print(quantile_return) 
    
    # generate df of factor returns (monthly)
    period = pd.date_range('{}-07'.format(start), '{}-07'.format(end+1), freq='M')  
    col = (np.arange(quantile) + 1).astype('str')
    quantile_return_df = pd.DataFrame(quantile_return-1, columns=col, index=period) # result format: 0.xx (not 0.9xx - 1.xxx)
    
    return quantile_return_df

#########
# TWO-WAY sort portfolio
#########
### NxM portfolio returns ###
# |         | ESG_1 | ... | ESG_q1  |
# |---------|-------|-----|---------|
# | Size_1  | 0     | ... | q1-1    |
# | ...     |       |     |         |
# | Size_q2 | q2-1  | ... | q1*q2-1 |

def multi_factor_cal(quantile_1, quantile_2, weight_mode, df_temp_1, df_temp_2, df_temp_p=df_P, df_temp_mv=df_MV, start_criterion=2002, end_criterion=2020):
    '''
    Parameters
        quantile_1(2): quantile no. of the 1st(2nd) dataframe 
        Notice! set 1: target variable / 2: size ...
        weight mode \in {value, equal, log-value}
        df_temp: df_ESG, df_ESGC, ... (YEARLY)
        df_temp_p: price dateframe (MONTHLY)
        df_temp_mv: market value dataframe (MONTHLY)
        start_criterion, end_criterion: the start year and end year of the criterion (e.g. ESG)

    Return
        quantile_return_df: two-way portfolio returns (monthly) by the given quantile
    '''
    # drop firm without ESG or price or mv data
    ov_col = df_temp_p.columns.intersection(df_temp_mv.columns).intersection(df_temp_1.columns).intersection(df_temp_2.columns)
    df_temp_1 = df_temp_1[ov_col]
    df_temp_2 = df_temp_2[ov_col]
    df_temp_p = df_temp_p[ov_col]
    df_temp_mv = df_temp_mv[ov_col]

    # the start year of factor construction (the next year of the sort variable's year)
    start = int(start_criterion + 1)  
    end = int(end_criterion + 1) 

    # generate return dataframe
    df_temp_r = df_temp_p/df_temp_p.shift(1)

    # array for storing return values (no. of columns is q1*q2 to simultaneously store two-way sort results)
    quantile_return = np.empty((int((end_criterion-start_criterion+1)*12), quantile_1*quantile_2))
    
    # slice target data for the given period
    df_temp_1 = df_temp_1[(start_criterion <= df_temp_1.index ) & (df_temp_1.index <= end_criterion)] # from Jul 2003 to Jun 2022
    df_temp_2 = df_temp_2[(start_criterion <= df_temp_2.index ) & (df_temp_2.index <= end_criterion)]
    
    # calcualte factor returns
    for y, year in enumerate(df_temp_1.index):
        # first: TARGET, second: SIZE
        df_per_year_cr = df_temp_1.iloc[y][:].to_frame('first')
        df_per_year_cr['second'] = df_temp_2.iloc[y][:]

        # Delete companies without target var. and mv because they are unnecessary (deleted by the first df)
        df_per_year_cr.dropna(inplace=True)

        # Calculate the quantile values
        quantiles_first = pd.qcut(df_per_year_cr['first'], quantile_1, labels=False)
        quantiles_second = pd.qcut(df_per_year_cr['second'], quantile_2, labels=False)
        # Add a new column for the quantile categories
        df_per_year_cr['first_q'] = quantiles_first
        df_per_year_cr['second_q'] = quantiles_second
        # Add a new column considering two quantile
        df_per_year_cr['quantile'] = quantiles_first + quantile_1 * quantiles_second

        # generate partition array
        partition = []
        for _q in range(quantile_1*quantile_2):
            # Find the indices of rows with the target value
            temp_q = df_per_year_cr.index[df_per_year_cr['quantile'] == _q].to_numpy()
            partition.append(temp_q)
        
        # time range: from July of year t+1 to June of year t+2
        t_rg = pd.date_range(str(year+1)+'-06', freq='M', periods=13) # get July of year t+2 to calculate r_t+1

        ### calculate factor returns ###
        # weight (mv): one month ahead / return: the given month
        ## value weight
        if weight_mode == 'value':
            for q, firms_qt in enumerate(partition):
                # print('quantil no',q+1)
                for _, t in enumerate(t_rg[:-1]):
                    r_t1 = np.array(df_temp_r.loc[t_rg[_+1],firms_qt].values) # r_t+1 vector             
                    w_t = np.array(df_temp_mv.loc[t,firms_qt].values/df_temp_mv.loc[t,firms_qt].sum(skipna=True)) # w_t vector for value-weight portfolio
                    # print(t_rg[_+1], np.nansum(r_t1*w_t), no_stock(r_t1, w_t))            
                    # store
                    quantile_return[y*12+_,q] = np.nansum(r_t1*w_t)

        ## equal weight
        elif weight_mode == 'equal':
            for q, firms_qt in enumerate(partition):
                # print('quantil no',q+1)
                for _, t in enumerate(t_rg[:-1]):
                    r_t1 = np.array(df_temp_r.loc[t_rg[_+1],firms_qt].values) # r_t+1 vector
                    # print(t_rg[_+1], np.nanmean(r_t1), no_stock(r_t1, r_t1))            
                    # store
                    quantile_return[y*12+_,q] = np.nanmean(r_t1)
    
    # get factor return
    period = pd.date_range('{}-07'.format(start), '{}-07'.format(end+1), freq='M')  
    col = [ '{}'.format(i)  for i in range(quantile_1*quantile_2)]
    quantile_return_df = pd.DataFrame(quantile_return-1, columns=col, index=period)
    
    return quantile_return_df

###########
# Saving ONE-WAY portfolio return data
###########

def saving_oneway_pf_vw(quantile, df_temp, start_criterion=2002, end_criterion=2020):
    df_pf_value = factor_cal(quantile=quantile, 
    weight_mode='value', 
    df_temp=df_temp, 
    start_criterion=start_criterion, 
    end_criterion=end_criterion)
    
    # CAUTION: trim after 2022-01-31 to match JKP dataset
    period_cut = pd.date_range('{}-07'.format(start_criterion+1), '{}-01'.format(end_criterion+2), freq='M')  
    df_pf_value = df_pf_value.loc[period_cut,:]
    
    print(df_pf_value.describe())
    return df_pf_value


# Set quantile
quantile_set = 10

esg_value = saving_oneway_pf_vw(quantile=quantile_set, df_temp=df_ESG)
esg_value.to_csv(f'result/esg_value_q{quantile_set}.csv') # nxq 
(esg_value.iloc[:,-1] - esg_value.iloc[:,0]).to_csv(f'result/esg_value_factor_q{quantile_set}.csv')  # nx1 LS pf

esg_value_2 = saving_oneway_pf_vw(quantile=quantile_set, df_temp=df_ESG_match_SOC)
esg_value_2.to_csv(f'result/esg_value_2_q{quantile_set}.csv') # nxq 
(esg_value_2.iloc[:,-1] - esg_value_2.iloc[:,0]).to_csv(f'result/esg_value_2_factor_q{quantile_set}.csv')  # nx1 LS pf

env_value = saving_oneway_pf_vw(quantile=quantile_set, df_temp=df_Env_match_zero)
env_value.to_csv(f'result/env_value_q{quantile_set}.csv')
(env_value.iloc[:,-1] - env_value.iloc[:,0]).to_csv(f'result/env_value_factor_q{quantile_set}.csv')  # nx1 LS pf

soc_value = saving_oneway_pf_vw(quantile=quantile_set, df_temp=df_Soc_match)
soc_value.to_csv(f'result/soc_value_q{quantile_set}.csv')
(soc_value.iloc[:,-1] - soc_value.iloc[:,0]).to_csv(f'result/soc_value_factor_q{quantile_set}.csv')  # nx1 LS pf

gov_value = saving_oneway_pf_vw(quantile=quantile_set, df_temp=df_Gov_match)
gov_value.to_csv(f'result/gov_value_q{quantile_set}.csv')
(gov_value.iloc[:,-1] - gov_value.iloc[:,0]).to_csv(f'result/gov_value_factor_q{quantile_set}.csv')  # nx1 LS pf
 

##############
# Saving TWO-WAY portfolio return data
##############
### calculate N by M portfolio returns ###
def saving_twoway_pf_vw(quantile_1, quantile_2, df_temp_1, df_temp_2=df_MV_year, start_criterion=2002, end_criterion=2020):
    df_pf_value = multi_factor_cal(quantile_1=quantile_1, 
    quantile_2=quantile_2, 
    weight_mode='value', 
    df_temp_1=df_temp_1, 
    df_temp_2=df_temp_2, 
    start_criterion=start_criterion, 
    end_criterion=end_criterion)

    # CAUTION: trim after 2022-01-31 to match JKP dataset
    period_cut = pd.date_range('{}-07'.format(start_criterion+1), '{}-01'.format(end_criterion+2), freq='M')  
    df_pf_value = df_pf_value.loc[period_cut,:]
    
    print(df_pf_value.describe())
    return df_pf_value

# Environment: Set quantile
quantile_1_set = 3
quantile_2_set = 2

env_value_two_size = saving_twoway_pf_vw(quantile_1=quantile_1_set, quantile_2=quantile_2_set, df_temp_1=df_Env_match_zero)
soc_value_two_size = saving_twoway_pf_vw(quantile_1=quantile_1_set, quantile_2=quantile_2_set, df_temp_1=df_Soc_match)
gov_value_two_size = saving_twoway_pf_vw(quantile_1=quantile_1_set, quantile_2=quantile_2_set, df_temp_1=df_Gov_match)

env_value_two_size.to_csv(f'result/env_value_two_size_q{quantile_1_set}_{quantile_2_set}.csv')

### Calculate factor returns ###
def saving_twoway_factor_vw(df_temp_1, quantile_1, quantile_2, df_temp_2=df_MV_year, start_criterion=2002, end_criterion=2020):
    '''
    This function is to calculate factor returns from two-way sorting by the target variable and SIZE 
    For examle, using 3x2 portfolio consisting of size (small and big) and controlled ESG (high middle low)
    factor = 1/2*(small high + big high) - 1/2*(small low + big low)
    |         | ESG_L | ESG_M | ESG_H |
    |---------|-------|-------|-------|
    | Small   | 0     | 1     | 2     |
    | Big     | 3     | 4     | 5     |
    '''
    df_pf_value = multi_factor_cal(quantile_1=quantile_1, 
    quantile_2=quantile_2, 
    weight_mode='value', 
    df_temp_1=df_temp_1, 
    df_temp_2=df_temp_2, 
    start_criterion=start_criterion, 
    end_criterion=end_criterion)

    # CAUTION: trim after 2022-01-31 to match JKP dataset
    period_cut = pd.date_range('{}-07'.format(start_criterion+1), '{}-01'.format(end_criterion+2), freq='M')  
    df_pf_value = df_pf_value.loc[period_cut,:]
    
    # calculate factor
    short_pf = list(range(0, quantile_2*quantile_1, quantile_1))
    long_pf = list(range(quantile_1-1, quantile_2*quantile_1, quantile_1))
    factor_rtn = (df_pf_value.iloc[:,long_pf]).mean(axis=1) - (df_pf_value.iloc[:,short_pf]).mean(axis=1)
    
    return factor_rtn

# Environment: Set quantile
saving_twoway_factor_vw_q1 = 3
saving_twoway_factor_vw_q2 = 2

esg_value_two_size_factor = saving_twoway_factor_vw(df_temp_1=df_ESG, 
                                                        quantile_1=saving_twoway_factor_vw_q1, 
                                                        quantile_2=saving_twoway_factor_vw_q2)
esg_value_two_size_factor.to_csv(f'result/esg_value_two_size_factor_q{saving_twoway_factor_vw_q1}_{saving_twoway_factor_vw_q2}.csv')
esg_value_2_two_size_factor = saving_twoway_factor_vw(df_temp_1=df_ESG_match_SOC, 
                                                        quantile_1=saving_twoway_factor_vw_q1, 
                                                        quantile_2=saving_twoway_factor_vw_q2)
esg_value_2_two_size_factor.to_csv(f'result/esg_value_2_two_size_factor_q{saving_twoway_factor_vw_q1}_{saving_twoway_factor_vw_q2}.csv')
env_value_two_size_factor = saving_twoway_factor_vw(df_temp_1=df_Env_match_zero, 
                                                        quantile_1=saving_twoway_factor_vw_q1, 
                                                        quantile_2=saving_twoway_factor_vw_q2)
env_value_two_size_factor.to_csv(f'result/env_value_two_size_factor_q{saving_twoway_factor_vw_q1}_{saving_twoway_factor_vw_q2}.csv')
soc_value_two_size_factor = saving_twoway_factor_vw(df_temp_1=df_Soc_match, 
                                                        quantile_1=saving_twoway_factor_vw_q1, 
                                                        quantile_2=saving_twoway_factor_vw_q2)
soc_value_two_size_factor.to_csv(f'result/soc_value_two_size_factor_q{saving_twoway_factor_vw_q1}_{saving_twoway_factor_vw_q2}.csv')
gov_value_two_size_factor = saving_twoway_factor_vw(df_temp_1=df_Gov_match, 
                                                        quantile_1=saving_twoway_factor_vw_q1, 
                                                        quantile_2=saving_twoway_factor_vw_q2)
gov_value_two_size_factor.to_csv(f'result/gov_value_two_size_factor_q{saving_twoway_factor_vw_q1}_{saving_twoway_factor_vw_q2}.csv')