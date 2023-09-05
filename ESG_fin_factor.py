import pandas as pd
import numpy as np

#############################
#############################
# 2. Factor construction
#############################
#############################

##########################################################
# 2.1. Retrieving saved data: p, mv, controlled ESG
##########################################################
# controlled ESG data
df_ESG_to_at = pd.read_csv('data/df_ESG_to_at.csv', index_col=0)
df_ESG_to_sale = pd.read_csv('data/df_ESG_to_sale.csv', index_col=0)
df_ESG_dot_liq = pd.read_csv('data/df_ESG_dot_liq.csv', index_col=0)
df_ESG_dot_oancf = pd.read_csv('data/df_ESG_dot_oancf.csv', index_col=0)
df_ESG_to_booklev = pd.read_csv('data/df_ESG_to_booklev.csv', index_col=0)
df_ESG_to_ad = pd.read_csv('data/df_ESG_to_ad.csv', index_col=0)

# pre-processing P and MV df
df_P = pd.read_csv('data/df_P.csv', index_col=0) # [253 rows x 8246 columns]
df_P.index = pd.to_datetime(df_P.index)
df_P.index = df_P.asfreq(freq='M').index # set frequency to 'M' for factor construnction  

df_MV = pd.read_csv('data/df_MV.csv', index_col=0) # [253 rows x 8242 columns]
df_MV.index = pd.to_datetime(df_MV.index)
df_MV.index = df_MV.asfreq(freq='M').index 

# df_ESG_to_at.shape
# # drop all NaN columns
# df_ESG_to_at.dropna(axis=1, how='all').shape
# # drop exactly same columns
# df_ESG_to_at.T.drop_duplicates().T

# convert data frequency from month to year for df_MV
df_MV_year = df_MV.resample(rule='1Y').last() # monthly data -> yearly data, data: Dec 
df_MV_year.index = df_MV_year.index.year # re-indexing
df_MV_year.index.name = 'year'
df_MV_year = df_MV_year.loc[df_ESG_to_at.index,:] # match YEARs with df_ESG_...

##########################################################
# 2.2. Functions of calculation portfolio returns
##########################################################

def no_stock(x,y):
    '''
    return the number of non-NaN elements
    '''
    return min(x[~np.isnan(x)].shape[0], y[~np.isnan(y)].shape[0])

#########
# ONE-WAY sort portfolio
#########

# |       | High | ... | Low |
# |-------|------|-----|-----|
# | pf no.| 1    |     |q(>1)|

def factor_cal(quantile, weight_mode, df_temp, df_temp_p=df_P, df_temp_mv=df_MV, start_criterion=2002, end_criterion=2020):
    '''
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
        # sort by ESG score (descending)
        series_sort = df_temp.iloc[y][:].sort_values(ascending=False)
        # saving non-NaN firm array by boolean indexing (not NaN)
        notna_index = series_sort.notna()         
        series_sort = series_sort[notna_index]

        # storing firms in each quantile into partitions
        n = series_sort.shape[0]
        partition = []
        adj_tie = [0] # adjusting same scored firms
        for _ in range(quantile-1):
            __ = 1
            while series_sort[int((_+1)/quantile*n)] == series_sort[int((_+1)/quantile*n)+__] :
                __ += 1
            adj_tie.append(int((_+1)/quantile*n+__))
        adj_tie.append(n)
        # print(adj_tie)
        for _ in range(len(adj_tie)-1):
            partition.append(series_sort.index[adj_tie[_]:adj_tie[_+1]])

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

### ver2 ###
# def factor_cal_v2(quantile, weight_mode, df_temp, df_temp_p=df_P, df_temp_mv=df_MV, start_criterion=2002, end_criterion=2020):
#     '''
#     The purpose of this function is to replicate the above function using other logic.
#     It uses the pandas qcut method.

#     Parameters
#         quantile: quantile no.
#         weight mode \in {value, equal, log-value}
#         df_temp: df_ESG, df_ESGC, ... (YEARLY)
#         df_temp_p: price dateframe (MONTHLY)
#         df_temp_mv: market value dataframe (MONTHLY)
#         start_criterion, end_criterion: the start year and end year of the criterion (e.g. ESG)

#     Return
#         quantile_return_df: one-way portfolio returns (monthly) by the given quantile
#     '''
#     # drop firm without ESG or price or mv data
#     ov_col = df_temp_p.columns.intersection(df_temp_mv.columns).intersection(df_temp.columns)
#     df_temp = df_temp[ov_col]
#     df_temp_p = df_temp_p[ov_col]
#     df_temp_mv = df_temp_mv[ov_col]

#     # the start year of factor construction (the next year of the sort variable's year)
#     start = int(start_criterion + 1)  
#     end = int(end_criterion + 1) 

#     # generate stock return dataframe
#     df_temp_r = df_temp_p/df_temp_p.shift(1)

#     # array for storing return values
#     quantile_return = np.empty((int((end_criterion-start_criterion+1)*12),quantile))

#     # slice target data for the given period
#     df_temp = df_temp[(start_criterion <= df_temp.index) & (df_temp.index <= end_criterion)] # from Jul 2003 to Jun 2022

#     # calcualte factor returns
#     for y, year in enumerate(df_temp.index):

#         df_per_year_cr = df_temp.iloc[y][:].to_frame('first') # row of year y (series) -> dataframe (shape=(n,1)) 
#         '''note. due to to_frame() the df is transposed '''

# '''
# 아래 수정중
# '''
#         # ESG 점수 및 MV가 없는 기업은 불필요하므로 삭제 (first 기준으로 삭제)
#         df_per_year_cr.dropna(inplace=True)

#         # generate quantile column, default is the no. of quantiles
#         df_per_year_cr['quantile'] = quantile - 1 # last no. of pf
        
#         # 100pt인 기업과 그 미만인 기업으로 분리
#         df_per_year_cr_100 = df_per_year_cr[df_per_year_cr['first']==100.0]
#         df_per_year_cr_under100 = df_per_year_cr[df_per_year_cr['first']<100.0]
        
#         # 100pt 미만 기업에 대하여 quantile 계산, 100pt의 quantile 값은 default 그대로 사용
#         df_per_year_cr_under100['quantile'] = pd.qcut(df_per_year_cr_under100['first'], quantile-1, labels=False)
        
#         # merge
#         df_per_year_cr = pd.concat([df_per_year_cr_100, df_per_year_cr_under100], axis=0)

#         # 각 pf no.에 맞는 array 생성
#         partition = []
#         for _q in range(quantile):
#             # Find the indices of rows with the target value
#             temp_q = df_per_year_cr.index[df_per_year_cr['quantile'] == _q].to_numpy()
#             partition.append(temp_q)

#         ##############################################################################################
#         # sort by ESG score (descending)
#         series_sort = df_temp.iloc[y][:].sort_values(ascending=False)
#         # saving non-NaN firm array by boolean indexing (not NaN)
#         notna_index = series_sort.notna()         
#         series_sort = series_sort[notna_index]

#         # storing firms in each quantile into partitions
#         n = series_sort.shape[0]
#         partition = []
#         adj_tie = [0] # adjusting same scored firms
#         for _ in range(quantile-1):
#             __ = 1
#             while series_sort[int((_+1)/quantile*n)] == series_sort[int((_+1)/quantile*n)+__] :
#                 __ += 1
#             adj_tie.append(int((_+1)/quantile*n+__))
#         adj_tie.append(n)
#         # print(adj_tie)
#         for _ in range(len(adj_tie)-1):
#             partition.append(series_sort.index[adj_tie[_]:adj_tie[_+1]])

#         # set time range : from July of year t+1 to June of year t+2
#         t_rg = pd.date_range(str(year+1)+'-06', freq='M', periods=13)  # get July of year t+2 to calculate r_t+1

#         ### calculate factor returns ###
#         # weight (mv): one month ahead / return: the given month
#         ## value weight
#         if weight_mode == 'value':
#             for q, firms_qt in enumerate(partition):
#                 # print('quantil no',q+1)
#                 for _, t in enumerate(t_rg[:-1]):
#                     r_t1 = np.array(df_temp_r.loc[t_rg[_+1],firms_qt].values) # r_t+1 vector             
#                     w_t = np.array(df_temp_mv.loc[t,firms_qt].values/df_temp_mv.loc[t,firms_qt].sum(skipna=True)) # w_t vector for value-weight portfolio
#                     # print(t_rg[_+1], np.nansum(r_t1*w_t), no_stock(r_t1, w_t)) # date - factor return - no. of stock            
#                     quantile_return[y*12+_,q] = np.nansum(r_t1*w_t) # store

#         ## equal weight
#         elif weight_mode == 'equal':
#             for q, firms_qt in enumerate(partition):
#                 # print('quantil no',q+1)
#                 for _, t in enumerate(t_rg[:-1]):
#                     r_t1 = np.array(df_temp_r.loc[t_rg[_+1],firms_qt].values) # r_t+1 vector
#                     # print(t_rg[_+1], np.nanmean(r_t1), no_stock(r_t1, r_t1))            
#                     quantile_return[y*12+_,q] = np.nanmean(r_t1) # store
#     # print(quantile_return) 
    
#     # generate df of factor returns (monthly)
#     period = pd.date_range('{}-07'.format(start), '{}-07'.format(end+1), freq='M')  
#     col = (np.arange(quantile) + 1).astype('str')
#     quantile_return_df = pd.DataFrame(quantile_return-1, columns=col, index=period) # result format: 0.xx (not 0.9xx - 1.xxx)
    
#     return quantile_return_df

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

##########################################################
# 2.3. Saving ONE-WAY portfolio return data
##########################################################

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

# Environment: Set quantile
quantile_set = 5

### save long-short pf as factor (high - low)
# Size
esg_to_at_value = saving_oneway_pf_vw(quantile=quantile_set, df_temp=df_ESG_to_at)
(esg_to_at_value.iloc[:,0] - esg_to_at_value.iloc[:,-1]).to_csv(f'result/esg_to_at_value_q{quantile_set}.csv') 
esg_to_sale_value = saving_oneway_pf_vw(quantile=quantile_set, df_temp=df_ESG_to_sale)
(esg_to_sale_value.iloc[:,0] - esg_to_sale_value.iloc[:,-1]).to_csv(f'result/esg_to_sale_value_q{quantile_set}.csv')

# Agency problem
esg_dot_liq_value = saving_oneway_pf_vw(quantile=quantile_set, df_temp=df_ESG_dot_liq)
(esg_dot_liq_value.iloc[:,0] - esg_dot_liq_value.iloc[:,-1]).to_csv(f'result/esg_dot_liq_value_q{quantile_set}.csv')
esg_dot_oancf_value = saving_oneway_pf_vw(quantile=quantile_set, df_temp=df_ESG_dot_oancf)
(esg_dot_oancf_value.iloc[:,0] - esg_dot_oancf_value.iloc[:,-1]).to_csv(f'result/esg_dot_oancf_value_q{quantile_set}.csv')
esg_to_booklev_value = saving_oneway_pf_vw(quantile=quantile_set, df_temp=df_ESG_to_booklev)
(esg_to_booklev_value.iloc[:,0] - esg_to_booklev_value.iloc[:,-1]).to_csv(f'result/esg_to_booklev_value_q{quantile_set}.csv')

# Perception
esg_to_ad_value = saving_oneway_pf_vw(quantile=quantile_set, df_temp=df_ESG_to_ad)
(esg_to_ad_value.iloc[:,0] - esg_to_ad_value.iloc[:,-1]).to_csv(f'result/esg_to_ad_value_q{quantile_set}.csv')

esg_to_at_value.describe()
esg_to_sale_value.describe()
esg_dot_liq_value.describe()
esg_dot_oancf_value.describe()
esg_to_booklev_value.describe()
esg_to_ad_value.describe()

##########################################################
# 2.4. Saving TWO-WAY portfolio return data
##########################################################
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

# Size
esg_to_at_value_two_size = saving_twoway_pf_vw(quantile_1=quantile_1_set, quantile_2=quantile_2_set, df_temp_1=df_ESG_to_at)
esg_to_at_value_two_size.to_csv(f'result/esg_to_at_value_two_size_q{quantile_1_set}_{quantile_2_set}.csv')
esg_to_sale_value_two_size = saving_twoway_pf_vw(quantile_1=quantile_1_set, quantile_2=quantile_2_set, df_temp_1=df_ESG_to_sale)
esg_to_sale_value_two_size.to_csv(f'result/esg_to_sale_value_two_size_q{quantile_1_set}_{quantile_2_set}.csv')

# Agency problem
esg_dot_liq_value_two_size = saving_twoway_pf_vw(quantile_1=quantile_1_set, quantile_2=quantile_2_set, df_temp_1=df_ESG_dot_liq)
esg_dot_liq_value_two_size.to_csv(f'result/esg_dot_liq_value_two_size_q{quantile_1_set}_{quantile_2_set}.csv')
esg_dot_oancf_value_two_size = saving_twoway_pf_vw(quantile_1=quantile_1_set, quantile_2=quantile_2_set, df_temp_1=df_ESG_dot_oancf)
esg_dot_oancf_value_two_size.to_csv(f'result/esg_dot_oancf_value_two_size_q{quantile_1_set}_{quantile_2_set}.csv')
esg_to_booklev_value_two_size = saving_twoway_pf_vw(quantile_1=quantile_1_set, quantile_2=quantile_2_set, df_temp_1=df_ESG_to_booklev)
esg_to_booklev_value_two_size.to_csv(f'result/esg_to_booklev_value_two_size_q{quantile_1_set}_{quantile_2_set}.csv')

# Perception
esg_to_ad_value_two_size = saving_twoway_pf_vw(quantile_1=quantile_1_set, quantile_2=quantile_2_set, df_temp_1=df_ESG_to_ad)
esg_to_ad_value_two_size.to_csv(f'result/esg_to_ad_value_two_size_q{quantile_1_set}_{quantile_2_set}.csv')

# describe
esg_to_at_value_two_size.describe()
esg_to_sale_value_two_size.describe()
esg_dot_liq_value_two_size.describe()
esg_dot_oancf_value_two_size.describe()
esg_to_booklev_value_two_size.describe()

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
    df_temp_1 = df_ESG_to_ad
    df_temp_2 = df_MV_year
    

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

# Size
esg_to_at_value_two_size_factor = saving_twoway_factor_vw(df_temp_1=df_ESG_to_at, 
                                                        quantile_1=saving_twoway_factor_vw_q1, 
                                                        quantile_2=saving_twoway_factor_vw_q2)
esg_to_at_value_two_size_factor.to_csv(f'result/esg_to_at_value_two_size_factor_q{saving_twoway_factor_vw_q1}_{saving_twoway_factor_vw_q2}.csv')
esg_to_sale_value_two_size_factor = saving_twoway_factor_vw(df_temp_1=df_ESG_to_sale,
                                                        quantile_1=saving_twoway_factor_vw_q1, 
                                                        quantile_2=saving_twoway_factor_vw_q2)
esg_to_sale_value_two_size_factor.to_csv(f'result/esg_to_sale_value_two_size_factor_q{saving_twoway_factor_vw_q1}_{saving_twoway_factor_vw_q2}.csv')

# Agency problem
esg_dot_liq_value_two_size_factor = saving_twoway_factor_vw(df_temp_1=df_ESG_dot_liq, 
                                                        quantile_1=saving_twoway_factor_vw_q1, 
                                                        quantile_2=saving_twoway_factor_vw_q2)
esg_dot_liq_value_two_size_factor.to_csv(f'result/esg_dot_liq_value_two_size_factor_q{saving_twoway_factor_vw_q1}_{saving_twoway_factor_vw_q2}.csv')
esg_dot_oancf_value_two_size_factor = saving_twoway_factor_vw(df_temp_1=df_ESG_dot_oancf,
                                                        quantile_1=saving_twoway_factor_vw_q1, 
                                                        quantile_2=saving_twoway_factor_vw_q2)
esg_dot_oancf_value_two_size_factor.to_csv(f'result/esg_dot_oancf_value_two_size_factor_q{saving_twoway_factor_vw_q1}_{saving_twoway_factor_vw_q2}.csv')
esg_to_booklev_value_two_size_factor = saving_twoway_factor_vw(df_temp_1=df_ESG_to_booklev,
                                                        quantile_1=saving_twoway_factor_vw_q1, 
                                                        quantile_2=saving_twoway_factor_vw_q2)
esg_to_booklev_value_two_size_factor.to_csv(f'result/esg_to_booklev_value_two_size_factor_q{saving_twoway_factor_vw_q1}_{saving_twoway_factor_vw_q2}.csv')

# Perception
esg_to_ad_value_two_size_factor = saving_twoway_factor_vw(df_temp_1=df_ESG_to_ad,
                                                        quantile_1=saving_twoway_factor_vw_q1, 
                                                        quantile_2=saving_twoway_factor_vw_q2)
esg_to_ad_value_two_size_factor.to_csv(f'result/esg_to_ad_value_two_size_factor_q{saving_twoway_factor_vw_q1}_{saving_twoway_factor_vw_q2}.csv')
