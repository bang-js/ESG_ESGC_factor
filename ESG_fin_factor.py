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
df_ESG_to_at = pd.read_csv('df_ESG_to_at.csv', index_col=0)
df_ESG_to_sale = pd.read_csv('df_ESG_to_sale.csv', index_col=0)
df_ESG_dot_liq = pd.read_csv('df_ESG_dot_liq.csv', index_col=0)
df_ESG_dot_oancf = pd.read_csv('df_ESG_dot_oancf.csv', index_col=0)
df_ESG_to_booklev = pd.read_csv('df_ESG_to_booklev.csv', index_col=0)
df_ESG_to_ad = pd.read_csv('df_ESG_to_ad.csv', index_col=0)

# pre-processing P and MV df
df_P = pd.read_csv('df_P.csv', index_col=0) # [253 rows x 8246 columns]
df_P.index = pd.to_datetime(df_P.index)
df_P.index = df_P.asfreq(freq='M').index # set frequency to 'M' for factor construnction  

df_MV = pd.read_csv('df_MV.csv', index_col=0) # [253 rows x 8242 columns]
df_MV.index = pd.to_datetime(df_MV.index)
df_MV.index = df_MV.asfreq(freq='M').index 

df_ESG_to_at.shape
# drop all NaN columns
df_ESG_to_at.dropna(axis=1, how='all').shape
# drop exactly same columns
df_ESG_to_at.T.drop_duplicates().T

# convert data frequency from month to year for df_MV
df_MV_year = df_MV.resample(rule='1Y').last() # monthly data -> yearly data, data: Dec 
df_MV_year.index = df_MV_year.index.year # re-indexing
df_MV_year.index.name = 'year'
df_MV_year = df_MV_year.loc[df_ESG_to_at.index,:] # match YEARs with df_ESG_...

##########################################################
# 2.2. 
##########################################################

def no_stock(x,y):
    '''
    return the number of non-NaN elements
    '''
    return min(x[~np.isnan(x)].shape[0], y[~np.isnan(y)].shape[0])

# ONE-WAY sort portfolio
def factor_cal(quantile, weight_mode, df_temp, df_temp_p=df_P, df_temp_mv=df_MV, start_criterion=2002, end_criterion=2020):
    '''
    quantile: quantile no.
    weight mode \in {value, equal, log-value}
    df_temp: df_ESG, df_ESGC, ... (YEARLY)
    df_temp_p: price dateframe (MONTHLY)
    df_temp_mv: market value dataframe (MONTHLY)
    start_criterion, end_criterion: the start year and end year of the criterion (e.g. ESG)
    '''
    # drop ESG or price or mv 없는 firm 
    ov_col = df_P.columns.intersection(df_ESG_to_sale.columns)    
    ov_col = ov_col.intersection(df_MV.columns)
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

    # slice esg data for the given period
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
        t_rg = pd.date_range(str(year+1)+'-06', freq='M', periods=13)  # r_t+1 계산을 위해 t+2 July 까지 확보

        ### calculate factor returns ###
        # weight (mv): one-head month / return: the given month
        ## value weight
        if weight_mode == 'value':
            for q, firms_qt in enumerate(partition):
                print('quantil no',q+1)
                for _, t in enumerate(t_rg[:-1]):
                    r_t1 = np.array(df_temp_r.loc[t_rg[_+1],firms_qt].values) # r_t+1 vector             
                    w_t = np.array(df_temp_mv.loc[t,firms_qt].values/df_temp_mv.loc[t,firms_qt].sum(skipna=True)) # w_t vector for value-weight portfolio
                    print(t_rg[_+1], np.nansum(r_t1*w_t), no_stock(r_t1, w_t)) # date - factor return - no. of stock            
                    # store
                    quantile_return[y*12+_,q] = np.nansum(r_t1*w_t)

        ## equal weight
        elif weight_mode == 'equal':
            for q, firms_qt in enumerate(partition):
                print('quantil no',q+1)
                for _, t in enumerate(t_rg[:-1]):
                    r_t1 = np.array(df_temp_r.loc[t_rg[_+1],firms_qt].values) # r_t+1 vector
                    print(t_rg[_+1], np.nanmean(r_t1), no_stock(r_t1, r_t1))            
                    # store
                    quantile_return[y*12+_,q] = np.nanmean(r_t1)
    # print(quantile_return) 
    
    # esg factor 계산
    period = pd.date_range('{}-07'.format(start), '{}-07'.format(end+1), freq='M')  
    col = (np.arange(quantile) + 1).astype('str')
    quantile_return_df = pd.DataFrame(quantile_return-1, columns=col, index=period) # result format: 0.xx (not 0.9xx - 1.xxx)
    quantile_return_df['factor'] = quantile_return_df['1'] - quantile_return_df[col[-1]] # high - low
    
    return quantile_return_df['factor']

##########################################################
# 2.3.
##########################################################

quantile = 5
_weight = 'value'
esg_to_at_value = factor_cal(quantile, _weight, df_temp=df_ESG_to_at)
esg_to_at_value.describe()

factor_temp = esg_to_at_value + 1
factor_temp = factor_temp.cumprod()
factor_temp.plot()
factor_temp