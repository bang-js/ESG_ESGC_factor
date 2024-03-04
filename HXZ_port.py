import pandas as pd
import numpy as np
import os
import datetime
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt

##############
# 1. merge downloaded data
# Hou, Xue, and Zhang's testing portfolio database: https://global-q.org/testingportfolios.html
##############
# ONE-WAY
def get_reframed(df_temp,name):
    '''
    rearrange dataframe from (months*10,1) to (months,10)
    '''
    df_temp.columns = ['year','month','rank','nstocks','ret_vw'] # rename for the 'rank_...'
    
    # calculate quantile
    quantile = df_temp['rank'].unique().shape[0]
    
    # generate date col. as type of datetime64[ns]
    df_temp['date'] = df_temp['year'].astype('str') + '-' + df_temp['month'].astype('str')
    df_temp['date'] = pd.to_datetime(df_temp['date'])
    
    # full date check
    df_temp_check = df_temp[df_temp['rank']==1]
    flag = (df_temp_check['date'] == pd.date_range(start=df_temp.iloc[0]['date'], end=df_temp.iloc[-1]['date'], freq='MS')).all()
    
    if flag:
        df_temp_rank = [] # seperating each rank
        for r in range(int(quantile)): # quantile =10
            _ = df_temp[df_temp['rank'] == r+1]

            # replace the date col. (the first day) to the last day
            _.iloc[:]['date'] = pd.date_range(start=df_temp.iloc[0]['date'], end=df_temp.iloc[-1]['date']+relativedelta(months=1), freq='M') # add one month to contain '2021-12-31' 

            _ = _[['date','ret_vw']]
            _.set_index('date', inplace=True)
            _.columns = ['ret_vw_{}_{}'.format(r+1,name)]
            df_temp_rank.append(_) # store

            # join into one dataframe (600x10) reculsively
            if r > 0:
                df_temp_rank[-1] = df_temp_rank[-1].join(df_temp_rank[-2])
    else: 
        print('check date_range')

    return df_temp_rank[-1]

# assign the folder path 
os.chdir('...\\HXZ')
HXZ_cate = os.listdir()

# load every portfolio file from every category
reframe_store = []
for c in HXZ_cate:
    os.chdir('...\\HXZ\\{}'.format(c))
    print(os.getcwd())
    port_lst_temp = os.listdir()
    for f in port_lst_temp:
        print(f)
        df = pd.read_csv('...\\HXZ\\' + c + '\\' + f)
        _ = get_reframed(df, f[6:-17]) # to constrain [6:-17] for pf name
        print(_.shape)
        
        # check the shape
        if df.shape[0] != _.shape[0] * _.shape[1] :
            print(df.shape, _.shape)
            print('match-error')
            break
        
        # merge the matrices
        reframe_store.append(_)
        if len(reframe_store) > 1:
             reframe_store[-1]= reframe_store[-1].join(reframe_store[-2], how='outer')
df_tot = reframe_store[-1]

# save
df_tot.to_csv('HXZ_port.csv')

#################
# 2. Analyze the portfolio 
#################
# load every portfolio file from every category
# show the cumulative return chart of portfolios if the cumulative return is larger than the given threshold
cum_thr = 100            # threshold for cumulative returns of portfolios

reframe_store = []
for c in HXZ_cate:
    os.chdir('...\\HXZ\\{}'.format(c))
    print(os.getcwd())
    port_lst_temp = os.listdir()
    for f in port_lst_temp:
        print(f)
        df = pd.read_csv('...\\HXZ\\' + c + '\\' + f)
        _ = get_reframed(df, f[6:-17])
        print(_.shape)
        
        # check the shape
        if df.shape[0] != _.shape[0] * _.shape[1] :
            print(df.shape, _.shape)
            print('match-error')
            break
        
        # cumulative return for each anomaly
        _r = ((_.iloc[:,0] - _.iloc[:,-1])*0.01+1).cumprod()
        
        '''Long short'''
#         if _r.iloc[-1] / _r.iloc[0] < 1: # if negative, long Low short High
#             _r = ((_.iloc[:,-1] - _.iloc[:,0])*0.01+1).cumprod()
#         if _r.iloc[-1] / _r.iloc[0] > cum_thr: # show only high cum return anomaly
#             _r.plot()
#             plt.show()

        '''long only'''
        # cumulative return for each anomaly
        _r = ((_.iloc[:,0])*0.01+1).cumprod()
        if _r.iloc[-1] / _r.iloc[0] < 1: # if negative, long Low short High
            _r = ((_.iloc[:,-1] )*0.01+1).cumprod()
        if _r.iloc[-1] / _r.iloc[0] > cum_thr: # show only high cum return anomaly
            # plot the log chart
            np.log(_r).plot()
            plt.show()
