# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
from numpy import matlib as mb
from matplotlib import pyplot as plt
import sys
import seaborn as sns
import datetime
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import Lasso, lasso_path, LassoCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.decomposition import PCA
import calendar
from scipy.stats import t


# # import and merge

# ## Q and FF5

def convert_index_format(index_str):
    """Convert the index with the format 'month name-last two digits of the year' to 'year-month-the last day'.
    Parameters:
        index_str (str): The index value in the original format."""
    # Parse the original index to extract month and year
    month_name, last_two_digits = index_str.split('-')
    month = datetime.datetime.strptime(month_name, '%b').month
    if int(last_two_digits) > 50: # in the 20th century
        year = int('19' + last_two_digits)  
    else: # in the 21st century
        year = int('20' + last_two_digits)
        
    # Calculate the last day of the month using the calendar module
    last_day = calendar.monthrange(year, month)[1]

    # Format the date in the desired format: 'year-month-the last day'
    new_index = f'{year}-{month:02d}-{last_day:02d}'

    return new_index


# import Qfactor
Qfactor = pd.read_csv('C:/Users/Jeongseok_Bang/OneDrive/바탕 화면/석사과정/FinLab/23 4 ESG controversies risk factor/data/Qfactor.csv', index_col=0)
Qfactor.index = pd.date_range(convert_index_format(Qfactor.index[0]), convert_index_format(Qfactor.index[-1]), freq='M')
Qfactor

# import FF5 factor
FF5 = pd.read_csv('C:/Users/Jeongseok_Bang/OneDrive/바탕 화면/석사과정/FinLab/23 4 ESG controversies risk factor/data/FF5.csv', index_col=0)
FF5.index = pd.date_range('1963-07-31', '2023-02-28', freq='M')
# FF5 *= 0.01
FF5.rename(columns={'Mkt-RF':'MktRF'}, inplace=True)
FF5

# emerge FF5 and Qfactor
allfactors = FF5.join(Qfactor, how='outer')
allfactors

# ## ESG factor

quantile_1_set = 3
quantile_2_set = 2

esg_factor_data_list = [f'esg_value_two_size_factor_q{quantile_1_set}_{quantile_2_set}', 
                        f'esg_value_2_two_size_factor_q{quantile_1_set}_{quantile_2_set}', 
                        f'env_value_two_size_factor_q{quantile_1_set}_{quantile_2_set}', 
                        f'soc_value_two_size_factor_q{quantile_1_set}_{quantile_2_set}', 
                        f'gov_value_two_size_factor_q{quantile_1_set}_{quantile_2_set}']

# +
# esg_factor_data_list = [f'env_value_two_size_factor_q{quantile_1_set}_{quantile_2_set}', 
#                         f'soc_value_two_size_factor_q{quantile_1_set}_{quantile_2_set}', 
#                         f'gov_value_two_size_factor_q{quantile_1_set}_{quantile_2_set}']
# -

# ### merge allfactors and esg_factor

# +
# esg_temp =  pd.read_csv('C:/Users/Jeongseok_Bang/OneDrive/바탕 화면/석사과정/FinLab/22 11 ESG risk factor (제안서)/code/esg_size_mulfactor.csv', index_col=0)
# esg_temp.index = pd.to_datetime(esg_temp.index)
# esg_temp.columns = ['esg'.format(i)]
# esg_temp *= 100
# esg_temp = esg_temp.iloc[:222,:]
# -

# import and merge all esg factors
for i, e in enumerate(esg_factor_data_list):
    e_temp =  pd.read_csv('result/{}.csv'.format(e), index_col=0)
    e_temp.index = pd.to_datetime(e_temp.index)
    e_temp.columns = ['esg_f_{}'.format(i)]
    e_temp *= 100
    
    if i == 0:
        esg_factors = e_temp
    else: # i>0
        esg_factors = esg_factors.join(e_temp, how='outer')

# +
# esg_factors = esg_temp.join(esg_factors, how='outer')
# -

esg_factors.columns = ['ESG','ESG2','ENV', 'SOC', 'GOV']

_1 = esg_factors['ESG2'].T.quantile(np.arange(0,1.1,0.1)).round(2).T
_2 = esg_factors['ESG'].T.quantile(np.arange(0,1.1,0.1)).round(2).T
pd.concat([_1,_2], axis=1)

esg_factors.describe()

# ### merge allfactors and esg_factor

allfactors = allfactors.join(esg_factors, how='outer')

# ## Period cut

factors = allfactors[(allfactors.index >= '2003-07-31') & (allfactors.index <= '2021-12-31')]
factors

factors.dropna(inplace=True)
factors

# ## test portfolio

# import HXZ portfolio
HXZ_port = pd.read_csv('C:/Users/Jeongseok_Bang/OneDrive/바탕 화면/석사과정/FinLab/22 11 ESG risk factor (제안서)/code/HXZ_port.csv', index_col=0)
HXZ_port.index = pd.date_range('1967-01-31', '2021-12-31', freq='M')

HXZ_port = HXZ_port.loc[factors.index[0]:][:]

# import JKP factors
JKP = pd.read_csv('C:/Users/Jeongseok_Bang/OneDrive/바탕 화면/석사과정/FinLab/22 11 ESG risk factor (제안서)/code/df_factorzoo_JKP.csv', index_col=0)
JKP.index.names = ['Date']
JKP.index = pd.to_datetime(JKP.index)

JKP = JKP.loc[factors.index[0]:][:]

# ## Momentum from JKP

factors['MOM'] = JKP['ret_12_1'] * 100
factors['market_equity'] = JKP['market_equity'] * 100
factors['be_me'] = JKP['be_me'] * 100

# ## select factors (from 1st_sel.ipynb)

# 0 MktRf 0.925\
# 42 qmj 0.875\
# 68 tangibility 0.85\
# 45 rd_me 0.85\
# 11 dbnetis_at 0.825\
# 31 ni_inc8q 0.8\
# 18 earnings_variability 0.8\
# 50 ret_1_0 0.75\
# 19 eqnetis_at 0.75\
# 20 f_score 0.725

factors['qmj'] = JKP['qmj'] * 100
factors['tangibility'] = JKP['tangibility'] * 100
factors['rd_me'] = JKP['rd_me'] * 100
factors['dbnetis_at'] = JKP['dbnetis_at'] * 100
factors['ni_inc8q'] = JKP['ni_inc8q'] * 100
factors['earnings_variability'] = JKP['earnings_variability'] * 100
factors['ret_1_0'] = JKP['ret_1_0'] * 100
factors['eqnetis_at'] = JKP['eqnetis_at'] * 100
factors['f_score'] = JKP['f_score'] * 100

# ## summary stat.

round(factors.describe(), 3)

total_factor_list = ['MktRF', 'SMB', 'HML', 'MOM', 'RMW', 'CMA',	'R_IA',	'R_ROE']
total_factor_list.extend(esg_factors.columns.to_list())
round(factors[total_factor_list].corr(),3)

sel_factor_list = ['qmj', 'tangibility', 'rd_me', 'dbnetis_at', 'ni_inc8q', 'earnings_variability', 'ret_1_0', 'eqnetis_at', 'f_score']
sel_factor_list.extend(esg_factors.columns.to_list())
round(factors[sel_factor_list].corr(),3)

t_list = ['MktRF', 'SMB', 'HML', 'MOM', 'RMW', 'CMA',	'R_IA',	'R_ROE'] + ['qmj', 'tangibility', 'rd_me', 'dbnetis_at', 'ni_inc8q', 'earnings_variability', 'ret_1_0', 'eqnetis_at', 'f_score'] \
+ list(esg_factors.columns)
factors[t_list].corr().to_csv('factor_corr.csv')

# +
cum_factors = factors*0.01 + 1
cum_factors = cum_factors.cumprod()

cum_factors['MktRF'].plot()
cum_factors[esg_factors.columns].plot()
# -

# # Decile pf analysis

quantile_set = 10

# ## one sample: $\bar{R}$

esg_oneway_pf_list = [f'esg_value_q{quantile_set}',
                      f'esg_value_2_q{quantile_set}',
                        f'env_value_q{quantile_set}', 
                        f'soc_value_q{quantile_set}',
                        f'gov_value_q{quantile_set}']

esg_oneway_hl_list = [f'esg_value_factor_q{quantile_set}', 
                      f'esg_value_2_factor_q{quantile_set}', 
                        f'env_value_factor_q{quantile_set}', 
                        f'soc_value_factor_q{quantile_set}',
                        f'gov_value_factor_q{quantile_set}']

esg_col = ['ESG','ESG2','ENV', 'SOC', 'GOV']

# import and merge esg decile pf
for i in range(len(esg_oneway_pf_list)):
    e_temp =  pd.read_csv('result/{}.csv'.format(esg_oneway_pf_list[i]), index_col=0)
    e_temp.index = pd.to_datetime(e_temp.index)
    e_temp.columns = [f'{esg_col[i]}_p{k+1}' for k in range(quantile_set)]
    
    e_temp_hl = pd.read_csv('result/{}.csv'.format(esg_oneway_hl_list[i]), index_col=0)
    e_temp_hl.index = pd.to_datetime(e_temp_hl.index)
    e_temp_hl.columns = [f'{esg_col[i]}_HL']
    
    e_temp *= 100
    e_temp_hl *= 100
    
    if i == 0:
        esg_oneway_pf = e_temp
        esg_oneway_pf = esg_oneway_pf.join(e_temp_hl, how='outer')
    else: # i>0
        esg_oneway_pf = esg_oneway_pf.join(e_temp, how='outer')
        esg_oneway_pf = esg_oneway_pf.join(e_temp_hl, how='outer')

esg_oneway_pf

print('check the period of RF and esg_oneway_pf:', factors['R_F'].shape[0] == esg_oneway_pf.shape[0])

oneway_pf = esg_oneway_pf[esg_oneway_pf.columns[esg_oneway_pf.columns.str.contains('ESG2')]]
oneway_pf.columns

oneway_pf_ex = oneway_pf - factors['R_F'].values.reshape((-1,1))
oneway_pf_ex

oneway_pf_ex.describe().round(3)


# Define a function to calculate Newey-West standard errors
def calculate_newey_west_std_errors(data, lag=6):
    X = sm.add_constant(data)
    model = sm.OLS(X[:, 1], X[:, 0])
    result = model.fit(cov_type='HAC', cov_kwds={'maxlags': lag})
    
    # asterisk
    if 0.1 >= result.pvalues > 0.05:
        k = 1
    elif 0.05 >= result.pvalues> 0.01:
        k = 2
    elif 0.01 >= result.pvalues :
        k = 3
    else:
        k = 0 
    pv_ast = '*'*k
    
    return result.tvalues[0], result.pvalues[0], pv_ast


temp_lst = []
for c in oneway_pf_ex.columns:
    mean = oneway_pf_ex.describe().loc['mean',c]
    tv, pv, pv_ast = calculate_newey_west_std_errors(oneway_pf_ex[c].values)
    temp_lst.append([f'{mean:.3f}'+pv_ast, f'({tv:.2f})'])

pd.DataFrame(np.array(temp_lst).T ,index=['mean','t'],columns=oneway_pf_ex.columns)


# ## alpha for decile

# ### store_alpha_coef_tstat

def store_alpha_coef_tstat(x):
    '''
    store only coef and tstat of the alpha (constant)
    
    return: coefficient+asterisk, t-statistics
    '''
    coefs = x.params
    tstats = x.tvalues 
    pvalues = x.pvalues 
    
    if 0.1 >= pvalues[0] > 0.05:
        k = 1
    elif 0.05 >= pvalues[0] > 0.01:
        k = 2
    elif 0.01 >= pvalues[0] :
        k = 3
    else:
        k = 0 
    ast_ = '*'*k
    
    return f'{coefs[0]:.3f}{ast_}', f'({tstats[0]:.2f})'


# ### analysis

pf_mg_factors = oneway_pf_ex.join(factors, how='outer')

pf_mg_factors

# +
coef_lst = []
tstat_lst = []

for i, f_name in enumerate(oneway_pf_ex.columns):
    print('#'*100)
    print('#'+f_name+'#')
    print('#'*100)
    
    target_alpha = f_name
    
    temp_coef_lst = []
    temp_tstat_lst = []
    
    # CAPM
    capm = sm.OLS(pf_mg_factors[target_alpha], sm.add_constant(pf_mg_factors['MktRF'])).fit(cov_type='HAC',cov_kwds={'maxlags':6})
    temp_c, temp_t = store_alpha_coef_tstat(capm)
    temp_coef_lst.append(temp_c)
    temp_tstat_lst.append(temp_t)
    print(capm.summary())
    
    # FF3
    ff3 = smf.ols(formula='{} ~ MktRF + SMB + HML'.format(target_alpha), data=pf_mg_factors.iloc[:][:]).fit(cov_type='HAC',cov_kwds={'maxlags':6})
    temp_c, temp_t = store_alpha_coef_tstat(ff3)
    temp_coef_lst.append(temp_c)
    temp_tstat_lst.append(temp_t)
    print(ff3.summary())
    
    # Carhart 4
    c4 = smf.ols(formula='{} ~ MktRF + SMB + HML + MOM'.format(target_alpha), data=pf_mg_factors.iloc[:][:]).fit(cov_type='HAC',cov_kwds={'maxlags':6})
    temp_c, temp_t = store_alpha_coef_tstat(c4)
    temp_coef_lst.append(temp_c)
    temp_tstat_lst.append(temp_t)
    print(c4.summary())
    
    # FF5
    ff5 = smf.ols(formula='{} ~ MktRF + SMB + HML + RMW + CMA'.format(target_alpha), data=pf_mg_factors.iloc[:][:]).fit(cov_type='HAC',cov_kwds={'maxlags':6})
    temp_c, temp_t = store_alpha_coef_tstat(ff5)
    temp_coef_lst.append(temp_c)
    temp_tstat_lst.append(temp_t)
    print(ff5.summary())
    
    # q
    q_factor = smf.ols(formula='{} ~ MktRF + SMB + R_IA + R_ROE'.format(target_alpha), data=pf_mg_factors.iloc[:][:]).fit(cov_type='HAC',cov_kwds={'maxlags':6})
    temp_c, temp_t = store_alpha_coef_tstat(q_factor)
    temp_coef_lst.append(temp_c)
    temp_tstat_lst.append(temp_t)
    print(q_factor.summary())
    
    # new
    new_model = smf.ols(formula='{} ~ MktRF + qmj+ tangibility+ rd_me+ dbnetis_at+ ni_inc8q+ earnings_variability+ ret_1_0+ eqnetis_at+ f_score'.format(target_alpha), data=pf_mg_factors.iloc[:][:]).fit(cov_type='HAC',cov_kwds={'maxlags':6})
    temp_c, temp_t = store_alpha_coef_tstat(new_model)
    temp_coef_lst.append(temp_c)
    temp_tstat_lst.append(temp_t)
    print(new_model.summary())
    
    coef_lst.append(temp_coef_lst)
    tstat_lst.append(temp_tstat_lst)
# -

factor_model_name = ['capm', 'ff3', 'c4', 'ff4', 'q5', 'new']

arr_coef_tstat = np.empty((len(factor_model_name)*2, oneway_pf_ex.shape[1]), dtype=object)
arr_coef = np.array(coef_lst).T # (no of fac mo, no of pf)
arr_tstat = np.array(tstat_lst).T
for i in range(arr_coef.shape[0]):
    arr_coef_tstat[2*i,:] = arr_coef[i,:]
    arr_coef_tstat[2*i+1,:] = arr_tstat[i,:]

index_ = [f'{__}_{_}' for _ in factor_model_name for __ in ['a','t']]
pd.DataFrame(arr_coef_tstat, index=index_, columns=oneway_pf_ex.columns)



# # alpha - pricing anomaly

for i, f_name in enumerate(esg_factors.columns.to_list()):
    factors[f'y{i}'] = factors[f_name] - factors['R_F'] 

factors


def show_coef_tstat(x):
    coefs = round(x.params, 3)
    tstats = round(x.tvalues, 2) 
    pvalues = x.pvalues 
    
    for i in range(coefs.shape[0]):
        if 0.1 >= pvalues[i] > 0.05:
            k = 1
        elif 0.05 >= pvalues[i] > 0.01:
            k = 2
        elif 0.01 >= pvalues[i] :
            k = 3
        else:
            k = 0 
        print('{:.3f}'.format(coefs[i])+'*'*k+'\n({:.2f})'.format(tstats[i]))


def alpha_anl(factors):
    for i, f_name in enumerate(esg_factors.columns):
        print('#'*100)
        print('#'+f_name+'#')
        print('#'*100)

        target_alpha = f'y{i}'

        # CAPM
        capm = sm.OLS(factors[target_alpha], sm.add_constant(factors['MktRF'])).fit(cov_type='HAC',cov_kwds={'maxlags':6})
        show_coef_tstat(capm)
        print(capm.summary())

        # FF3
        ff3 = smf.ols(formula='{} ~ MktRF + SMB + HML'.format(target_alpha), data=factors.iloc[:][:]).fit(cov_type='HAC',cov_kwds={'maxlags':6})
        show_coef_tstat(ff3)
        print(ff3.summary())

        # Carhart 4
        c4 = smf.ols(formula='{} ~ MktRF + SMB + HML + MOM'.format(target_alpha), data=factors.iloc[:][:]).fit(cov_type='HAC',cov_kwds={'maxlags':6})
        show_coef_tstat(c4)
        print(c4.summary())

        # FF5
        ff5 = smf.ols(formula='{} ~ MktRF + SMB + HML + RMW + CMA'.format(target_alpha), data=factors.iloc[:][:]).fit(cov_type='HAC',cov_kwds={'maxlags':6})
        show_coef_tstat(ff5)
        print(ff5.summary())

        # q
        q_factor = smf.ols(formula='{} ~ MktRF + SMB + R_IA + R_ROE'.format(target_alpha), data=factors.iloc[:][:]).fit(cov_type='HAC',cov_kwds={'maxlags':6})
        show_coef_tstat(q_factor)
        print(q_factor.summary())

        # new
        new_model = smf.ols(formula='{} ~ MktRF + qmj+ tangibility+ rd_me+ dbnetis_at+ ni_inc8q+ earnings_variability+ ret_1_0+ eqnetis_at+ f_score'.format(target_alpha), data=factors.iloc[:][:]).fit(cov_type='HAC',cov_kwds={'maxlags':6})
        show_coef_tstat(new_model)
        print(new_model.summary())


alpha_anl(factors)

id_ = esg_factors.columns.to_list().index('ESG2')
target_alpha = f'y{id_}'

# +
# new Top 4 (1st select rate: ~0.85)
new_model = smf.ols(formula='{} ~ MktRF + qmj+ tangibility+ rd_me'.format(target_alpha), data=factors.iloc[:][:]).fit(cov_type='HAC',cov_kwds={'maxlags':6})
show_coef_tstat(new_model)
print(new_model.summary())

# new Top 5 (~0.825)
new_model = smf.ols(formula='{} ~ MktRF + qmj+ tangibility+ rd_me+ dbnetis_at'.format(target_alpha), data=factors.iloc[:][:]).fit(cov_type='HAC',cov_kwds={'maxlags':6})
show_coef_tstat(new_model)
print(new_model.summary())

# new Top 7 (~0.8)
new_model = smf.ols(formula='{} ~ MktRF + qmj+ tangibility+ rd_me+ dbnetis_at+ ni_inc8q+ earnings_variability'.format(target_alpha), data=factors.iloc[:][:]).fit(cov_type='HAC',cov_kwds={'maxlags':6})
show_coef_tstat(new_model)
print(new_model.summary())

# new Top 9 (~0.75)
new_model = smf.ols(formula='{} ~ MktRF + qmj+ tangibility+ rd_me+ dbnetis_at+ ni_inc8q+ earnings_variability+ ret_1_0+ eqnetis_at'.format(target_alpha), data=factors.iloc[:][:]).fit(cov_type='HAC',cov_kwds={'maxlags':6})
show_coef_tstat(new_model)
print(new_model.summary())

# new all= Top10 (~0.725)
new_model = smf.ols(formula='{} ~ MktRF + qmj+ tangibility+ rd_me+ dbnetis_at+ ni_inc8q+ earnings_variability+ ret_1_0+ eqnetis_at+ f_score'.format(target_alpha), data=factors.iloc[:][:]).fit(cov_type='HAC',cov_kwds={'maxlags':6})
show_coef_tstat(new_model)
print(new_model.summary())


# -

# # Cov. risk price - simple

def PriceRisk_OLS(Ri, gt, ht):
    n, T = Ri.shape
    d = gt.shape[0]
    p = ht.shape[0]

    np_Ri = Ri.values

    cov_h = np.empty((n,p))
    for nn in range(n):
        Rh = np.append((np_Ri[nn,:]).reshape(-1,1), ht.T, axis=1) # T x 1 + T x p 
        Rh_cov = pd.DataFrame(Rh).dropna().cov() # (1+p)x(1+p)
        cov_h[nn, :] = Rh_cov.values[0, 1:] # [cov(r1,f1), cov(r1,f2), ...] 로 구성된 1xp vector

    cov_g = np.empty((n,d))
    for nn in range(n):    
        Rg = np.append((np_Ri[nn,:]).reshape(-1,1), gt.T, axis=1) # T x 1 + T x d = Tx2
        Rg_cov = pd.DataFrame(Rg).dropna().cov()
        cov_g[nn, :] = Rg_cov.values[0, 1:]

    ER  = np.mean(Ri, axis=1)
    ER = ER.values
    
    one_vec = np.ones((n,1))
    M0 = np.eye(n) - one_vec@ np.linalg.inv(one_vec.T@one_vec)@one_vec.T # projection matrix of one vector

    # For no selection OLS
    X = np.append(cov_g, cov_h, axis=1) # n x (p+d)
    X_zero = np.append(np.ones((n,1)), X, axis=1) # n x (1+p+d)
    lambda_full_zero = np.linalg.inv(X_zero.T@X_zero) @ (X_zero.T @ ER) # OLS estimator
    lambda_full = np.linalg.inv(X.T@M0@X) @ (X.T@M0@ER) # OLS estimator w/o constant
    lambdag_ols = lambda_full[0:d]

    # R2
    ESS = (X_zero @ lambda_full_zero - ER.mean()).T @ (X_zero @ lambda_full_zero - ER.mean()) 
    TSS = (ER - ER.mean()).T @ (ER - ER.mean())
    R2 = ESS/TSS
    
    # eliminate rows with missing values
    hg = np.append(ht, gt, axis=0)
    nomissing = np.where((np.isnan(hg)).any(axis=0) == False)[0]
    Lnm = nomissing.shape[0]

    # calculate avar_ols
    zhat = np.empty((d,Lnm))

    for i in range(d):
        M_mdl = np.eye(Lnm) - ht[:,nomissing].T @ np.linalg.inv(ht[:,nomissing]@ht[:,nomissing].T) @ ht[:,nomissing] # orthogonal space of projection matrix of h(control)
        zhat[i,:] = (M_mdl @ gt[i,nomissing].T).reshape((-1,)) # residual from regression of g(test) on h

    Sigmazhat = (zhat @ zhat.T)/Lnm # RSS

    temp3 = np.zeros((d,d)) 

    for ii, l in enumerate(nomissing):
        mt = 1 - lambda_full.T @ (np.append(gt[:,l], ht[:,l], axis=0)).reshape((-1,1)) 
        temp3 += (mt**2) @ (np.linalg.inv(Sigmazhat) @ zhat[:,ii].reshape(-1,1) @ zhat[:,ii].reshape(1,-1) @ np.linalg.inv(Sigmazhat))

    avar_lambdag3 = np.diag(temp3)/Lnm
    se3 = np.sqrt(avar_lambdag3/Lnm)

    # scaled lambda for DS
    vt = np.append(gt[:,nomissing], ht[:,nomissing], axis=0)
    V_bar = vt - (np.mean(vt, axis=1)).reshape((-1,1)) @ np.ones((1,Lnm))
    var_v = (V_bar @ V_bar.T)/Lnm
    lambda_ols = np.multiply(np.diag(var_v), lambda_full) 
    
    return_dict = dict({
        'lambdag_ols': lambdag_ols,
        'se_ols': se3,
        'lambda_ols': lambda_ols,
        'lambda_ols_zero': lambda_full_zero,
        'R2': R2
    })
    
    return return_dict


def PriceRisk_OLS_C(Ri, ht):
    '''
    return is just regression results, so it need to manipulate with .summary or esle
    '''
    n, T = Ri.shape
    p = ht.shape[0]

    np_Ri = Ri.values

    cov_h = np.empty((n,p))
    for nn in range(n):
        Rh = np.append((np_Ri[nn,:]).reshape(-1,1), ht.T, axis=1) # T x 1 + T x p 
        Rh_cov = pd.DataFrame(Rh).dropna().cov()
        cov_h[nn, :] = Rh_cov.values[0, 1:]

    ER  = np.mean(Ri, axis=1)
    ER = ER.values

    X = np.append(np.ones((n,1)), cov_h, axis=1)
    results = sm.OLS(endog=ER, exog=X).fit(cov_type='HAC',cov_kwds={'maxlags':6})

    return results


HXZ_port_ex = HXZ_port - factors['R_F'].values.reshape((-1,1))
HXZ_port_ex

Ri = HXZ_port_ex.iloc[:][:]

factorname = factors.columns
factorname_full = factors.columns

factors_v = factors.iloc[:][:].values

# +
# CAPM
mkt_ind = int((np.where(factorname == 'MktRF'))[0])
CAPM = factors_v[:,[mkt_ind]].T

# Fama French 3 factor model
smb_ind = int((np.where(factorname == 'SMB'))[0]) 
hml_ind = int((np.where(factorname == 'HML'))[0]) 
FF3 = factors_v[:,[mkt_ind,smb_ind,hml_ind]].T

# Carharr 4 factor model
mom_ind = int((np.where(factorname == 'MOM'))[0])
C4 = factors_v[:,[mkt_ind, smb_ind, hml_ind, mom_ind]].T

# Fama French 5 factor model
rmw_ind = int((np.where(factorname == 'RMW'))[0]) 
cma_ind = int((np.where(factorname == 'CMA'))[0])
FF5 = factors_v[:,[mkt_ind, smb_ind, hml_ind, rmw_ind, cma_ind]].T

# HXZ q-factor model
me_ind = int((np.where(factorname == 'SMB'))[0]) 
ia_ind = int((np.where(factorname == 'R_IA'))[0]) 
roe_ind = int((np.where(factorname == 'R_ROE'))[0]) 
Qfactor = factors_v[:,[mkt_ind, me_ind, ia_ind, roe_ind]].T

# New model
qmj_ind = int((np.where(factorname =='qmj'))[0])
tangibility_ind = int((np.where(factorname =='tangibility'))[0])
rd_me_ind = int((np.where(factorname =='rd_me'))[0])
dbnetis_at_ind = int((np.where(factorname =='dbnetis_at'))[0])
ni_inc8q_ind = int((np.where(factorname =='ni_inc8q'))[0])
earnings_variability_ind = int((np.where(factorname =='earnings_variability'))[0])
ret_1_0_ind = int((np.where(factorname =='ret_1_0'))[0])
eqnetis_at_ind = int((np.where(factorname =='eqnetis_at'))[0])
f_score_ind = int((np.where(factorname =='f_score'))[0])
newmodel = factors_v[:,[mkt_ind, qmj_ind, tangibility_ind, rd_me_ind, dbnetis_at_ind, ni_inc8q_ind, earnings_variability_ind,
                       ret_1_0_ind, eqnetis_at_ind, f_score_ind]].T
# -

# New model SUBSET
newmodel_1 = factors_v[:,[mkt_ind, qmj_ind, tangibility_ind, rd_me_ind]].T
newmodel_2 = factors_v[:,[mkt_ind, qmj_ind, tangibility_ind, rd_me_ind, dbnetis_at_ind]].T
newmodel_3 = factors_v[:,[mkt_ind, qmj_ind, tangibility_ind, rd_me_ind, dbnetis_at_ind, ni_inc8q_ind, earnings_variability_ind]].T
newmodel_4 = factors_v[:,[mkt_ind, qmj_ind, tangibility_ind, rd_me_ind, dbnetis_at_ind, ni_inc8q_ind, earnings_variability_ind,
                       ret_1_0_ind, eqnetis_at_ind]].T


def PriceRisk_OLS_2(Ri, gt, ht):
    n, T = Ri.shape
    d = gt.shape[0]
    p = ht.shape[0]

    np_Ri = Ri.values

    cov_h = np.empty((n,p))
    for nn in range(n):
        Rh = np.append((np_Ri[nn,:]).reshape(-1,1), ht.T, axis=1) # T x 1 + T x p 
        Rh_cov = pd.DataFrame(Rh).dropna().cov()
        cov_h[nn, :] = Rh_cov.values[0, 1:]

    cov_g = np.empty((n,d))
    for nn in range(n):    
        Rg = np.append((np_Ri[nn,:]).reshape(-1,1), gt.T, axis=1) # T x 1 + T x d
        Rg_cov = pd.DataFrame(Rg).dropna().cov()
        cov_g[nn, :] = Rg_cov.values[0, 1:]

    ER  = np.mean(Ri, axis=1)
    ER = ER.values

    X = np.append(cov_g, cov_h, axis=1)
    X = np.append(np.ones((n,1)), X, axis=1)
    results = sm.OLS(endog=ER, exog=X).fit(cov_type='HAC',cov_kwds={'maxlags':6}) #222

    return results


def show_coef_tstat_np(x):
    coefs = np.round(x.params, 3)
    tstats = np.round(x.tvalues, 2) 
    pvalues = x.pvalues 
    
    for i in range(coefs.shape[0]):
        if 0.1 >= pvalues[i] > 0.05:
            k = 1
        elif 0.05 >= pvalues[i] > 0.01:
            k = 2
        elif 0.01 >= pvalues[i] :
            k = 3
        else:
            k = 0 
        print('{:.3f}'.format(coefs[i])+'*'*k+'\n({:.2f})'.format(tstats[i]))


# esg_factor_name_lst
esg_factors.columns

# +
target_factor = 'ESG2'

'''
one esg factor
'''
esg_ind = int((np.where(factorname == target_factor))[0]) 
gt = factors_v[:,esg_ind].reshape(1,-1)
Ctrs = [CAPM, FF3, C4, FF5, Qfactor, newmodel]
Ctrs_exp = ['const - ESG - mkt', 'const - ESG - mkt s h', 'const - ESG - mkt s h m', 'const - ESG - mkt s h r c', 'const - ESG - mkt s ia roe', 'const - ESG - mkt ...']

for i, c in enumerate(Ctrs):
    print(PriceRisk_OLS_2(Ri.T, gt, c).summary())
    print(Ctrs_exp[i])        
    show_coef_tstat_np(PriceRisk_OLS_2(Ri.T, gt, c))
    print('='*10)
    
'''
one esg factor: new models
'''
esg_ind = int((np.where(factorname == target_factor))[0]) 
gt = factors_v[:,esg_ind].reshape(1,-1)
Ctrs = [newmodel_1, newmodel_2, newmodel_3, newmodel_4, newmodel]
Ctrs_exp = ['const - ESG - mkt...rdme', 'const - ESG - mkt...dbnetis', 'const - ESG - mkt...EVar', 'const - ESG - mkt...eqnetis', 'const - ESG - mkt...fscore']

for i, c in enumerate(Ctrs):
    print(PriceRisk_OLS_2(Ri.T, gt, c).summary())
    print(Ctrs_exp[i])        
    show_coef_tstat_np(PriceRisk_OLS_2(Ri.T, gt, c))
    print('='*10)

# +
target_factor = 'ENV'

'''
one esg factor
'''
esg_ind = int((np.where(factorname == target_factor))[0]) 
gt = factors_v[:,esg_ind].reshape(1,-1)
Ctrs = [CAPM, FF3, C4, FF5, Qfactor, newmodel]
Ctrs_exp = ['const - ESG - mkt', 'const - ESG - mkt s h', 'const - ESG - mkt s h m', 'const - ESG - mkt s h r c', 'const - ESG - mkt s ia roe', 'const - ESG - mkt ...']

for i, c in enumerate(Ctrs):
    print(PriceRisk_OLS_2(Ri.T, gt, c).summary())
    print(Ctrs_exp[i])        
    show_coef_tstat_np(PriceRisk_OLS_2(Ri.T, gt, c))
    print('='*10)
    
'''
one esg factor: new models
'''
esg_ind = int((np.where(factorname == target_factor))[0]) 
gt = factors_v[:,esg_ind].reshape(1,-1)
Ctrs = [newmodel_1, newmodel_2, newmodel_3, newmodel_4, newmodel]
Ctrs_exp = ['const - ESG - mkt...rdme', 'const - ESG - mkt...dbnetis', 'const - ESG - mkt...EVar', 'const - ESG - mkt...eqnetis', 'const - ESG - mkt...fscore']

for i, c in enumerate(Ctrs):
    print(PriceRisk_OLS_2(Ri.T, gt, c).summary())
    print(Ctrs_exp[i])        
    show_coef_tstat_np(PriceRisk_OLS_2(Ri.T, gt, c))
    print('='*10)

# +
'''
one esg factor
'''
esg_ind = int((np.where(factorname == 'SOC'))[0]) 
gt = factors_v[:,esg_ind].reshape(1,-1)
Ctrs = [CAPM, FF3, C4, FF5, Qfactor, newmodel]
Ctrs_exp = ['const - ESG - mkt', 'const - ESG - mkt s h', 'const - ESG - mkt s h m', 'const - ESG - mkt s h r c', 'const - ESG - mkt s ia roe', 'const - ESG - mkt ...']

for i, c in enumerate(Ctrs):
    print(PriceRisk_OLS_2(Ri.T, gt, c).summary())
    print(Ctrs_exp[i])        
    show_coef_tstat_np(PriceRisk_OLS_2(Ri.T, gt, c))
    print('='*10)
    
'''
one esg factor: new models
'''
esg_ind = int((np.where(factorname == 'SOC'))[0]) 
gt = factors_v[:,esg_ind].reshape(1,-1)
Ctrs = [newmodel_1, newmodel_2, newmodel_3, newmodel_4, newmodel]
Ctrs_exp = ['const - ESG - mkt...rdme', 'const - ESG - mkt...dbnetis', 'const - ESG - mkt...EVar', 'const - ESG - mkt...eqnetis', 'const - ESG - mkt...fscore']

for i, c in enumerate(Ctrs):
    print(PriceRisk_OLS_2(Ri.T, gt, c).summary())
    print(Ctrs_exp[i])        
    show_coef_tstat_np(PriceRisk_OLS_2(Ri.T, gt, c))
    print('='*10)

# +
target_factor = 'GOV'

'''
one esg factor
'''
esg_ind = int((np.where(factorname == target_factor))[0]) 
gt = factors_v[:,esg_ind].reshape(1,-1)
Ctrs = [CAPM, FF3, C4, FF5, Qfactor, newmodel]
Ctrs_exp = ['const - ESG - mkt', 'const - ESG - mkt s h', 'const - ESG - mkt s h m', 'const - ESG - mkt s h r c', 'const - ESG - mkt s ia roe', 'const - ESG - mkt ...']

for i, c in enumerate(Ctrs):
    print(PriceRisk_OLS_2(Ri.T, gt, c).summary())
    print(Ctrs_exp[i])        
    show_coef_tstat_np(PriceRisk_OLS_2(Ri.T, gt, c))
    print('='*10)
    
'''
one esg factor: new models
'''
esg_ind = int((np.where(factorname == target_factor))[0]) 
gt = factors_v[:,esg_ind].reshape(1,-1)
Ctrs = [newmodel_1, newmodel_2, newmodel_3, newmodel_4, newmodel]
Ctrs_exp = ['const - ESG - mkt...rdme', 'const - ESG - mkt...dbnetis', 'const - ESG - mkt...EVar', 'const - ESG - mkt...eqnetis', 'const - ESG - mkt...fscore']

for i, c in enumerate(Ctrs):
    print(PriceRisk_OLS_2(Ri.T, gt, c).summary())
    print(Ctrs_exp[i])        
    show_coef_tstat_np(PriceRisk_OLS_2(Ri.T, gt, c))
    print('='*10)
# -

# # DS

# ## data setting

# import HXZ portfolio
HXZ_port = pd.read_csv('C:/Users/Jeongseok_Bang/OneDrive/바탕 화면/석사과정/FinLab/22 11 ESG risk factor (제안서)/code/HXZ_port.csv', index_col=0)
HXZ_port.index = pd.date_range('1967-01-31', '2021-12-31', freq='M')
HXZ_port *= 0.01

HXZ_port

# import JKP factors
allfactors = pd.read_csv('C:/Users/Jeongseok_Bang/OneDrive/바탕 화면/석사과정/FinLab/22 11 ESG risk factor (제안서)/code/df_factorzoo_JKP.csv', index_col=0)
allfactors.index.names = ['Date']
allfactors.index = pd.to_datetime(allfactors.index)

allfactors.shape

# import FF3 dataframe
FF5 = pd.read_csv('C:/Users/Jeongseok_Bang/OneDrive/바탕 화면/석사과정/FinLab/23 4 ESG controversies risk factor/data/FF5.csv', index_col=0)
FF5.index = pd.date_range('1963-07-31', '2023-02-28', freq='M')
FF5 *= 0.01

FF5.shape

# import Qfactor
Qfactor = pd.read_csv('C:/Users/Jeongseok_Bang/OneDrive/바탕 화면/석사과정/FinLab/23 4 ESG controversies risk factor/data/Qfactor.csv', index_col=0)
Qfactor.index = pd.date_range('1967-01-31', '2022-12-31', freq='M')
Qfactor *= 0.01

Qfactor.shape

FF5, Qfactor

# emerge JKP and RF in FF5
allfactors.insert(0, 'MktRf', FF5['Mkt-RF'])
allfactors.insert(0, 'RF', FF5['RF'])
allfactors = allfactors.join(FF5[['SMB','HML','RMW','CMA']], how='outer')
allfactors = allfactors.join(Qfactor[['R_ME','R_IA','R_ROE']], how='outer')

allfactors.shape

# print(FF5 data와 JKP data 유사성 비교
print(FF5.join(allfactors['market_equity']).corr()) # SMB
print(FF5.join(allfactors['be_me']).corr()) # HML
print(FF5.join(allfactors['at_be']).corr()) # HML

# ## ESG factor

esg_factors *= 0.01

# emerge allfactors and esg_factor
allfactors = allfactors.join(esg_factors, how='outer')

allfactors

# date cut
allfactors_periodcut = allfactors[(allfactors.index >= '2003-07-31') & (allfactors.index <= '2021-12-31')]

allfactors_periodcut

# Eliminate Rf (risk free return)

factors = allfactors_periodcut.copy()
date = factors.index
rf = factors['RF']
factors_no_rf = factors.iloc[:,1:]
factors = factors_no_rf.values # except date and Rf
L = date.shape[0]
P = factors.shape[1]
L,P

# +
HXZ_port = HXZ_port[(HXZ_port.index >= '2003-07-31') & (HXZ_port.index <= '2021-12-31')]

HXZ_port_ex = HXZ_port - rf.values.reshape((-1,1))
HXZ_port_ex
# -

# drop the first column (Rf)
factorname = allfactors.columns[1:]
factorname_full = allfactors.columns[1:]


factorname

# ## Factor zoo pre-analysis

allfactors_periodcut # RF remain

# 서로 상관성이 강한 factor 선별

threshold = 0.8

for i in range(allfactors_periodcut.shape[1]):
    # basis factor
    print('#'*100, '\n', allfactors_periodcut.columns[i])
    for j in range(i+1,allfactors_periodcut.shape[1]):
        factor_coef = allfactors_periodcut.iloc[:,i].corr(allfactors_periodcut.iloc[:,j])
        if abs(factor_coef) > threshold:
            print('=>', allfactors_periodcut.columns[j], round(factor_coef,2))

threshold = 0.5

no_f = allfactors_periodcut.shape[1]
for i in range(no_f-esg_factors.columns.shape[0],no_f):
    print('#'*100, '\n', allfactors_periodcut.columns[i])
    for j in range(allfactors_periodcut.shape[1]):
        factor_coef = allfactors_periodcut.iloc[:,i].corr(allfactors_periodcut.iloc[:,j])
        if abs(factor_coef) > threshold:
            print('=>', allfactors_periodcut.columns[j], round(factor_coef,2))

# 아래는 corr가 높은 factor를 제거하여 새롭게 만든 factor list

# threshold = 0.9
factor_sel= ['MktRf', 'age', 'aliq_at', 'aliq_mat', 'ami_126d', 'at_be', 'at_gr1', 'at_turnover', 'be_gr1a', 'be_me',
'beta_60m', 'beta_dimson_21d', 'capex_abn', 'capx_gr1', 'capx_gr2', 'cash_at', 'coa_gr1a', 'col_gr1a', 'cop_at',
'corr_1260d', 'coskew_21d', 'cowc_gr1a', 'dbnetis_at', 'debt_gr3', 'dgp_dsale', 'div12m_me', 'dolvol_var_126d',
'dsale_dinv', 'dsale_drec', 'dsale_dsga', 'earnings_variability', 'ebit_sale', 'emp_gr1', 'eqnetis_at', 'f_score',
'fcf_me', 'fnl_gr1a', 'gp_at', 'inv_gr1', 'inv_gr1a', 'iskew_capm_21d', 'iskew_ff3_21d', 'iskew_hxz4_21d',
'ival_me', 'kz_index', 'lti_gr1a', 'mispricing_mgmt', 'mispricing_perf', 'ncol_gr1a', 'netdebt_me', 'netis_at',
'nfna_gr1a', 'ni_ar1', 'ni_be', 'ni_inc8q', 'ni_ivol', 'ni_me', 'niq_at', 'niq_at_chg1', 'niq_su', 'noa_at', 'noa_gr1a',
'o_score', 'oaccruals_at', 'oaccruals_ni', 'ocf_at', 'ocf_at_chg1', 'ocf_me', 'ocfq_saleq_std', 'ope_bel1', 'pi_nix',
'ppeinv_gr1a', 'prc', 'prc_highprc_252d', 'qmj', 'qmj_growth', 'qmj_safety', 'rd5_at', 'rd_me', 'resff3_12_1',
'resff3_6_1', 'ret_12_1', 'ret_12_7', 'ret_1_0', 'ret_3_1', 'ret_60_12', 'ret_6_1', 'rmax5_rvol_21d', 'rskew_21d',
'rvol_21d', 'sale_bev', 'sale_emp_gr1', 'sale_gr1', 'sale_gr3', 'sale_me', 'saleq_su', 'seas_11_15an', 'seas_11_15na',
'seas_16_20an', 'seas_16_20na', 'seas_1_1an', 'seas_2_5an', 'seas_2_5na', 'seas_6_10an', 'seas_6_10na',
'sti_gr1a', 'taccruals_at', 'taccruals_ni', 'tangibility', 'tax_gr1a', 'z_score', 'zero_trades_252d', 'SMB', 'HML', 
'RMW', 'CMA', 'R_ME', 'R_IA', 'R_ROE'] + list(esg_factors.columns)

# threshold = 0.8
factor_sel= ['MktRf', 'age', 'at_gr1', 'at_turnover', 
             'beta_dimson_21d', 'capex_abn', 'capx_gr1', 'cop_at', 'corr_1260d', 
             'coskew_21d', 'cowc_gr1a', 'dbnetis_at', 'debt_gr3', 'dgp_dsale', 'dolvol_var_126d', 'dsale_dinv',
             'dsale_drec', 'dsale_dsga','ebit_sale', 'earnings_variability', 'eqnetis_at', 'f_score',
             'fnl_gr1a', 'gp_at', 'inv_gr1', 'iskew_capm_21d', 'iskew_ff3_21d', 'iskew_hxz4_21d',
             'kz_index', 'lti_gr1a', 
             'ni_ar1', 'ni_be', 'ni_inc8q', 'ni_me', 'niq_su', 'noa_at', 'noa_gr1a',
             'oaccruals_at', 'oaccruals_ni', 'ocf_at_chg1', 'ocfq_saleq_std', 'pi_nix', 'ppeinv_gr1a',
             'qmj', 'qmj_growth', 'rd5_at', 'rd_me', 'resff3_12_1', 'resff3_6_1', 'ret_12_1',
             'ret_12_7', 'ret_1_0', 'ret_3_1', 'ret_60_12', 'ret_6_1', 'rmax5_rvol_21d', 'rskew_21d',
             'sale_emp_gr1', 'saleq_su', 'seas_11_15an', 'seas_11_15na', 'seas_16_20an', 
             'seas_16_20na', 'seas_1_1an', 'seas_2_5na', 'seas_6_10an', 'seas_6_10na', 'sti_gr1a',
             'taccruals_ni', 'tangibility', 'tax_gr1a', 'SMB', 'HML', 'RMW', 'CMA', 'R_IA',
             'R_ROE'] + list(esg_factors.columns)

len(factor_sel) - len(list(esg_factors.columns))


# ## functions

# -	1st LASSO opt tune: 1.8636811970183956e-11
# -	2nd LASSO opt tune: 1.3832372648053784e-14

# +
def TSCV(Ri, gt, ht, lambda_, Kfld, Jrep, alpha=1, seednum=101):

#lambda_=lambda0
    p, T = ht.shape
    LL3 = lambda_.shape[0]

    cvm3 = np.empty((LL3, Kfld, Jrep))
    cvm33 = []
    
    # finding NaN row index (gt, ht 하나라도 포함되는 행)
    hg = np.append(ht, gt, axis=0)
    nomissing = ~ (np.isnan(hg)).any(axis=0)
    ht = (hg[:,nomissing])[:-gt.shape[0],:]
    ht = ht.T # n' x p
    gt = (hg[:,nomissing])[-gt.shape[0]:,:]
    gt = gt.T # n' x d
    
    for j in range(Jrep):    
        # cross-validation
        kf = KFold(n_splits=Kfld, shuffle=True, random_state=seednum+j)
        k =0
        for train_index, test_index in kf.split(ht):
            ht_train, ht_test = ht[train_index,:], ht[test_index,:]
            gt_train, gt_test = gt[train_index,:], gt[test_index,:]

            # LASSO estimation w/ fit_intercept=False
            _, coefs_lasso, _ = lasso_path(ht_train, gt_train, alphas=lambda_, max_iter=10_000) #10_000_000
            coefs_lasso = (np.array(coefs_lasso)).reshape((p, LL3)) # p x LL3
            
            # predicted gt
            gt_pred = ht_test @ coefs_lasso #(p x n').T x (p x LL3) = n' x LL3

            # gt_test.T를 LL3 회 가로 방향 누적 후 gt_pred를 빼준 값의 제곱 후 평균.T
            # test 값에는 nan이 포함됨
            cvm3[0:LL3,k,j] = np.nanmean((mb.repmat(gt_test,1,LL3) - gt_pred)**2,0).T # LL3x1
            k+=1

    cvm33 = cvm3.reshape((cvm3.shape[0], cvm3.shape[1]*Jrep))
    # Jrep =1 이므로 사실상 cvm33과 cvm3은 같음
    
    # 최소의 RSS를 갖는 penalty의 index만 추출
    cvm333 = np.mean(cvm33,1)
    l_sel3 = np.argmin(cvm333)

    # one-standard-error rule ?
    cv_sd3 = np.std(cvm33.T, axis=0)/np.sqrt(Kfld*Jrep)
    cvm33ub = cvm333[l_sel3] + cv_sd3[l_sel3] # upper bound
    #lambda는 정렬된 상태, 조건을 충족하는 마지막 하나의 index
    if (np.where(cvm333[:l_sel3] >= cvm33ub)[0]).shape[0] == 0: 
        l3_1se = l_sel3
    else:
        l3_1se = (np.where(cvm333[:l_sel3] >= cvm33ub)[0])[-1]

    # to reestimate the model with all data
    # refit the model
    model3 = Lasso(alpha=lambda_[l_sel3], max_iter=1_000_000)
    model3.fit(ht, gt) # nomissing: n x 1 

    # 0이 아닌 계수들의 index를 select
    sel3 = (np.where(model3.coef_ !=0))[0] # constant 제외, 
    sel3 = sel3.astype(int)

    model3_1 = Lasso(alpha=lambda_[l3_1se], max_iter=1_000_000)
    model3_1.fit(ht, gt)

    # 0이 아닌 계수들의 index를 select
    sel3_1se = (np.where(model3_1.coef_ !=0))[0] # constant 제외,
    sel3_1se = sel3_1se.astype(int)

    # TSCV['WANT']
    return dict({'sel3':sel3, 'lambda3':lambda_[l_sel3], 'sel3_1se':sel3_1se, 'lambda3_1se':lambda_[l3_1se]})


# -

def infer(Ri, gt, ht, sel1, sel2, sel3):

    n = Ri.shape[0]
    p = ht.shape[0]
    d = gt.shape[0]

    ### pd 툴 사용 with dropna
    tot1 = np.append(gt.T, Ri.T, axis=1)
    tmp1_pd = pd.DataFrame(tot1).dropna().cov() # 1x498 750x498 -> 751x751
    cov_g = tmp1_pd.values[d:,:d] # gt와 Ri 각 벡터 사이의 cov 값들로 이루어진 vector,gt의 var는 (0,0)이므로 제외

    tot2 = np.append(ht.T, Ri.T, axis=1)
    tmp2_pd = pd.DataFrame(tot2).dropna().cov() # 135x498 750x498 -> 885 x 885
    cov_h = tmp2_pd.values[p:,:p] # ht와 Ri 각 벡터 사이의 cov 값들로 이루어진 vector,gt의 var는 (0,0)이므로 제외

    ER  = np.mean(Ri, axis=1) 
    ER = ER.values 

    select = np.union1d(sel1,sel2)

    # projection matrix of one-vector
    one_vec = np.ones((n,1))
    M0 = np.eye(n) - one_vec@ np.linalg.inv(one_vec.T@one_vec)@one_vec.T

    # eliminate rows with missing values
    hg = np.append(ht, gt, axis=0)
    nomissing = np.where((np.isnan(hg)).any(axis=0) == False)[0]
    '''caution: TSCV에서도 nomissing 변수 사용'''

    Lnm = nomissing.shape[0]

    cov_h_sel = cov_h[:, select.astype(int)] # 선택된 변수만 포함한 cov_h

    # constant를 제외한 coef 계산 (partition regression)
    X = np.append(cov_g, cov_h_sel, axis=1)
    lambda_full = np.linalg.inv(X.T@M0@X) @ (X.T@M0@ER)
    lambdag = lambda_full[0:d]

    ht_sel = (ht[sel3, :])[:,nomissing] # ht select (by sel3) & nomiss
    gt_ = gt[:, nomissing]         # gt nomiss

    # For double selection inference: AVAR
    zhat = np.empty((d,Lnm))

    for i in range(d):
        M_mdl = np.eye(Lnm) - ht_sel.T @ np.linalg.inv(ht_sel@ht_sel.T) @ ht_sel
        zhat[i,:] = (M_mdl @ gt_.T).reshape((M_mdl.shape[0],)) # (I-P)g, residual?

    Sigmazhat = (zhat @ zhat.T)/Lnm # RSS, scalar

    temp2 = np.zeros((d,d))

    ht_select = ht[select.astype(int),:] # ht select by sel1 & 2

    for ii, l in enumerate(nomissing):
        mt = 1 - lambda_full.T @ (np.append(gt[0:d,l], ht_select[:,l], axis=0)).reshape((-1,1)) 
        temp2 += (mt**2) @ (np.linalg.inv(Sigmazhat) @ zhat[:,ii].reshape(-1,1) @ zhat[:,ii].reshape(1,-1) @ np.linalg.inv(Sigmazhat))

    avar_lambdag = np.diag(temp2)/Lnm
    se = np.sqrt(avar_lambdag/Lnm)

    # ht select (by select) & nomiss
    ht_select_no = (ht[select.astype(int), :])[:,nomissing]

    # scaled lambda for DS
    '''it represents the estimated average excess return in
    basis points per month of a portfolio with unit univariate beta with respect to that factor.
    각 factor들의 scale(평균)이 다르기 때문에 비교를 위해 평균에서 deviated 정도를 곱해줌    
    reverse-normalize?'''    
    vt = np.append(gt_, ht_select_no, axis=0)
    V_bar = vt - (np.mean(vt, axis=1)).reshape((-1,1)) @ np.ones((1,Lnm)) # 원래값 - 평균값, n x n
    var_v = (V_bar @ V_bar.T)/Lnm # deviated from mean
    gamma = np.multiply(np.diag(var_v), lambda_full) 

    return dict({ 'lambdag':lambdag, 'se':se, 'gamma':gamma })


def DS(Ri, gt, ht, tune_1st=1e-40, tune_2nd=1e-40, sel3=np.array([]), seednum=101, mode='given'):
    '''
    Parameters
        mode: if 'given', the hyperparameters (tune_1st and 2nd) are the given value; 
                if 'choose', the hyperparameters are automatically chosen by cross validation
    '''
    # data information
    n = Ri.shape[0] 
    p = ht.shape[0] 
    d = gt.shape[0] # 1

    '''일부 nan 데이터때문에 np.cov 바로 사용불가
    df.dropna을 통해 omitrows와 동일한 기능 수행'''
    tot1 = np.append(gt.T, Ri.T, axis=1)
    tmp1_pd = pd.DataFrame(tot1).dropna().cov() # 1x498 750x498 -> 751x751
    cov_g = tmp1_pd.values[d:,:d] # gt와 Ri 각 벡터 사이의 cov 값들로 이루어진 vector,gt의 var는 (0,0)이므로 제외
    tot2 = np.append(ht.T, Ri.T, axis=1)
    tmp2_pd = pd.DataFrame(tot2).dropna().cov() # 135x498 750x498 -> 885 x 885
    cov_h = tmp2_pd.values[p:,:p] # ht와 Ri 각 벡터 사이의 cov 값들로 이루어진 vector,gt의 var는 (0,0)이므로 제외
    # cov_h[:,i]는 각 factor별 R1~Rn까지의 cov값으로 이루어진 벡터
    
    ER  = np.mean(Ri, axis=1) # ER은 portfolio 평균으로 계산
    ER = ER.values
    
    # see Bryzgalova (2015)
    beta = np.empty((n,p))
    '''ddof: numpy는 0이 default, matlab은 1이 default'''
    for i in range(p):
        beta[:,i] = cov_h[:,i]/np.nanvar(ht[i,:], ddof=1) # 각 factor의 Ri에 대한 beta값 
    penalty = np.nanmean(beta**2, axis=0) 
    penalty = penalty/np.nanmean(penalty) # normalize
    
    lambda0 = np.exp(np.linspace(0,-35,35)) # penalty parameter array
    
    '''1st selection'''    
    if mode == 'choose':
        # Use cross-validation to find the optimal tuning parameter: wtf lassocv.alpha_
        lassocv = LassoCV(alphas=lambda0, cv=10, max_iter=1_000_000, random_state=seednum)
        lassocv.fit(cov_h@(np.diag(penalty)), ER)
        print('1st LASSO opt tune:', lassocv.alpha_)
        # fit LASSO by the optimal tuning paramter
        model1 = Lasso(alpha=lassocv.alpha_, max_iter=1_000_000, tol=1e-7)
    elif mode == 'given':
        model1 = Lasso(alpha=tune_1st, max_iter=1_000_000, tol=1e-7)
   
    model1.fit(cov_h@(np.diag(penalty)), ER) # cov_h@(np.diag(penalty)) = [h1 h2 ...] @ diag([p1 p2 ...]) = [p1h1 p2h1 ...]
    model1_est = np.append(model1.intercept_, model1.coef_) 
    
    # select the indices of non-zero coef.
    sel1 = (np.where(model1_est[1:p+2]!=0))[0] # except constant
    
    # get RSS of model 1
    y_hat_1 = np.append(np.ones((n,1)), cov_h@(np.diag(penalty)), axis=1)
    beta_hat_1 = model1_est.reshape((-1,1))
    err1 = np.mean( (ER - y_hat_1@beta_hat_1)**2 )
    
    '''2nd selection'''
    sel2 = np.array([])
    err2 = np.empty((d,1))
    for i in range(d): # d=1
        if mode == 'choose':
            # Use cross-validation to find the optimal tuning parameter:  wtf lassocv.alpha_
            lassocv = LassoCV(alphas=lambda0, cv=10, max_iter=1_000_000, random_state=seednum)
            lassocv.fit(cov_h@(np.diag(penalty)), cov_g[:,i])
            print('2nd LASSO opt tune:', lassocv.alpha_)
            # fit LASSO by the optimal tuning paramter
            model2 = Lasso(alpha=lassocv.alpha_, max_iter=1_000_000, tol=1e-7)
        elif mode == 'given':
            model2 = Lasso(alpha=tune_2nd, max_iter=1_000_000, tol=1e-7)
        
        model2.fit(cov_h@(np.diag(penalty)), cov_g[:,i])
        model2_est = np.append(model2.intercept_, model2.coef_)
        # select the indices of non-zero coef.
        sel2_temp = (np.where(model2_est[1:p+2]!=0))[0] # except constant
        # append sel2 arraies over loop
        sel2 = np.concatenate((sel2, sel2_temp))

        # get RSS of model 2
        y_hat_2 = np.append(np.ones((n,1)), cov_h@(np.diag(penalty)), axis=1)
        beta_hat_2 = model2_est.reshape((-1,1))
        err1 = np.mean( (cov_g[:,i] - y_hat_2@beta_hat_2)**2 )
    sel2 = np.unique(sel2).T
    
    '''3rd selection'''
    if sel3.shape[0] == 0:
        sel3 = np.array([])
        for i in range(d):
            TSCVout = TSCV(Ri, gt[i,:].reshape(1,-1) , ht, lambda0, 10, 1)
            sel3 = np.concatenate((sel3, TSCVout['sel3_1se']))        
        sel3 = np.unique(sel3)
        sel3 = sel3.astype(int)
    
    '''post selection estimation and inference'''
    dsout = infer(Ri, gt, ht, sel1, sel2, sel3)
    ssout = infer(Ri, gt, ht, sel1, np.array([]), sel3)
    
    '''return'''
    return_dict = dict({
        # return for DS
        'lambdag_ds':dsout['lambdag'],
        'se_ds':dsout['se'],
        'gamma_ds':dsout['gamma'],
        # return for SS
        'lambdag_ss':ssout['lambdag'],
        'se_ss':ssout['se'],
        'gamma_ss':ssout['gamma'],
        # selection return
        'sel1':sel1,
        'sel2':sel2,
        'sel3':sel3,
        'select':np.union1d(sel1,sel2),
        'err1':err1,
        'err2':err2
    })
    return return_dict


def PriceRisk_OLS(Ri, gt, ht):
    n, T = Ri.shape
    d = gt.shape[0]
    p = ht.shape[0]

    np_Ri = Ri.values

    cov_h = np.empty((n,p))
    for nn in range(n):
        Rh = np.append((np_Ri[nn,:]).reshape(-1,1), ht.T, axis=1) # T x 1 + T x p 
        Rh_cov = pd.DataFrame(Rh).dropna().cov()
        cov_h[nn, :] = Rh_cov.values[0, 1:]

    cov_g = np.empty((n,d))
    for nn in range(n):    
        Rg = np.append((np_Ri[nn,:]).reshape(-1,1), gt.T, axis=1) # T x 1 + T x d
        Rg_cov = pd.DataFrame(Rg).dropna().cov()
        cov_g[nn, :] = Rg_cov.values[0, 1:]

    ER  = np.mean(Ri, axis=1)
    ER = ER.values
    
    one_vec = np.ones((n,1))
    M0 = np.eye(n) - one_vec@ np.linalg.inv(one_vec.T@one_vec)@one_vec.T

    # For no selection OLS
    X = np.append(cov_g, cov_h, axis=1)
    X_zero = np.append(np.ones((n,1)), X, axis=1)
    lambda_full_zero = np.linalg.inv(X_zero.T@X_zero) @ (X_zero.T @ ER)
    lambda_full = np.linalg.inv(X.T@M0@X) @ (X.T@M0@ER)
    lambdag_ols = lambda_full[0:d]

    # eliminate rows with missing values
    hg = np.append(ht, gt, axis=0)
    nomissing = np.where((np.isnan(hg)).any(axis=0) == False)[0]
    Lnm = nomissing.shape[0]

    # calculate avar_ols
    zhat = np.empty((d,Lnm))

    for i in range(d):
        M_mdl = np.eye(Lnm) - ht[:,nomissing].T @ np.linalg.inv(ht[:,nomissing]@ht[:,nomissing].T) @ ht[:,nomissing]
        zhat[i,:] = (M_mdl @ gt[i,nomissing].T).reshape((-1,))

    Sigmazhat = (zhat @ zhat.T)/Lnm

    temp3 = np.zeros((d,d))

    for ii, l in enumerate(nomissing):
        mt = 1 - lambda_full.T @ (np.append(gt[:,l], ht[:,l], axis=0)).reshape((-1,1)) 
        temp3 += (mt**2) @ (np.linalg.inv(Sigmazhat) @ zhat[:,ii].reshape(-1,1) @ zhat[:,ii].reshape(1,-1) @ np.linalg.inv(Sigmazhat))

    avar_lambdag3 = np.diag(temp3)/Lnm
    se3 = np.sqrt(avar_lambdag3/Lnm)

    # scaled lambda for DS
    vt = np.append(gt[:,nomissing], ht[:,nomissing], axis=0)
    V_bar = vt - (np.mean(vt, axis=1)).reshape((-1,1)) @ np.ones((1,Lnm))
    var_v = (V_bar @ V_bar.T)/Lnm
    lambda_ols = np.multiply(np.diag(var_v), lambda_full) 
    
    return_dict = dict({
        'lambdag_ols': lambdag_ols,
        'se_ols': se3,
        'lambda_ols': lambda_ols,
        'lambda_ols_zero': lambda_full_zero
    })
    
    return return_dict


# ## main result

Ri = HXZ_port_ex

# +
# CAPM
mkt_ind = int((np.where(factorname == 'MktRf'))[0])
CAPM = factors[:,[mkt_ind]].T

# Fama French 3 factor model
# smb_ind = int((np.where(factorname == 'market_equity'))[0])  #  at_be
# hml_ind = int((np.where(factorname == 'be_me'))[0]) #  at_me
smb_ind = int((np.where(factorname == 'SMB'))[0]) 
hml_ind = int((np.where(factorname == 'HML'))[0]) 
FF3 = factors[:,[mkt_ind,smb_ind,hml_ind]].T

# Carharr 4 factor model
mom_ind = int((np.where(factorname == 'ret_12_1'))[0])
C4 = factors[:,[mkt_ind, smb_ind, hml_ind, mom_ind]].T

# Fama French 5 factor model
rmw_ind = int((np.where(factorname == 'RMW'))[0]) 
cma_ind = int((np.where(factorname == 'CMA'))[0]) 
FF5 = factors[:,[mkt_ind, smb_ind, hml_ind, rmw_ind, cma_ind]].T

# HXZ q-factor model
me_ind = int((np.where(factorname == 'R_ME'))[0]) 
ia_ind = int((np.where(factorname == 'R_IA'))[0]) 
roe_ind = int((np.where(factorname == 'R_ROE'))[0]) 
Qfactor = factors[:,[mkt_ind, me_ind, ia_ind, roe_ind]].T

# +
# get column no. of ESG factors
esg_ind = [factors_no_rf.columns.get_loc(e) for e in esg_factors.columns]

# test factors
TestFactor = factors[:, esg_ind]

# choose control factors
ctr_ind = [factors_no_rf.columns.get_loc(c) for c in factors_no_rf.columns if c not in esg_factors.columns]
ControlFactor = factors[:, ctr_ind]
# -

# CHECK
# Is ESG factor returns from factors_no_rf (came from allfactors) equal to TestFactor (array) ? 
print(factors_no_rf.columns[esg_ind])
print((factors_no_rf.iloc[:,esg_ind].values == TestFactor).all())

# ### factor selection - mul.corr.

factors_no_rf[factor_sel].columns

date = allfactors_periodcut.index
rf = allfactors_periodcut['RF']
factors_no_rf_sel = allfactors_periodcut.loc[:,factor_sel]
factors_sel = factors_no_rf_sel.values # except date and Rf

# +
# get column no. of ESG factors
esg_ind = [factors_no_rf_sel.columns.get_loc(e) for e in esg_factors.columns]

# test factors
TestFactor = factors_sel[:, esg_ind]

# choose control factors
ctr_ind = [factors_no_rf_sel.columns.get_loc(c) for c in factors_no_rf_sel.columns if c not in esg_factors.columns]
ControlFactor = factors_sel[:, ctr_ind]
# -

# ### Choose hyperparameter version

# +
# result_table = np.empty((len(esg_ind), 10))
for j in range(len(esg_ind)):
    gt = TestFactor[:,j].reshape(1,-1)
    ht = ControlFactor.T
    
    # Show esg factor name
    print('#'*100)
    print(f'order:{j} - {esg_factors.columns[j]}')
    
    # Double-Selection results 
    model_ds = DS(Ri.T, gt, ht, mode='choose')
    tstat_ds = model_ds['lambdag_ds']/model_ds['se_ds']
    lambda_ds = model_ds['gamma_ds'][0]
    print('sel1:', model_ds['sel1'], '\n sel2:', model_ds['sel2'])
    
#     # Single-Selection results
#     model_ss = DS(Ri.T, gt, ht, tune_2nd=0.0, mode=)
#     tstat_ss = model_ds['lambdag_ss']/model_ds['se_ss']
#     lambda_ss = model_ds['gamma_ss'][0]

    # controlling everything, no selection, OLS
    model_ols = PriceRisk_OLS(Ri.T, gt, ht)
    tstat_ols = model_ols['lambdag_ols']/model_ols['se_ols']
    lambda_ols = model_ols['lambda_ols'][0]

    # only control CAPM by OLS
    model_CAPM = PriceRisk_OLS(Ri.T, gt, CAPM)
    tstat_CAPM = model_CAPM['lambdag_ols']/model_CAPM['se_ols']
    lambda_CAPM = model_CAPM['lambda_ols'][0]
    lambda_all_CAPM = model_CAPM['lambda_ols_zero']

    # only control FF3 by OLS
    model_FF3 = PriceRisk_OLS(Ri.T, gt, FF3)
    tstat_FF3 = model_FF3['lambdag_ols']/model_FF3['se_ols']
    lambda_FF3 = model_FF3['lambda_ols'][0]
    lambda_all_FF3 = model_FF3['lambda_ols_zero']

    # only control Carhart4 by OLS
    model_C4 = PriceRisk_OLS(Ri.T, gt, C4)
    tstat_C4 = model_C4['lambdag_ols']/model_C4['se_ols']
    lambda_C4 = model_C4['lambda_ols'][0]
    lambda_all_C4 = model_C4['lambda_ols_zero']

    # only control FF5 by OLS
    model_FF5 = PriceRisk_OLS(Ri.T, gt, FF5)
    tstat_FF5 = model_FF5['lambdag_ols']/model_FF5['se_ols']
    lambda_FF5 = model_FF5['lambda_ols'][0]
    lambda_all_FF5 = model_FF5['lambda_ols_zero']

    # only control q-factor by OLS
    model_Qfactor = PriceRisk_OLS(Ri.T, gt, Qfactor)
    tstat_Qfactor = model_Qfactor['lambdag_ols']/model_Qfactor['se_ols']
    lambda_Qfactor = model_Qfactor['lambda_ols'][0]
    lambda_all_Qfactor = model_Qfactor['lambda_ols_zero']

    print(f'DS\t{lambda_ds*10000:10.2f} bp, \t\t{tstat_ds[0,]:10.2f}')
    print(f'OLS\t{lambda_ols*10000:10.2f} bp, \t\t{tstat_ols[0,]:10.2f}')
    print(f'CAPM\t{lambda_CAPM*10000:10.2f} bp, \t\t{tstat_CAPM[0,]:10.2f}')
    print(f'FF3\t{lambda_FF3*10000:10.2f} bp, \t\t{tstat_FF3[0,]:10.2f}')
    print(f'C4\t{lambda_C4*10000:10.2f} bp, \t\t{tstat_C4[0,]:10.2f}')
    print(f'FF5\t{lambda_FF5*10000:10.2f} bp, \t\t{tstat_FF5[0,]:10.2f}')
    print(f'Q\t{lambda_Qfactor*10000:10.2f} bp, \t\t{tstat_Qfactor[0,]:10.2f}')
# -

# ### given hyperparameter version

# +
# result_table = np.empty((len(esg_ind), 10))

tune_1st_lst = [1e-8]*6
tune_2nd_lst = [1e-8]*6

for j in range(len(esg_ind)):
    gt = TestFactor[:,j].reshape(1,-1)
    ht = ControlFactor.T
    
    # Show esg factor name
    print('#'*100)
    print(f'order:{j} - {esg_factors.columns[j]}')
    
    # Double-Selection results 
    model_ds = DS(Ri.T, gt, ht, tune_1st=tune_1st_lst[j], tune_2nd=tune_2nd_lst[j], mode='given')
    tstat_ds = model_ds['lambdag_ds']/model_ds['se_ds']
    lambda_ds = model_ds['gamma_ds'][0]
    print('sel1:', model_ds['sel1'], '\n sel2:', model_ds['sel2'])
    
#     # Single-Selection results
#     model_ss = DS(Ri.T, gt, ht, tune_1st=tune_1st_lst[j], tune_2nd=0.0, mode='given')
#     tstat_ss = model_ds['lambdag_ss']/model_ds['se_ss']
#     lambda_ss = model_ds['gamma_ss'][0]

    # controlling everything, no selection, OLS
    model_ols = PriceRisk_OLS(Ri.T, gt, ht)
    tstat_ols = model_ols['lambdag_ols']/model_ols['se_ols']
    lambda_ols = model_ols['lambda_ols'][0]

    # only control CAPM by OLS
    model_CAPM = PriceRisk_OLS(Ri.T, gt, CAPM)
    tstat_CAPM = model_CAPM['lambdag_ols']/model_CAPM['se_ols']
    lambda_CAPM = model_CAPM['lambda_ols'][0]
    lambda_all_CAPM = model_CAPM['lambda_ols_zero']

    # only control FF3 by OLS
    model_FF3 = PriceRisk_OLS(Ri.T, gt, FF3)
    tstat_FF3 = model_FF3['lambdag_ols']/model_FF3['se_ols']
    lambda_FF3 = model_FF3['lambda_ols'][0]
    lambda_all_FF3 = model_FF3['lambda_ols_zero']

    # only control Carhart4 by OLS
    model_C4 = PriceRisk_OLS(Ri.T, gt, C4)
    tstat_C4 = model_C4['lambdag_ols']/model_C4['se_ols']
    lambda_C4 = model_C4['lambda_ols'][0]
    lambda_all_C4 = model_C4['lambda_ols_zero']

    # only control FF5 by OLS
    model_FF5 = PriceRisk_OLS(Ri.T, gt, FF5)
    tstat_FF5 = model_FF5['lambdag_ols']/model_FF5['se_ols']
    lambda_FF5 = model_FF5['lambda_ols'][0]
    lambda_all_FF5 = model_FF5['lambda_ols_zero']

    # only control q-factor by OLS
    model_Qfactor = PriceRisk_OLS(Ri.T, gt, Qfactor)
    tstat_Qfactor = model_Qfactor['lambdag_ols']/model_Qfactor['se_ols']
    lambda_Qfactor = model_Qfactor['lambda_ols'][0]
    lambda_all_Qfactor = model_Qfactor['lambda_ols_zero']

    print(f'DS\t{lambda_ds*10000:10.2f} bp, \t\t{tstat_ds[0,]:10.2f}')
    print(f'OLS\t{lambda_ols*10000:10.2f} bp, \t\t{tstat_ols[0,]:10.2f}')
    print(f'CAPM\t{lambda_CAPM*10000:10.2f} bp, \t\t{tstat_CAPM[0,]:10.2f}')
    print(f'FF3\t{lambda_FF3*10000:10.2f} bp, \t\t{tstat_FF3[0,]:10.2f}')
    print(f'C4\t{lambda_C4*10000:10.2f} bp, \t\t{tstat_C4[0,]:10.2f}')
    print(f'FF5\t{lambda_FF5*10000:10.2f} bp, \t\t{tstat_FF5[0,]:10.2f}')
    print(f'Q\t{lambda_Qfactor*10000:10.2f} bp, \t\t{tstat_Qfactor[0,]:10.2f}')
# -

# ### heatmap

# range of penalty
lambda1 = 10.0**np.arange(-12, -7, 1)
lambda2 = 10.0**np.arange(-12, -7, 1)

# NaN 3D array
temp = 1
result_heatmap = np.empty((lambda1.shape[0],lambda2.shape[0], TestFactor.shape[1]))
result_heatmap[:,:,:] = np.nan

for j in range(TestFactor.shape[1]):
    gt = TestFactor[:,j].reshape(1,-1)
    ht = ControlFactor.T
    print(f'order:{j} - {esg_factors.columns[j]}') # show the testfactor which is being analyzed

    for l1_ind, l1 in enumerate(lambda1):
        for l2_ind, l2 in enumerate(lambda2):
            # calculate tstat_ds
            tstat_ds = np.nan # reset the var.
            model_ds = DS(Ri.T, gt, ht, tune_1st=l1, tune_2nd=l2, mode='given')
            tstat_ds = model_ds['lambdag_ds']/model_ds['se_ds']

            # insert into the 3D array
            result_heatmap[l1_ind,l2_ind,j] = tstat_ds

            # show the iteration state
            l1_per = l1_ind/lambda1.shape[0]
            l2_per = l2_ind/lambda2.shape[0]
            sys.stdout.write("\r")
            sys.stdout.write("l1 {:5.2f}, l2 {:5.2f}, tstat {:5.2f}".format(np.log10(l1), np.log10(l2), result_heatmap[l1_ind,l2_ind,j]))
            sys.stdout.flush()

    print('\n')

result_heatmap_flat = np.empty((lambda1.shape[0] * TestFactor.shape[1], lambda2.shape[0]))
result_heatmap_flat[:,:] = np.nan
for j in range(result_heatmap.shape[2]):
    result_heatmap_flat[lambda1.shape[0]*j:lambda1.shape[0]*(j+1),:] = result_heatmap[:,:,j]

np.save('result_heatmap_ESG_pillar_US_flat',result_heatmap_flat)

result = pd.DataFrame(result_heatmap_flat.T)

b = int(result.shape[1]/TestFactor.shape[1])
for j in range(TestFactor.shape[1]):
    result_temp = result.iloc[:,b*j:b*(j+1)]
    result_temp.index = np.round(np.log10(lambda2), 3) 
    result_temp.columns = np.round(np.log10(lambda1), 3)
    
    print(esg_factors.columns[j])
    sns.heatmap( result_temp, annot=True, cmap='vlag', cbar=True, vmin=-3, vmax=3) # 색방향 반대로 하려면 _r 붙이기  viridis
    plt.xlabel('penalty for 1st')
    plt.ylabel('penalty for 2nd')
    plt.show()

# ## PCA

# ### factor selection - mul.corr.

del ControlFactor

date = allfactors_periodcut.index
rf = allfactors_periodcut['RF']

factors_no_rf = allfactors_periodcut.iloc[:,1:]
factors = factors_no_rf.values # except date and Rf

# +
# get column no. of ESG factors
esg_ind = [factors_no_rf.columns.get_loc(e) for e in esg_factors.columns]

# test factors
TestFactor = factors[:, esg_ind]

# choose control factors
ctr_ind = [factors_no_rf.columns.get_loc(c) for c in factors_no_rf.columns if c not in esg_factors.columns]
ControlFactor = factors[:, ctr_ind]
# -

ht_pca = PCA().fit_transform(ControlFactor).T
ht_pca.shape

# ### choose

# +
# result_table = np.empty((len(esg_ind), 10))
for j in range(len(esg_ind)):
    gt = TestFactor[:,j].reshape(1,-1)
    
    # Show esg factor name
    print('#'*100)
    print(f'order:{j} - {esg_factors.columns[j]}')
    
    # Double-Selection results 
    model_ds = DS(Ri.T, gt, ht_pca, mode='choose')
    tstat_ds = model_ds['lambdag_ds']/model_ds['se_ds']
    lambda_ds = model_ds['gamma_ds'][0]
    print('sel1:', model_ds['sel1'], '\n sel2:', model_ds['sel2'])
    
#     # Single-Selection results
#     model_ss = DS(Ri.T, gt, ht_pca.T, tune_2nd=0.0, mode=)
#     tstat_ss = model_ds['lambdag_ss']/model_ds['se_ss']
#     lambda_ss = model_ds['gamma_ss'][0]

    # controlling everything, no selection, OLS
    model_ols = PriceRisk_OLS(Ri.T, gt, ht_pca)
    tstat_ols = model_ols['lambdag_ols']/model_ols['se_ols']
    lambda_ols = model_ols['lambda_ols'][0]

    # only control CAPM by OLS
    model_CAPM = PriceRisk_OLS(Ri.T, gt, CAPM)
    tstat_CAPM = model_CAPM['lambdag_ols']/model_CAPM['se_ols']
    lambda_CAPM = model_CAPM['lambda_ols'][0]
    lambda_all_CAPM = model_CAPM['lambda_ols_zero']

    # only control FF3 by OLS
    model_FF3 = PriceRisk_OLS(Ri.T, gt, FF3)
    tstat_FF3 = model_FF3['lambdag_ols']/model_FF3['se_ols']
    lambda_FF3 = model_FF3['lambda_ols'][0]
    lambda_all_FF3 = model_FF3['lambda_ols_zero']

    # only control Carhart4 by OLS
    model_C4 = PriceRisk_OLS(Ri.T, gt, C4)
    tstat_C4 = model_C4['lambdag_ols']/model_C4['se_ols']
    lambda_C4 = model_C4['lambda_ols'][0]
    lambda_all_C4 = model_C4['lambda_ols_zero']

    # only control FF5 by OLS
    model_FF5 = PriceRisk_OLS(Ri.T, gt, FF5)
    tstat_FF5 = model_FF5['lambdag_ols']/model_FF5['se_ols']
    lambda_FF5 = model_FF5['lambda_ols'][0]
    lambda_all_FF5 = model_FF5['lambda_ols_zero']

    # only control q-factor by OLS
    model_Qfactor = PriceRisk_OLS(Ri.T, gt, Qfactor)
    tstat_Qfactor = model_Qfactor['lambdag_ols']/model_Qfactor['se_ols']
    lambda_Qfactor = model_Qfactor['lambda_ols'][0]
    lambda_all_Qfactor = model_Qfactor['lambda_ols_zero']

    dof = 1700
    
    print(f'DS\t{lambda_ds*10000:10.2f} bp, \t\t{tstat_ds[0,]:10.2f}, \t{2*(t.cdf(-abs(tstat_ds[0,]), dof)):.4f}') 
    print(f'OLS\t{lambda_ols*10000:10.2f} bp, \t\t{tstat_ols[0,]:10.2f}, \t{2*(t.cdf(-abs(tstat_ols[0,]), dof)):.4f}')
    print(f'CAPM\t{lambda_CAPM*10000:10.2f} bp, \t\t{tstat_CAPM[0,]:10.2f}, \t{2*(t.cdf(-abs(tstat_CAPM[0,]), dof)):.4f}')
    print(f'FF3\t{lambda_FF3*10000:10.2f} bp, \t\t{tstat_FF3[0,]:10.2f}, \t{2*(t.cdf(-abs(tstat_FF3[0,]), dof)):.4f}')
    print(f'C4\t{lambda_C4*10000:10.2f} bp, \t\t{tstat_C4[0,]:10.2f}, \t{2*(t.cdf(-abs(tstat_C4[0,]), dof)):.4f}')
    print(f'FF5\t{lambda_FF5*10000:10.2f} bp, \t\t{tstat_FF5[0,]:10.2f}, \t{2*(t.cdf(-abs(tstat_FF5[0,]), dof)):.4f}')
    print(f'Q\t{lambda_Qfactor*10000:10.2f} bp, \t\t{tstat_Qfactor[0,]:10.2f}, \t{2*(t.cdf(-abs(tstat_Qfactor[0,]), dof)):.4f}')
    
# -

# ### given

# +
# result_table = np.empty((len(esg_ind), 10))

tune_1st_lst = [1e-9]*10
tune_2nd_lst = [1e-10]*10

# store t stat 
tstat_mat = np.empty((len(esg_ind),7))

for j in range(len(esg_ind)):
    gt = TestFactor[:,j].reshape(1,-1)
    
    # Show esg factor name
    print('#'*100)
    print(f'order:{j} - {esg_factors.columns[j]}')
    
    # Double-Selection results 
    model_ds = DS(Ri.T, gt, ht_pca, tune_1st=tune_1st_lst[j], tune_2nd=tune_2nd_lst[j], mode='given')
    tstat_ds = model_ds['lambdag_ds']/model_ds['se_ds']
    lambda_ds = model_ds['gamma_ds'][0]
    print('sel1:', model_ds['sel1'], '\n sel2:', model_ds['sel2'])
    
#     # Single-Selection results
#     model_ss = DS(Ri.T, gt, ht_pca, tune_1st=tune_1st_lst[j], tune_2nd=0.0, mode='given')
#     tstat_ss = model_ds['lambdag_ss']/model_ds['se_ss']
#     lambda_ss = model_ds['gamma_ss'][0]

    # controlling everything, no selection, OLS
    model_ols = PriceRisk_OLS(Ri.T, gt, ht_pca)
    tstat_ols = model_ols['lambdag_ols']/model_ols['se_ols']
    lambda_ols = model_ols['lambda_ols'][0]

    # only control CAPM by OLS
    model_CAPM = PriceRisk_OLS(Ri.T, gt, CAPM)
    tstat_CAPM = model_CAPM['lambdag_ols']/model_CAPM['se_ols']
    lambda_CAPM = model_CAPM['lambda_ols'][0]
    lambda_all_CAPM = model_CAPM['lambda_ols_zero']

    # only control FF3 by OLS
    model_FF3 = PriceRisk_OLS(Ri.T, gt, FF3)
    tstat_FF3 = model_FF3['lambdag_ols']/model_FF3['se_ols']
    lambda_FF3 = model_FF3['lambda_ols'][0]
    lambda_all_FF3 = model_FF3['lambda_ols_zero']

    # only control Carhart4 by OLS
    model_C4 = PriceRisk_OLS(Ri.T, gt, C4)
    tstat_C4 = model_C4['lambdag_ols']/model_C4['se_ols']
    lambda_C4 = model_C4['lambda_ols'][0]
    lambda_all_C4 = model_C4['lambda_ols_zero']

    # only control FF5 by OLS
    model_FF5 = PriceRisk_OLS(Ri.T, gt, FF5)
    tstat_FF5 = model_FF5['lambdag_ols']/model_FF5['se_ols']
    lambda_FF5 = model_FF5['lambda_ols'][0]
    lambda_all_FF5 = model_FF5['lambda_ols_zero']

    # only control q-factor by OLS
    model_Qfactor = PriceRisk_OLS(Ri.T, gt, Qfactor)
    tstat_Qfactor = model_Qfactor['lambdag_ols']/model_Qfactor['se_ols']
    lambda_Qfactor = model_Qfactor['lambda_ols'][0]
    lambda_all_Qfactor = model_Qfactor['lambda_ols_zero']

    dof = 1700
    
    print(f'DS\t{lambda_ds*10000:10.2f} bp, \t\t{tstat_ds[0,]:10.2f}, \t{2*(t.cdf(-abs(tstat_ds[0,]), dof)):.4f}') 
    print(f'OLS\t{lambda_ols*10000:10.2f} bp, \t\t{tstat_ols[0,]:10.2f}, \t{2*(t.cdf(-abs(tstat_ols[0,]), dof)):.4f}')
    print(f'CAPM\t{lambda_CAPM*10000:10.2f} bp, \t\t{tstat_CAPM[0,]:10.2f}, \t{2*(t.cdf(-abs(tstat_CAPM[0,]), dof)):.4f}')
    print(f'FF3\t{lambda_FF3*10000:10.2f} bp, \t\t{tstat_FF3[0,]:10.2f}, \t{2*(t.cdf(-abs(tstat_FF3[0,]), dof)):.4f}')
    print(f'C4\t{lambda_C4*10000:10.2f} bp, \t\t{tstat_C4[0,]:10.2f}, \t{2*(t.cdf(-abs(tstat_C4[0,]), dof)):.4f}')
    print(f'FF5\t{lambda_FF5*10000:10.2f} bp, \t\t{tstat_FF5[0,]:10.2f}, \t{2*(t.cdf(-abs(tstat_FF5[0,]), dof)):.4f}')
    print(f'Q\t{lambda_Qfactor*10000:10.2f} bp, \t\t{tstat_Qfactor[0,]:10.2f}, \t{2*(t.cdf(-abs(tstat_Qfactor[0,]), dof)):.4f}')
    
# -

# ### heatmap

# range of penalty
lambda1 = 10**np.arange(-14.0, -7.0, 1)
lambda2 = 10**np.arange(-14.0, -7.0, 1)

TestFactor.shape

# +
# NaN 3D array
temp = 1
result_heatmap = np.empty((lambda1.shape[0],lambda2.shape[0], TestFactor.shape[1]))
result_heatmap[:,:,:] = np.nan

for j in range(TestFactor.shape[1]):
    gt = TestFactor[:,j].reshape(1,-1)
    print(f'order:{j} - {esg_factors.columns[j]}') # show the testfactor which is being analyzed

    for l1_ind, l1 in enumerate(lambda1):
        for l2_ind, l2 in enumerate(lambda2):
            # calculate tstat_ds
            tstat_ds = np.nan # reset the var.
            model_ds = DS(Ri.T, gt, ht_pca, tune_1st=l1, tune_2nd=l2, mode='given')
            tstat_ds = model_ds['lambdag_ds']/model_ds['se_ds']

            # insert into the 3D array
            result_heatmap[l1_ind,l2_ind,j] = tstat_ds

            # show the iteration state
            l1_per = l1_ind/lambda1.shape[0]
            l2_per = l2_ind/lambda2.shape[0]
            sys.stdout.write("\r")
            sys.stdout.write("l1 {:5.2f}, l2 {:5.2f}, tstat {:5.2f}".format(np.log10(l1), np.log10(l2), result_heatmap[l1_ind,l2_ind,j]))
            sys.stdout.flush()

    print('\n')
# -

result_heatmap_flat = np.empty((lambda1.shape[0] * TestFactor.shape[1], lambda2.shape[0]))
result_heatmap_flat[:,:] = np.nan
for j in range(result_heatmap.shape[2]):
    result_heatmap_flat[lambda1.shape[0]*j:lambda1.shape[0]*(j+1),:] = result_heatmap[:,:,j]

import matplotlib.pyplot as plt
import seaborn as sns

np.save('result_heatmap_flat_pca_pillar_withESG2',result_heatmap_flat)

result = pd.DataFrame(result_heatmap_flat.T)

b = int(result.shape[1]/TestFactor.shape[1])
for j in range(TestFactor.shape[1]):
    result_temp = result.iloc[:,b*j:b*(j+1)]
    result_temp.index = np.round(np.log10(lambda2), 3) 
    result_temp.columns = np.round(np.log10(lambda1), 3)
    
    print(esg_factors.columns[j])
    sns.heatmap( result_temp, annot=True, cmap='vlag', cbar=True, vmin=-3, vmax=3) # for color reverse: _r
    plt.xlabel('penalty for 1st')
    plt.ylabel('penalty for 2nd')
    plt.show()

b = int(result.shape[1]/TestFactor.shape[1])
for j in range(TestFactor.shape[1]):
    result_temp = result.iloc[:,b*j:b*(j+1)]
    result_temp.index = np.round(np.log10(lambda2), 3) 
    result_temp.columns = np.round(np.log10(lambda1), 3)
    
    print(esg_factors.columns[j])
    sns.heatmap( result_temp, annot=False, cmap='vlag', cbar=True, vmin=-3, vmax=3) # 색방향 반대로 하려면 _r 붙이기  viridis
    plt.xlabel('penalty for 1st')
    plt.ylabel('penalty for 2nd')
    plt.show()

# # HXZ alt: bivariate

# ## import and concat HXZ bivariate pf








