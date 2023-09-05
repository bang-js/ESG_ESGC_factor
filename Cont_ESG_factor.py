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

# # import and merge

# ## Q and FF5

Qfactor = pd.read_csv('C:/Users/Jeongseok_Bang/OneDrive/바탕 화면/석사과정/FinLab/23 4 ESG controversies risk factor/data/Qfactor.csv', index_col=0)


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

allfactors['R_IA'].describe()

# ## ESG factor

# ### Select ONE-WAY or TWO-WAY

# #### one-way

quantile_set = 3

esg_factor_data_list = [f'esg_to_at_value_q{quantile_set}', 
                        f'esg_to_sale_value_q{quantile_set}', 
                        f'esg_dot_liq_value_q{quantile_set}',
                        f'esg_dot_oancf_value_q{quantile_set}', 
                        f'esg_to_booklev_value_q{quantile_set}', 
                        f'esg_to_ad_value_q{quantile_set}']

# #### two-way

quantile_1_set = 3
quantile_2_set = 2

esg_factor_data_list = [f'esg_to_at_value_two_size_factor_q{quantile_1_set}_{quantile_2_set}', 
                        f'esg_to_sale_value_two_size_factor_q{quantile_1_set}_{quantile_2_set}', 
                        f'esg_dot_liq_value_two_size_factor_q{quantile_1_set}_{quantile_2_set}',
                        f'esg_dot_oancf_value_two_size_factor_q{quantile_1_set}_{quantile_2_set}', 
                        f'esg_to_booklev_value_two_size_factor_q{quantile_1_set}_{quantile_2_set}', 
                        f'esg_to_ad_value_two_size_factor_q{quantile_1_set}_{quantile_2_set}']

# ### merge

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

esg_factors.columns = ['ESG_to_AT', 'ESG_to_SALE', 'ESG_dot_LIQ', 'ESG_dot_OANCF', 'ESG_to_LEV', 'ESG_to_AD']

esg_factors

esg_factors.describe()

# ### merge allfactors and esg_factor

allfactors = allfactors.join(esg_factors, how='outer')

allfactors

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

# ## Summary stat.

round(factors.describe(), 3)

total_factor_list = ['MktRF', 'SMB', 'HML', 'MOM', 'RMW', 'CMA',	'R_IA',	'R_ROE']
total_factor_list.extend(esg_factors.columns.to_list())
round(factors[total_factor_list].corr(),3)

# +
cum_factors = factors*0.01 + 1
cum_factors = cum_factors.cumprod()

cum_factors[esg_factors.columns].plot()
# -

cum_factors[['R_ROE','ESG_dot_OANCF']].plot()

# # alpha

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


# # Feng et al. 2020

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


# -

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


esg_factor_name_lst

# +
'''
one esg factor
'''
esg_ind = int((np.where(factorname == 'ESG_dot_OANCF'))[0]) 
gt = factors_v[:,esg_ind].reshape(1,-1)
Ctrs = [CAPM, FF3, C4, FF5, Qfactor]
Ctrs_exp = ['const - ESG - mkt', 'const - ESG - mkt s h', 'const - ESG - mkt s h m', 'const - ESG - mkt s h r c', 'const - ESG - mkt s ia roe']

for i, c in enumerate(Ctrs):
    print(PriceRisk_OLS_2(Ri.T, gt, c).summary())
    print(Ctrs_exp[i])        
    show_coef_tstat_np(PriceRisk_OLS_2(Ri.T, gt, c))
    print('='*10)

# +
'''
one esg factor
'''
esg_ind = int((np.where(factorname == 'ESG_to_AD'))[0]) 
gt = factors_v[:,esg_ind].reshape(1,-1)
Ctrs = [CAPM, FF3, C4, FF5, Qfactor]
Ctrs_exp = ['const - ESG - mkt', 'const - ESG - mkt s h', 'const - ESG - mkt s h m', 'const - ESG - mkt s h r c', 'const - ESG - mkt s ia roe']

for i, c in enumerate(Ctrs):
    print(PriceRisk_OLS_2(Ri.T, gt, c).summary())
    print(Ctrs_exp[i])        
    show_coef_tstat_np(PriceRisk_OLS_2(Ri.T, gt, c))
    print('='*10)
