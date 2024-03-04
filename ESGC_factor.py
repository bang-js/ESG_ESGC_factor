import pandas as pd
import numpy as np
from numpy import matlib as mb
from sklearn.linear_model import Lasso, lasso_path, LassoCV
from sklearn.model_selection import KFold, train_test_split
from matplotlib import pyplot as plt
import sys
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import statsmodels.api as sm
import statsmodels.formula.api as smf
import calendar
from numpy import mat, cov, mean, hstack, multiply,sqrt,diag, \
    squeeze, ones, array, vstack, kron, zeros, eye, savez_compressed
from numpy.linalg import inv
from scipy.stats import chi2
from pandas import read_csv
import statsmodels.api as sm

################################
# 1. import and merge
################################
################
# 1.1. Q-factor and FF5
################
Qfactor = pd.read_csv('Qfactor.csv', index_col=0)
# Source-> https://global-q.org/factors.html

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
Qfactor = pd.read_csv('Qfactor.csv', index_col=0)
Qfactor.index = pd.date_range(convert_index_format(Qfactor.index[0]), convert_index_format(Qfactor.index[-1]), freq='M')

# import FF5 factor
FF5 = pd.read_csv('FF5.csv', index_col=0)
FF5.index = pd.date_range('1963-07-31', '2023-02-28', freq='M')
FF5.rename(columns={'Mkt-RF':'MktRF'}, inplace=True)

# emerge FF5 and Qfactor
allfactors = FF5.join(Qfactor, how='outer')

################
# 1.2. ESG factor
################

esg_quantile = 3
esgc_quantile = 3

# select the portfolio
esg_factor_list = ['esgc_factor_value', f'esg_factor_value_{esg_quantile}'] # univariate (uni-breakpoint for ESGC)
# esg_factor_list = ['esgc_size_mulfactor', 'esg_size_mulfactor'] # bivariate (uni-breakpoint for ESGC)

# import and merge all esg factors
for i, e in enumerate(esg_factor_list):
    e_temp =  pd.read_csv('{}.csv'.format(e), index_col=0)
    e_temp.index = pd.to_datetime(e_temp.index)
    e_temp.columns = ['esg_f_{}'.format(i)]
    e_temp *= 100
    
    if i == 0:
        esg_factors = e_temp
    else: # i>0
        esg_factors = esg_factors.join(e_temp, how='outer')

# emerge allfactors and esg_factor
allfactors = allfactors.join(esg_factors, how='outer')

################
# 1.3. period cut
################
factors = allfactors[(allfactors.index >= '2003-07-31') & (allfactors.index <= '2021-12-31')]
factors.dropna(inplace=True)

################
# 1.4. test portfolio
################
# import HXZ portfolio
HXZ_port = pd.read_csv('HXZ_port.csv', index_col=0)
HXZ_port.index = pd.date_range('1967-01-31', '2021-12-31', freq='M')
HXZ_port = HXZ_port.loc[factors.index[0]:][:]

# import JKP factors
JKP = pd.read_csv('df_factorzoo_JKP.csv', index_col=0)
JKP.index.names = ['Date']
JKP.index = pd.to_datetime(JKP.index)
JKP = JKP.loc[factors.index[0]:][:]

################
# 1.5. MOM from JKP
################
# Adding momentum for Carhart 4 factor model
factors['MOM'] = JKP['ret_12_1'] * 100

################
# 1.6. summary & plot
################
print(round(factors.describe(), 3))
print(round(factors[['MktRF', 'SMB', 'HML', 'MOM', 'RMW', 'CMA',	'R_IA',	'R_ROE',	'ESG',	'ESGC', 'ESG_size']].corr(),3))

cum_factors = factors*0.01 + 1
cum_factors = cum_factors.cumprod()
plt.plot(factors.index, cum_factors['ESG'], label='ESG', linestyle='dashed')
# plt.plot(factors.index, cum_factors['ESG_size'], label='ESGN', linestyle='dotted')
plt.plot(factors.index, cum_factors['ESGC'], label='ESGC')
plt.xlabel('year')
plt.ylabel('cumulative return')
plt.legend()
plt.show()

################################
# 2. alpha
################################
factors['y1'] = factors['ESGC'] - factors['R_F'] 
factors['y2'] = factors['ESG'] - factors['R_F'] 

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

# set the target
target_alpha = 'y1' # ESGC
# target_alpha = 'y2' # ESG

'''capm'''
capm = sm.OLS(factors[target_alpha], sm.add_constant(factors['MktRF'])).fit(cov_type='HAC',cov_kwds={'maxlags':6})
show_coef_tstat(capm)
print(capm.summary())

'''FF3'''
ff3 = smf.ols(formula='{} ~ MktRF + SMB + HML'.format(target_alpha), data=factors.iloc[:][:]).fit(cov_type='HAC',cov_kwds={'maxlags':6})
show_coef_tstat(ff3)
print(ff3.summary())

'''C4'''
c4 = smf.ols(formula='{} ~ MktRF + SMB + HML + MOM'.format(target_alpha), data=factors.iloc[:][:]).fit(cov_type='HAC',cov_kwds={'maxlags':6})
show_coef_tstat(c4)
print(c4.summary())

'''FF5'''
ff5 = smf.ols(formula='{} ~ MktRF + SMB + HML + RMW + CMA'.format(target_alpha), data=factors.iloc[:][:]).fit(cov_type='HAC',cov_kwds={'maxlags':6})
show_coef_tstat(ff5)
print(ff5.summary())

'''q-factor'''
q_factor = smf.ols(formula='{} ~ MktRF + SMB + R_IA + R_ROE'.format(target_alpha), data=factors.iloc[:][:]).fit(cov_type='HAC',cov_kwds={'maxlags':6})
show_coef_tstat(q_factor)
print(q_factor.summary())

################################
# 3. Price of covariance risk 
################################
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

HXZ_port_ex = HXZ_port - factors['R_F'].values.reshape((-1,1))
Ri = HXZ_port_ex.iloc[:][:]
factorname = factors.columns
factorname_full = factors.columns
factors_v = factors.iloc[:][:].values
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

################
# 3.1. ESG or ESGC factor
################
esg_ind = int((np.where(factorname == 'ESGC'))[0])
# esg_ind = int((np.where(factorname == 'ESG'))[0])
gt = factors_v[:,esg_ind].reshape(1,-1)
Ctrs = [CAPM, FF3, C4, FF5, Qfactor]

for i, c in enumerate(Ctrs):
    print(PriceRisk_OLS_2(Ri.T, gt, c).summary())
    print('\n constant - ESG - controls[mkt...]')
    show_coef_tstat_np(PriceRisk_OLS_2(Ri.T, gt, c))
    print('='*10)

################
# 3.2. both ESG and ESGC factor
################
esg2_ind = int((np.where(factorname == 'ESG'))[0]) # target
esg3_ind = int((np.where(factorname == 'ESGC'))[0]) 

gt = factors_v[:,esg2_ind].reshape(1,-1)
ctl_m = [CAPM, FF3, C4, FF5, Qfactor]
temp_ctl_m = []
for _ in ctl_m:
    __ = np.concatenate((_, factors_v[:,esg3_ind].reshape(1,-1)), axis=0)
    temp_ctl_m.append(__)
    
CAPM_esg, FF3_esg, C4_esg, FF5_esg, Qfactor_esg = temp_ctl_m

for i, c in enumerate(temp_ctl_m):
    print(PriceRisk_OLS_2(Ri.T, gt, c).summary())
    print('\n constant - ESG_target - controls[mkt...ESG_other]')
    show_coef_tstat_np(PriceRisk_OLS_2(Ri.T, gt, c))
    print('='*10)

###############################
# 4. Fama-MacBeth regression
###############################
data_FM = pd.concat([Ri, factors], axis=1)

def FMB(flst):

    # Split using both named colums and ix for larger blocks
    factormodel = data_FM[flst].values #
    excessReturns = data_FM.iloc[:, :1853].values 

    # Use mat for easier linear algebra
    factormodel = mat(factormodel)
    excessReturns = mat(excessReturns)

    # Shape information
    T,K = factormodel.shape
    T,N = excessReturns.shape

    # Time series regressions
    X = sm.add_constant(factormodel)
    ts_res = sm.OLS(excessReturns, X).fit()
    alpha = ts_res.params[0]
    beta = ts_res.params[1:]
    avgExcessReturns = mean(excessReturns, 0)

    # Cross-section regression
    beta_wconst = sm.add_constant(beta.T)
    cs_res = sm.OLS(avgExcessReturns.T, beta_wconst).fit(cov_type='HAC',cov_kwds={'maxlags':6})  # no constant
    
    return cs_res

orders = [
    ['MktRF'],
    ['MktRF', 'SMB', 'HML'],
    ['MktRF', 'SMB', 'HML', 'MOM'],
    ['MktRF', 'SMB', 'HML', 'RMW', 'CMA'],
    ['MktRF', 'SMB', 'R_IA', 'R_ROE'],
    ['ESG', 'MktRF'],
    ['ESG', 'MktRF', 'SMB', 'HML'],
    ['ESG', 'MktRF', 'SMB', 'HML', 'MOM'],
    ['ESG', 'MktRF', 'SMB', 'HML', 'RMW', 'CMA'],
    ['ESG', 'MktRF', 'SMB', 'R_IA', 'R_ROE'],
    ['ESGC', 'MktRF'],
    ['ESGC', 'MktRF', 'SMB', 'HML'],
    ['ESGC', 'MktRF', 'SMB', 'HML', 'MOM'],
    ['ESGC', 'MktRF', 'SMB', 'HML', 'RMW', 'CMA'],
    ['ESGC', 'MktRF', 'SMB', 'R_IA', 'R_ROE'],
    ['ESG', 'ESGC', 'MktRF'],
    ['ESG', 'ESGC', 'MktRF', 'SMB', 'HML'],
    ['ESG', 'ESGC', 'MktRF', 'SMB', 'HML', 'MOM'],
    ['ESG', 'ESGC', 'MktRF', 'SMB', 'HML', 'RMW', 'CMA'],
    ['ESG', 'ESGC', 'MktRF', 'SMB', 'R_IA', 'R_ROE']
]

for o in orders:
    print(o)
    FMB(o)
    print(FMB(o).summary())
    print(show_coef_tstat_np(FMB(o)))
