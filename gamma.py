import pandas as pd
import numpy as np
import scipy
from scipy.stats import norm
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta, date, datetime
import calendar 

filename = 'spx_quotedata.csv'
"""
Black Scholes with mertons formula accounting for dividens
"""
def Gamma( S, K, T, r, q, sigma, oType, OI):
    """
    C = Call option price
    S = Current Stock price
    K = Strike
    r = interest rate
    q = dividend yield
    t = time to maturity
    N = norm distrubution
    gamma = [e^(-q*t)/(S*sigma*root(t))]*N'(d1)
    """
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    """
    SUM gamma contribuions gives total gamma exposure
    # total option’s change in delta per ONE POINT move in the index
    # convert to %: amt of points 1% = S * 0.01 
    """
    if oType == 'call':
        gamma = np.exp(-q*T) * norm.pdf(d1) / (S*sigma*np.sqrt(T))
        return (OI*100*S**2*0.01*gamma)
    else:
        gamma = K * np.exp(-r*T) * norm.pdf(d2) / (S * S * vol * np.sqrt(T))
        return OI * 100 * S * S * 0.01 * gamma 

"""
Monthly expierys every 3rd friday
3rd days of week fall between 15-21
2nd 8-14
1st 1-7
"""

def is_third_friday(s):
    d = datetime.strptime(s, '%b %d, %Y')
    return d.weekday() == 4 and 15 <= d.day <= 21 #3rd Friday
print (is_third_friday('Jan 19, 2024'))  # True
print (is_third_friday('Feb 16, 2024'))  # True
print (is_third_friday('Jan 12, 2024'))  # False