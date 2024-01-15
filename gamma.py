import pandas as pd
import numpy as np
import scipy
from scipy.stats import norm
import matplotlib.pyplot as plt

from datetime import timedelta, date, datetime



pd.options.display.float_format = '{:,.4f}'.format
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
        gamma = K * np.exp(-r*T) * norm.pdf(d2) / (S * S * sigma * np.sqrt(T))
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

"""
CBOE file import
"""

# This assumes the CBOE file format hasn't been edited, i.e. table beginds at line 4
optionsFile = open(filename)
optionsFileData = optionsFile.readlines()
optionsFile.close()

# Get SPX Spot
spotLine = optionsFileData[1]
spotPrice = float(spotLine.split('Last:')[1].split(',')[0])
fromStrike = 0.8 * spotPrice
toStrike = 1.2 * spotPrice

# Get Today's Date
dateLine = optionsFileData[2]
todayDate = dateLine.split('Date: ')[1].split(',')
monthDay = todayDate[0].split(' ')


# Get index Options Data
df = pd.read_csv(filename, sep=",", header=None, skiprows=4)
df.columns = ['ExpirationDate','Calls','CallLastSale','CallNet','CallBid','CallAsk','CallVol',
              'CallIV','CallDelta','CallGamma','CallOpenInt','StrikePrice','Puts','PutLastSale',
              'PutNet','PutBid','PutAsk','PutVol','PutIV','PutDelta','PutGamma','PutOpenInt']

df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'], format='%a %b %d %Y')
df['ExpirationDate'] = df['ExpirationDate'] + timedelta(hours=16)
df['StrikePrice'] = df['StrikePrice'].astype(float)
df['CallIV'] = df['CallIV'].astype(float)
df['PutIV'] = df['PutIV'].astype(float)
df['CallGamma'] = df['CallGamma'].astype(float)
df['PutGamma'] = df['PutGamma'].astype(float)
df['CallOpenInt'] = df['CallOpenInt'].astype(float)
df['PutOpenInt'] = df['PutOpenInt'].astype(float)

"""


CALCULATE SPOT GAMMA 
Gamma Exposure = Unit Gamma * Open Interest * Contract Size * Spot Price 
To further convert into 'per 1% move' quantity, multiply by 1% of spotPrice
"""
df['CallGEX'] = df['CallGamma'] * df['CallOpenInt'] * 100 * spotPrice * spotPrice * 0.01
df['PutGEX'] = df['PutGamma'] * df['PutOpenInt'] * 100 * spotPrice * spotPrice * 0.01 * -1

df['TotalGamma'] = (df.CallGEX + df.PutGEX) / 10**9
dfAgg = df.groupby(['StrikePrice']).sum(numeric_only=True)
strikes = dfAgg.index.values


# Chart 1: Absolute Gamma Exposure
plt.grid()
plt.bar(strikes, dfAgg['TotalGamma'].to_numpy(), width=10, linewidth=0.1, edgecolor='k', label="Gamma Exposure")
plt.xlim([fromStrike, toStrike])
chartTitle = "Total Gamma: $" + str("{:.2f}".format(df['TotalGamma'].sum())) + " Bn per 1% SPX Move"
plt.title(chartTitle, fontweight="bold", fontsize=20)
plt.xlabel('Strike', fontweight="bold")
plt.ylabel('Spot Gamma Exposure ($ billions/1% move)', fontweight="bold")
plt.axvline(x=spotPrice, color='r', lw=1, label="SPX Spot: " + str("{:,.0f}".format(spotPrice)))
plt.legend()
plt.show()
