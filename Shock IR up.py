# -*- coding: utf-8 -*-
"""
Created on Sat May  4 00:18:17 2024

@author: vince
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Simulation :
m_MC = 100 # Number of simulations
T = 50 # Time horizon in years
N = 50 # Number of time steps (daily frequency)
dt = T / N # Time step
initial_population = 10000  # Initial population size (number of contracts at t=0)


# Fund / Premium :
F0 = 100000 # Initial value of the fund / Premium Invested
sigma_equity = 0.20 # Volatility for equity
sigma_property = 0.10 # Volatility for property
PR_weight = 0.2 # Weight of PR in the portfolio
EQ_weight = 0.8 # Weight of EQ in the portfolio

# Liabilities :
RFR = 0.03
rd_rate = 0.022
COMM = 0.014
inflation_rate = 0.02
lapse_rate = 0.15  # Annual lapse rate
cst_cost = 50 # Constant unitary cost

# Generate the 50€ vector following the inflation :
Expenses = np.zeros(N)
Expenses[1:] = [cst_cost * (1 + inflation_rate) ** i for i in range(1,N)]
Expenses

# Path to your Excel file
file_path = 'EIOPA_RFR_20240331_Term_Structures.xlsx'

# Read Excel file
df = pd.read_excel(file_path, "Spot_NO_VA_shock_UP", usecols="C", )
df = df[11:]
df = df.rename(columns={'Unnamed: 2': 'EIOPA EU without VA, MAR'})
rt_up = np.array(df).T
rt_up = np.array(rt_up, np.float64)
rt_up = rt_up[0, :N] # Time-varying risk-free rate

df
rt_up

def simulate_gbm_vec(F0, r, sigma, dt, N, m_MC):
    np.random.seed(123456)#1234)
    # Generate N samples from a GBM with risk-free rate r and volatility sigma
    # r and sigma are assumed to be annualized
    # dt is the time step
    # S0 is the initial value of the process
    S = np.zeros((m_MC, N))
    S[:, 0] = np.full(m_MC, F0)
    for t in range(1, N):
        # Generate m_MC samples of random normal variables
        z = np.random.normal(size=m_MC)
        # Update the stock price for all iterations simultaneously
        S[:, t] = S[:, t - 1] * np.exp((r[t] - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return S

# Simulate equity and property paths
EQt_rt_up = simulate_gbm_vec(F0, rt_up, sigma_equity, dt, N, m_MC)
PRt_rt_up = simulate_gbm_vec(F0, rt_up, sigma_property, dt, N, m_MC)

# Calculate the fund value at each time step
fund_value_rt_up = EQ_weight * EQt_rt_up + PR_weight * PRt_rt_up
F_rt_up = np.squeeze(fund_value_rt_up)

# Define time interval correctly
time = np.linspace(0,T,N)
# Require numpy array that is the same shape as F
tt = np.full(shape=(m_MC,N), fill_value=time).T
# Check that the dimensions are OK for the plot
print(str(tt.shape) + str(fund_value_rt_up.T.shape))

plt.plot(tt, fund_value_rt_up.T);
plt.xlabel("Years $(t)$")
plt.ylabel("Portfolio Value $(F_t)$")
plt.title("Portfolio value (€) VS time (year)\n $F_t = EQ_t + PR_t$, $F_0 = 100000€$")
plt.show()

EIOPA_yield_curve = pd.DataFrame()
EIOPA_yield_curve['Years'] = np.arange(0, 51, 1)
EIOPA_yield_curve['EIOPA EU without VA, MARCH 2024'] = np.insert(rt_up, 0, 0)  # EIOPA RFR spot rate no VA, with 0 at year = 0 so that discount = 1
EIOPA_yield_curve['Discount Factors'] = 1 / (1 + EIOPA_yield_curve['EIOPA EU without VA, MARCH 2024']) ** EIOPA_yield_curve['Years']
EIOPA_yield_curve

df_LT = pd.read_excel("LT_M_only.xlsx", skiprows=[0], index_col=0)
df_LT

Life_table = pd.DataFrame()
Life_table['Age'] = np.arange(60, 120, 1)
Life_table['Death probability qx'] = np.array(df_LT['Probability of death (per thousand)  qx'][60:120]) / 1000
Life_table['Cumulative survival probability px'] = (1 - Life_table['Death probability qx']).cumprod()
Life_table

# Life table (taken from the real italian life table)
life_table = Life_table['Death probability qx']

# Initialize lists to store population data
population_in_contract = [initial_population]
population_lapsed = [0]
population_dead = [0]

# Simulate population evolution over 50 years
for year in range(1, N + 1):
    # Calculate number of deaths
    # deaths = np.random.binomial(population_in_contract[-1], life_table[min(year, max(life_table.keys()))])
    deaths = np.random.binomial(population_in_contract[-1], life_table[year - 1])

    # Calculate number of lapses
    lapses = int(population_in_contract[-1] * lapse_rate)

    # Update population
    population_in_contract.append(max(0, population_in_contract[-1] - deaths - lapses))
    population_lapsed.append(population_lapsed[-1] + lapses)
    population_dead.append(population_dead[-1] + deaths)

# Plot population evolution
years_range = range(N + 1)
plt.plot(years_range, population_in_contract, label='In Contract')
plt.plot(years_range, population_lapsed, label='Lapsed')
plt.plot(years_range, population_dead, label='Dead')
plt.xlabel('Years')
plt.ylabel('Population')
plt.title('Population Evolution Over Time')
plt.legend()
plt.show()



# Computation of the lapse liabilities :

# Here in the MC, the "randomness" comes from

# : since in case of lapse, the beneficiary gets the value of the fund at the time of lapse, with 20 euros of penalties applied, the lapse liabilities obviously depend on the value of the fund. So we can simulate a vast number of funds at 50 years horizon and then compute (the same vast number of times) the lapse liabilities and finally take the average (expectation).
# Get the data from the ISTATS life table and the EIOPA term structure :
    
px_cumul = Life_table["Cumulative survival probability px"]
discount_factors = EIOPA_yield_curve["Discount Factors"][:-1]

def lapse_proba(px_cumul, lapse_rate):
    """
    Calculate the lapse probability for each period based on the cumulative survival probability and lapse rate.

    Parameters:
    px_cumul (array-like): Array of cumulative survival probabilities.
    lapse_rate (float): Lapse rate for the insurance policy.

    Returns:
    array-like: Array of lapse probabilities for each period.
    """
    # lapse_p = np.array([px_cumul[i - 1] * (1 - lapse_rate) ** i * lapse_rate for i in range(len(px_cumul))])
    lapse_p = np.zeros(px_cumul.shape)

    for i in range(1, len(px_cumul)):
        lapse_p[i] = px_cumul[i - 1] * (1 - lapse_rate) ** (i - 1) * lapse_rate
    return lapse_p

def lapse_what_do_you_get(Ft, discount_factors):
    """
    Calculate the discounted (present) value of future cash flows based on the discount factors and what you get when you lapse.

    Parameters:
    Ft (array-like): Array of future cash flows.
    discount_factors (array-like): Array of discount factors.

    Returns:
    array-like: Array of present values of future cash flows.
    """
    DPV = (Ft - 20) * discount_factors
    DPV[0] = 0
    return DPV

def lapse_liabilities(RD, lapse_proba, lapse_what_do_you_get):
    """
    Calculate the lapse liabilities i.e. the net present value of future cash flows that will be paid to the policyholders who lapse.
    i.e. the expected value of the discounted cash flows that will be paid to the policyholders who lapse.

    Returns:
    array-like: Array of lapse liabilities.
    """
    return (1 - RD) * lapse_proba * lapse_what_do_you_get

def MC_lapse_liabilities(m_MC, F, discount_factors, px_cumul, lapse_rate, RD):
    """
    Calculate the lapse liabilities using Monte Carlo simulation.

    Parameters:
    m_MC (int): Number of Monte Carlo simulations.
    Ft (array-like): Array of future cash flows.
    discount_factors (array-like): Array of discount factors.
    px_cumul (array-like): Array of cumulative survival probabilities.
    lapse_rate (float): Lapse rate for the insurance policy.
    RD (float): Residual death benefit.

    Returns:
    array-like: Array of lapse liabilities.
    """
    # Calculate lapse probabilities
    lapse_p = lapse_proba(px_cumul, lapse_rate)

    sum = 0

    for m in range(m_MC):
        # Calculate what you get when you lapse
        lapse_what_you_get = lapse_what_do_you_get(F[m], discount_factors)

        # Calculate lapse liabilities
        lapse_liab = lapse_liabilities(RD, lapse_p, lapse_what_you_get)
        sum += lapse_liab

    # Calculate the lapse liabilities using Monte Carlo simulation
    lapse_liab_MC = sum / m_MC
    return np.array(lapse_liab_MC)

# Example for the lapse liability :

# MC :
MC_liab = MC_lapse_liabilities(m_MC, F_rt_up, discount_factors, px_cumul[:50], lapse_rate, rd_rate)
MC_liab

MC_liab.sum()

proba = lapse_proba(px_cumul, 0.15)
proba = proba[:50]
what = lapse_what_do_you_get(F_rt_up[0], discount_factors)
# liab = lapse_liabilities(0.022, proba, what)

Lapse = pd.DataFrame()
Lapse["Proba Lapse"] = proba
Lapse["Discounted Present Value of the Lapse Cash Flows"] = what
Lapse["Lapse Liabilities"] = MC_liab
Lapse.head()

# Here we can do a MC simulation also on the population to see if we have kind of the same result :

def forward_diff(arr):
    """
    Compute the forward difference of an array.
    Example :
    arr = [1, 3, 6, 10, 15]
    result = forward_diff(arr)
    print("Forward difference:", result)
    [2, 3, 4, 5]
    """
    arr = np.asarray(arr)  # Convert input to NumPy array
    forward_diff = arr[1:] - arr[:-1]  # Compute forward difference
    return forward_diff

def lapse(array_of_diff, F, discount_factors, T, RD):
    lapses = np.zeros(T)
    for t in range(T):
        lapses[t] = discount_factors[t] * array_of_diff[t] * (F[t] - 20)
    return (1 - RD) * lapses

def MC_lapse(m_MC, rd_rate, discount_factors, T, F, population_lapsed):
    lapses = np.zeros((m_MC, T))
    for i in range(m_MC):
        diff = forward_diff(population_lapsed)
        lapses[i] = lapse(diff, F[i], discount_factors, T, rd_rate)
    return lapses, np.sum(lapses, axis=0)


# Computation of the death liabilities :

# Here in the MC, the "randomness" comes from

# : since in case of death, the beneficiary gets the maximum between the invested premium and the value of the fund, the death liabilities obviously depend on the value of the fund. So we can simulate a vast number of funds at 50 years horizon and then compute (the same vast number of times) the death liabilities and finally take the average (expectation).
# Get the data from the ISTATS life table and the EIOPA term structure :

qx = Life_table["Death probability qx"]

def death_proba(px_cumul, lapse_rate, qx):
    """
    """
    death_p = np.zeros(px_cumul.shape)
    for k in range(1, len(px_cumul)):
        if k == 1:
            death_p[k] = 1 * (1 - lapse_rate) ** (k - 1) * qx[k - 1]
        else:
            death_p[k] = px_cumul[k - 2] * (1 - lapse_rate) ** (k - 1) * qx[k - 1]
    return death_p

def death_what_do_you_get(Ft, discount_factors, C0):
    """
    Compute the maximum between the invested premium and the value of the fund.
    """
    DPV = np.array([max(C0, value) for value in Ft]) * discount_factors
    DPV[0] = 0
    return DPV

def death_liabilities(RD, death_proba, death_what_do_you_get):
    """
    """
    return (1 - RD) * death_proba * death_what_do_you_get

def MC_death_liabilities(m_MC, F, discount_factors, px_cumul, lapse_rate, RD):
    """
    Calculate the death liabilities using Monte Carlo simulation.
    """
    # Calculate death probabilities
    death_p = death_proba(px_cumul, lapse_rate, life_table)

    sum = 0

    for m in range(m_MC):
        # Calculate what you get when you die
        death_what_you_get = death_what_do_you_get(F[m], discount_factors, F0)

        # Calculate death liabilities
        death_liab = death_liabilities(RD, death_p, death_what_you_get)
        sum += death_liab
    
    # Calculate the death liabilities using Monte Carlo simulation
    death_liab_MC = sum / m_MC
    return np.array(death_liab_MC)

# Example from the death liability :

# MC :
d_MC_liab = MC_death_liabilities(m_MC, F_rt_up, discount_factors, px_cumul[:50], lapse_rate, rd_rate)
d_MC_liab

d_MC_liab.sum()

d_proba = death_proba(px_cumul, 0.15, qx)
d_proba = d_proba[:50]
d_what = death_what_do_you_get(F_rt_up[0], discount_factors, F0)

Death = pd.DataFrame()
Death["Proba Death"] = d_proba
Death["Discounted Present Value of the Death Cash Flows"] = d_what
Death["Death Liabilities"] = d_MC_liab
Death.head()


# Computation of the lapse and death commission liabilities :

def L_and_D_COMM_Liabilities(d_proba, l_proba, d_DPV, l_DPV, COMM):
    d_COMM_liabilities = d_proba * d_DPV * COMM
    l_COMM_liabilities = l_proba * l_DPV * COMM
    return np.array(d_COMM_liabilities), np.array(l_COMM_liabilities)

b = L_and_D_COMM_Liabilities(d_proba, proba, d_what, what, COMM)
b[0].sum() + b[1].sum()

def L_and_D_COMM_Liabilities_MC(m_MC, COMM, N):
    """
    Calculate the commission liabilities for death and lapse using Monte Carlo simulation.
    """
    # Calculate death probabilities
    death_p = death_proba(px_cumul, lapse_rate, life_table)[:N]
    # Calculate lapse probabilities
    lapse_p = lapse_proba(px_cumul, lapse_rate)[:N]

    sum = 0

    for m in range(m_MC):
        # Calculate what you get when you die
        death_what_you_get = death_what_do_you_get(F_rt_up[m], discount_factors, F0)
        # Calculate what you get when you lapse
        lapse_what_you_get = lapse_what_do_you_get(F_rt_up[m], discount_factors)

        # Calculate L and D commission liabilities
        L_and_D_COMM_Liab = L_and_D_COMM_Liabilities(death_p, lapse_p, death_what_you_get, lapse_what_you_get, COMM)
        sum += L_and_D_COMM_Liab[0] + L_and_D_COMM_Liab[1]
    
    # Calculate the COMM liabilities using Monte Carlo simulation
    COMM_Liab_MC = sum / m_MC
    return np.array(COMM_Liab_MC)

L_n_D_COMM_liab = L_and_D_COMM_Liabilities_MC(m_MC, COMM, N)
L_n_D_COMM_liab.sum()


# Computation of the expenses liabilities :

def Expenses_liabilities(Expenses, discount_factors, lapse_rate, px_cumul, N):
    Expenses_liab = np.zeros(N)
    Expenses_liab[1] = Expenses[1] * discount_factors[1] * 1 * 1 # (1 - lapse_rate) ** 0 = 1 and px_cumul = 1 at zeroth year
    for i in range(2, N):
        Expenses_liab[i] = Expenses[i] * discount_factors[i] * (1 - lapse_rate) ** (i - 1) * px_cumul[i - 2]
    return Expenses_liab

Exp_liab = Expenses_liabilities(Expenses, discount_factors, lapse_rate, px_cumul, N)
Exp_liab.sum()

# Creation of a liabilities dataframe :

def liabilities_summary(lapse_liab, death_liab, comm_liab, exp_liab, N):
    summary = pd.DataFrame()
    summary["Years"] = np.arange(0, N, 1)
    summary["Lapse"] = lapse_liab
    summary["Death"] = death_liab
    summary["COMM"] = comm_liab
    summary["Expenses"] = exp_liab
    summary["Total per year"] = lapse_liab + death_liab + comm_liab + exp_liab
    return summary

summary = liabilities_summary(MC_liab, d_MC_liab, L_n_D_COMM_liab, Exp_liab, N)
summary

TOT_Liabilities = summary["Total per year"].sum()
TOT_Liabilities

BOF_rt_up = F_rt_up[-1] - TOT_Liabilities # a corriger !
BOF_rt_up

#Supposing that BOF (BOF whithout shock) was computed before

d_BOF_rt_up = BOF - BOF_rt_up
d_BOF_rt_up

SCR_rt_up= max(0, d_BOF_rt_up)
SCR_rt_up