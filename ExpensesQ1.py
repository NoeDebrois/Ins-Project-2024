import numpy as np 

def Expenses_liabilities(Expenses, discount_factors, lapse_rate, px_cumul, N):
    Expenses_liab = np.zeros(N)
    Expenses_liab[1] = Expenses[1] * discount_factors[1] * 1 * 1 # (1 - lapse_rate) ** 0 = 1 and px_cumul = 1 at zeroth year
    for i in range(2, N):
        Expenses_liab[i] = Expenses[i] * discount_factors[i] * (1 - lapse_rate) ** (i - 1) * px_cumul[i - 2]
    return Expenses_liab

def Expenses_liabilities_VaR(Expenses, discount_factors, lapse_rate, px_cumul, N, m_MC):
    Expenses_array = np.zeros(m_MC)
    for m in range(m_MC):
        Expenses_liab = Expenses_liabilities(Expenses, discount_factors, lapse_rate, px_cumul, N)
        Expenses_array[m] = Expenses_liab.sum() # Total expenses liabilities for each scenario
    return Expenses_array