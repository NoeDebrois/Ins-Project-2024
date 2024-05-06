import numpy as np

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

def MC_death_liabilities(m_MC, F, discount_factors, px_cumul, lapse_rate, RD, qx):
    """
    Calculate the death liabilities using Monte Carlo simulation.
    """
    # Calculate death probabilities
    death_p = death_proba(px_cumul, lapse_rate, qx)

    sum = 0

    for m in range(m_MC):
        # Calculate what you get when you die
        death_what_you_get = death_what_do_you_get(F[m], discount_factors, F[0][0])

        # Calculate death liabilities
        death_liab = death_liabilities(RD, death_p, death_what_you_get)
        sum += death_liab
    
    # Calculate the death liabilities using Monte Carlo simulation
    death_liab_MC = sum / m_MC
    return np.array(death_liab_MC)

def MC_death_liabilities_VaR(m_MC, F, discount_factors, px_cumul, lapse_rate, RD, qx):
    # Calculate death probabilities
    death_p = death_proba(px_cumul, lapse_rate, qx)
    death_array = np.zeros(m_MC)
    for m in range(m_MC):
        # Calculate what you get when you die
        death_what_you_get = death_what_do_you_get(F[m], discount_factors, F[0][0])
        # Calculate death liabilities
        death_liab = death_liabilities(RD, death_p, death_what_you_get)
        death_array[m] = death_liab.sum() # Total death liabilities for each scenario
    return death_array