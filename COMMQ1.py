import numpy as np
import DeathQ1 as DQ1
import LapseQ1 as LQ1

def L_and_D_COMM_Liabilities(d_proba, l_proba, d_DPV, l_DPV, COMM):
    d_COMM_liabilities = d_proba * d_DPV * COMM
    l_COMM_liabilities = l_proba * l_DPV * COMM
    return np.array(d_COMM_liabilities), np.array(l_COMM_liabilities)

def L_and_D_COMM_Liabilities_MC(m_MC, COMM, N, lapse_rate, px_cumul, discount_factors, qx, F):
    """
    Calculate the commission liabilities for death and lapse using Monte Carlo simulation.
    """
    # Calculate death probabilities
    death_p = DQ1.death_proba(px_cumul, lapse_rate, qx)[:N]
    # Calculate lapse probabilities
    lapse_p = LQ1.lapse_proba(px_cumul, lapse_rate)[:N]

    sum = 0

    for m in range(m_MC):
        # Calculate what you get when you die
        death_what_you_get = DQ1.death_what_do_you_get(F[m], discount_factors, F[0][0])
        # Calculate what you get when you lapse
        lapse_what_you_get = LQ1.lapse_what_do_you_get(F[m], discount_factors)

        # Calculate L and D commission liabilities
        L_and_D_COMM_Liab = L_and_D_COMM_Liabilities(death_p, lapse_p, death_what_you_get, lapse_what_you_get, COMM)
        sum += L_and_D_COMM_Liab[0] + L_and_D_COMM_Liab[1]
    
    # Calculate the COMM liabilities using Monte Carlo simulation
    COMM_Liab_MC = sum / m_MC
    return np.array(COMM_Liab_MC)

def L_and_D_COMM_Liabilities_VaR(m_MC, COMM, N, lapse_rate, px_cumul, discount_factors, qx, F):
    # Calculate death probabilities
    death_p = DQ1.death_proba(px_cumul, lapse_rate, qx)[:N]
    # Calculate lapse probabilities
    lapse_p = LQ1.lapse_proba(px_cumul, lapse_rate)[:N]
    COMM_array = np.zeros(m_MC)
    for m in range(m_MC):
        # Calculate what you get when you die
        death_what_you_get = DQ1.death_what_do_you_get(F[m], discount_factors, F[0][0])
        # Calculate what you get when you lapse
        lapse_what_you_get = LQ1.lapse_what_do_you_get(F[m], discount_factors)
        # Calculate L and D commission liabilities
        L_and_D_COMM_Liab = L_and_D_COMM_Liabilities(death_p, lapse_p, death_what_you_get, lapse_what_you_get, COMM)
        COMM_array[m] = L_and_D_COMM_Liab[0].sum() + L_and_D_COMM_Liab[1].sum() # Total L and D commission liabilities for each scenario
    return COMM_array