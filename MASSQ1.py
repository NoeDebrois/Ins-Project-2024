import numpy as np
import DeathQ1 as DQ1
import COMMQ1 as CQ1
import LapseQ1 as LQ1

def lapse_proba(px_cumul, lapse_rate, lapse_MASS):
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
    return lapse_p * (1 - lapse_MASS) * (1 / (1 - lapse_rate)) * lapse_rate

def death_proba(px_cumul, lapse_rate, qx, lapse_MASS):
    """
    """
    death_p = np.zeros(px_cumul.shape)
    for k in range(1, len(px_cumul)):
        if k == 1:
            death_p[k] = 1 * (1 - lapse_rate) ** (k - 1) * qx[k - 1]
        else:
            death_p[k] = px_cumul[k - 2] * (1 - lapse_rate) ** (k - 1) * qx[k - 1]
    return death_p * (1 - lapse_MASS) * (1 / (1 - lapse_rate))

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

def MASS_lapse_liabilities(RD, lapse_proba, lapse_what_do_you_get):
    """
    Calculate the lapse liabilities i.e. the net present value of future cash flows that will be paid to the policyholders who lapse.
    i.e. the expected value of the discounted cash flows that will be paid to the policyholders who lapse.

    Returns:
    array-like: Array of lapse liabilities.
    """
    return (1 - RD) * lapse_proba * lapse_what_do_you_get

def MASS_LL_VaR(m_MC, F, discount_factors, px_cumul, lapse_rate, rd_rate, lapse_MASS):
    # Calculate lapse probabilities
    lapse_p = lapse_proba(px_cumul, lapse_rate, lapse_MASS)
    lapse_array = np.zeros(m_MC)
    for m in range(m_MC):
        # Calculate what you get when you lapse
        lapse_what_you_get = lapse_what_do_you_get(F[m], discount_factors)
        # Calculate lapse liabilities
        lapse_liab = LQ1.lapse_liabilities(rd_rate, lapse_p, lapse_what_you_get)
        lapse_array[m] = lapse_liab.sum() # Total lapse liabilities for each scenario
    return lapse_array

def MASS_L_and_D_COMM_L_VaR(m_MC, COMM, N, lapse_rate, px_cumul, discount_factors, qx, F, lapse_MASS):
    # Calculate death probabilities
    death_p = death_proba(px_cumul, lapse_rate, qx, lapse_MASS)[:N]
    # Calculate lapse probabilities
    lapse_p = lapse_proba(px_cumul, lapse_rate, lapse_MASS)[:N]
    COMM_array = np.zeros(m_MC)
    for m in range(m_MC):
        # Calculate what you get when you die
        death_what_you_get = DQ1.death_what_do_you_get(F[m], discount_factors, 100000)
        # Calculate what you get when you lapse
        lapse_what_you_get = LQ1.lapse_what_do_you_get(F[m], discount_factors)
        # Calculate L and D commission liabilities
        L_and_D_COMM_Liab = CQ1.L_and_D_COMM_Liabilities(death_p, lapse_p, death_what_you_get, lapse_what_you_get, COMM)
        COMM_array[m] = L_and_D_COMM_Liab[0].sum() + L_and_D_COMM_Liab[1].sum() # Total L and D commission liabilities for each scenario
    return COMM_array