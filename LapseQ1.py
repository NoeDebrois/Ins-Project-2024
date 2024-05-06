import numpy as np
import DeathQ1 as DQ1
import COMMQ1 as CQ1

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

def MC_lapse_liabilities_VaR(m_MC, F, discount_factors, px_cumul, lapse_rate, rd_rate):
    # Calculate lapse probabilities
    lapse_p = lapse_proba(px_cumul, lapse_rate)
    lapse_array = np.zeros(m_MC)
    for m in range(m_MC):
        # Calculate what you get when you lapse
        lapse_what_you_get = lapse_what_do_you_get(F[m], discount_factors)
        # Calculate lapse liabilities
        lapse_liab = lapse_liabilities(rd_rate, lapse_p, lapse_what_you_get)
        lapse_array[m] = lapse_liab.sum() # Total lapse liabilities for each scenario
    return lapse_array