from langchain.tools import tool

@tool
def calculate_future_value(
    initial_amount: float,
    monthly_contribution: float,
    annual_return_rate: float,
    years: int
) -> float:
    """ calculate the future value of an investment given an initial amount, monthly contribution, annual return rate, and time frame in years. """
    months = years * 12
    monthly_rate = annual_return_rate / 12

    if monthly_rate == 0:
        return initial_amount + (monthly_contribution * months)

    future_lump_sum = initial_amount * (1 + monthly_rate) ** months

    future_monthly_contributions = monthly_contribution * (
        ((1 + monthly_rate) ** months - 1) / monthly_rate
    )

    return future_lump_sum + future_monthly_contributions