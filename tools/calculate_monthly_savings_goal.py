from langchain.tools import tool

@tool
def calculate_monthly_savings_goal(
    target_amount: float,
    initial_amount: float,
    annual_return_rate: float,
    years: int
) -> float:
    """ calculate the monthly savings goal to reach a target amount given an initial amount, annual return rate, and time frame in years. """
    months = years * 12
    monthly_rate = annual_return_rate / 12

    if monthly_rate == 0:
        return max(0, (target_amount - initial_amount) / months)

    future_lump_sum = initial_amount * (1 + monthly_rate) ** months

    remaining_target = target_amount - future_lump_sum

    if remaining_target <= 0:
        return 0

    savings_factor = ((1 + monthly_rate) ** months - 1) / monthly_rate

    monthly_contribution = remaining_target / savings_factor

    return monthly_contribution