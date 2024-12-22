import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set the start and end dates
start_date = "2023-01-01"
end_date = "2024-12-31"

# Generate a full date range and identify business days
date_range = pd.date_range(start=start_date, end=end_date, freq="D")
business_days = pd.date_range(start=start_date, end=end_date, freq="B")

# Initialize the DataFrame with all dates
data = pd.DataFrame({"date": date_range})
data["in"] = 0
data["out"] = 0


# Function to generate simulated fund flow data
def simulate_fund_flow(business_days, base_amount=5e9):
    np.random.seed(42)  # For reproducibility

    # Identify holidays and their adjacent trading days
    holidays = pd.to_datetime(
        [
            "2023-01-01",
            "2023-01-22",
            "2023-01-23",
            "2023-01-24",
            "2023-05-01",
            "2023-10-01",
            "2024-01-01",
            "2024-02-10",
            "2024-05-01",
            "2024-10-01",
        ]
    )
    holiday_adjacent = []
    for holiday in holidays:
        prev_bday = holiday - pd.offsets.BDay(1)
        next_bday = holiday + pd.offsets.BDay(1)
        holiday_adjacent.extend([prev_bday, next_bday])
    holiday_adjacent = pd.to_datetime(holiday_adjacent)

    # Generate flows with specific patterns
    in_flow = []
    out_flow = []
    for date in business_days:
        # Higher inflows/outflows near holidays
        if date in holiday_adjacent:
            daily_in = base_amount * np.random.uniform(1.2, 1.5)
            daily_out = base_amount * np.random.uniform(1.2, 1.5)
        # Stable values during mid-periods
        elif np.random.rand() < 0.1:  # 10% chance of an unusual day
            daily_in = base_amount * np.random.uniform(0.5, 1.0)
            daily_out = base_amount * np.random.uniform(0.5, 1.0)
        else:
            daily_in = base_amount * np.random.uniform(0.8, 1.2)
            daily_out = base_amount * np.random.uniform(0.8, 1.2)

        in_flow.append(daily_in)
        out_flow.append(daily_out)

    return pd.DataFrame({"date": business_days, "in": in_flow, "out": out_flow})


# Generate the business day data
business_day_data = simulate_fund_flow(business_days)

# Merge with the full date range, filling non-business days with 0
full_data = data.merge(business_day_data, on="date", how="left", suffixes=("_orig", ""))
full_data["in"] = full_data["in"].fillna(0)
full_data["out"] = full_data["out"].fillna(0)

# Add a 'clus' column with a constant value for simplicity
full_data["clus"] = "clus"

# Save to a CSV file
full_data.to_csv("fund_flow_simulated.csv", index=False)

# Plot the data for visualization
plt.figure(figsize=(14, 7))
plt.plot(full_data["date"], full_data["in"], label="Inflow", alpha=0.7)
plt.plot(full_data["date"], full_data["out"], label="Outflow", alpha=0.7)
plt.title("Simulated Fund Flow (2023-2024)")
plt.xlabel("Date")
plt.ylabel("Amount")
plt.legend()
plt.show()
