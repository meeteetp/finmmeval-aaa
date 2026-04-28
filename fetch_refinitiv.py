"""
Fetch historical analyst data for TSLA from Refinitiv Eikon.

Prerequisites:
  1. Refinitiv Eikon or Workspace must be RUNNING on this machine (the python
     library connects to the local desktop app, not directly to a server).
  2. Get your App Key from Eikon: type "APPKEY" in the Eikon search bar,
     register a new app (Eikon Data API), and copy the key.
  3. pip install eikon pandas

Output: refinitiv_tsla.csv in this folder, one row per business day.
"""

import os
import sys
import pandas as pd
import eikon as ek

# ----- CONFIG -----
APP_KEY = os.environ.get("EIKON_APP_KEY", "PASTE_YOUR_APP_KEY_HERE")
TICKER = "TSLA.O"  # .O = Nasdaq RIC for TSLA
START_DATE = "2025-07-01"
END_DATE = "2026-04-28"
OUTPUT_CSV = "refinitiv_tsla.csv"

# Fields to pull (TR.* are Refinitiv data items)
FIELDS = [
    "TR.EPSMean(Period=FQ1)",                # mean EPS estimate, next fiscal quarter
    "TR.EPSMean(Period=FY1)",                # mean EPS estimate, current fiscal year
    "TR.RevenueMean(Period=FQ1)",            # mean revenue estimate, FQ1
    "TR.PriceTargetMean",                    # mean analyst target price
    "TR.PriceTargetHigh",
    "TR.PriceTargetLow",
    "TR.RecMean",                            # mean recommendation (1=StrongBuy ... 5=StrongSell)
    "TR.NumberOfAnalysts",
    "TR.NumberOfStrongBuyRecommendations",
    "TR.NumberOfBuyRecommendations",
    "TR.NumberOfHoldRecommendations",
    "TR.NumberOfSellRecommendations",
    "TR.NumberOfStrongSellRecommendations",
]


def main():
    if APP_KEY == "PASTE_YOUR_APP_KEY_HERE":
        sys.exit("ERROR: set EIKON_APP_KEY env var or edit APP_KEY in this file.")

    ek.set_app_key(APP_KEY)
    print(f"Fetching {TICKER} analyst data from {START_DATE} to {END_DATE} ...")

    # Daily point-in-time pull
    df, err = ek.get_data(
        instruments=[TICKER],
        fields=FIELDS,
        parameters={
            "SDate": START_DATE,
            "EDate": END_DATE,
            "Frq": "D",   # daily; switch to "W" for weekly if rate-limited
        },
    )

    if err:
        print("Warnings/errors from Refinitiv:")
        print(err)

    if df is None or df.empty:
        sys.exit("No data returned. Check the ticker, date range, and your Eikon login.")

    # Clean up: lowercase column names, parse dates
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Refinitiv usually returns a 'date' column in this query mode
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_CSV}")
    print(df.head())


if __name__ == "__main__":
    main()
