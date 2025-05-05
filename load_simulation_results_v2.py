import sqlite3
import pandas as pd  # Used for easy data handling and display
import numpy as np  # Needed for percentile calculations in display
from datetime import datetime

DB_FILE = "retirement_sim.db"  # Database file created by V4 simulator


def list_runs(db_file):
    """Lists available simulation runs from the database."""
    print("\n--- Available Simulation Runs ---")
    try:
        conn = sqlite3.connect(db_file)
        query = """
        SELECT RunID, Timestamp, InitialPortfolioValue, NumberOfPaths, Notes
        FROM SimulationRuns
        ORDER BY Timestamp DESC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if not df.empty:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            df["InitialPortfolioValue"] = df["InitialPortfolioValue"].map(
                "${:,.0f}".format
            )
            print(df.to_string(index=False))
            return True
        else:
            print(f"No simulation runs found in {db_file}.")
            return False
    except (sqlite3.Error, pd.io.sql.DatabaseError) as e:
        print(f"Database error listing runs: {e}")
        print("(Has the simulation script been run successfully yet?)")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while listing runs: {e}")
        return False


def load_run_data(db_file, run_id):
    """Loads all data associated with a specific RunID from the database."""
    print(f"\n--- Loading data for RunID: {run_id} ---")
    loaded_data = {}
    try:
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row

        # 1. Load Parameters
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM SimulationRuns WHERE RunID = ?", (run_id,))
        params_row = cursor.fetchone()
        if params_row is None:
            print(f"Error: RunID {run_id} not found in SimulationRuns.")
            conn.close()
            return None
        loaded_data["params"] = dict(params_row)
        print(" Parameters loaded.")

        # 2. Load Income Streams into Pandas DataFrame
        query_income = """
        SELECT IncomeType, StartYearOffset, EndYearOffset, InitialAmount, InflationAdjustmentRule
        FROM IncomeStreams
        WHERE RunID = ?
        ORDER BY StartYearOffset, IncomeType
        """
        loaded_data["income_streams"] = pd.read_sql_query(
            query_income, conn, params=(run_id,)
        )
        if loaded_data["income_streams"].empty:
            print(" No income streams found for this RunID.")
        else:
            print(
                f" Income streams loaded ({len(loaded_data['income_streams'])} streams)."
            )

        # 3. Load Expense Streams into Pandas DataFrame
        query_expense = """
        SELECT ExpenseType, StartYearOffset, EndYearOffset, InitialAmount, InflationAdjustmentRule
        FROM Expenses
        WHERE RunID = ?
        ORDER BY StartYearOffset, ExpenseType
        """
        loaded_data["expense_streams"] = pd.read_sql_query(
            query_expense, conn, params=(run_id,)
        )
        if loaded_data["expense_streams"].empty:
            print(" No expense streams found for this RunID.")
        else:
            print(
                f" Expense streams loaded ({len(loaded_data['expense_streams'])} streams)."
            )

        # 4. Load Path Summaries into Pandas DataFrame
        query_results = """
        SELECT PathNumber, SolvedInitialWithdrawal, Status, FinalBalance_Nominal,
               FinalBalance_TodayDollars, IsKeyPercentilePath, PercentileRank
        FROM PathResults
        WHERE RunID = ?
        ORDER BY PathNumber
        """
        loaded_data["path_summaries"] = pd.read_sql_query(
            query_results, conn, params=(run_id,)
        )
        if loaded_data["path_summaries"].empty:
            print(" Warning: No path summary results found for this RunID.")
        else:
            print(
                f" Path summaries loaded ({len(loaded_data['path_summaries'])} paths)."
            )

        # 5. Load Key Path Yearly Data into Pandas DataFrame
        query_yearly = """
        SELECT pr.PercentileRank, pyd.*
        FROM PathYearlyData pyd
        JOIN PathResults pr ON pyd.ResultID = pr.ResultID
        WHERE pr.RunID = ? AND pr.IsKeyPercentilePath = 1
        ORDER BY pr.PercentileRank, pyd.YearNum
        """
        loaded_data["key_paths_yearly"] = pd.read_sql_query(
            query_yearly, conn, params=(run_id,)
        )
        if loaded_data["key_paths_yearly"].empty:
            print(" No detailed yearly data found for key paths for this RunID.")
        else:
            print(
                f" Key path yearly data loaded ({len(loaded_data['key_paths_yearly'])} rows)."
            )

        conn.close()
        return loaded_data

    except sqlite3.Error as e:
        print(f"Database error loading run data for RunID {run_id}: {e}")
        if conn:
            conn.close()
        return None
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        if conn:
            conn.close()
        return None


def display_run_data(run_data):
    """Displays the loaded simulation run data in a readable format."""

    # 1. Display Parameters
    print("\n------ Simulation Parameters ------")
    params = run_data["params"]
    if not params:
        print("No parameters loaded.")
        return

    param_display_order = [  # Removed Base parameters
        "RunID",
        "Timestamp",
        "InitialPortfolioValue",
        "SimulationHorizonYears",
        "TargetEndBalance_Base_TodayDollars",
        "TargetEndBalance_Range_TodayDollars",
        "ReturnMean",
        "ReturnStdDev",
        "InflationMean",
        "InflationStdDev",
        "WithdrawalSolverMinPct",
        "WithdrawalSolverMaxPct",
        "NumberOfPaths",
        "KeyPercentilesStored",
        "Notes",
    ]
    for key in param_display_order:
        value = params.get(key)
        if value is None:
            continue
        label = key.replace("_", " ").title()
        if isinstance(value, float):
            if "Pct" in key:
                formatted_value = f"{value*100:.2f}%"
            elif "Value" in key or "Dollars" in key:
                formatted_value = f"${value:,.0f}"
            else:
                formatted_value = f"{value:.4f}"
        elif key == "Timestamp":
            formatted_value = pd.to_datetime(value).strftime("%Y-%m-%d %H:%M:%S")
        else:
            formatted_value = str(value)
        print(f" {label:<35}: {formatted_value}")

    # 2. Display Income Streams
    print("\n------ Income Streams ------")
    income_streams_df = run_data.get("income_streams")
    if income_streams_df is not None and not income_streams_df.empty:
        # Format the InitialAmount column for display
        income_streams_df_display = income_streams_df.copy()
        income_streams_df_display["InitialAmount"] = income_streams_df_display[
            "InitialAmount"
        ].map("${:,.0f}".format)
        print(income_streams_df_display.to_string(index=False))
    else:
        print("No income streams loaded for this run.")

    # 3. Display Expense Streams
    print("\n------ Expense Streams ------")
    expense_streams_df = run_data.get("expense_streams")
    if expense_streams_df is not None and not expense_streams_df.empty:
        # Format the InitialAmount column for display
        expense_streams_df_display = expense_streams_df.copy()
        expense_streams_df_display["InitialAmount"] = expense_streams_df_display[
            "InitialAmount"
        ].map("${:,.0f}".format)
        print(expense_streams_df_display.to_string(index=False))
    else:
        print("No expense streams loaded for this run.")

    # 4. Display Summary Statistics
    print("\n------ Simulation Results Summary ------")
    path_summaries = run_data["path_summaries"]
    if path_summaries.empty:
        print("No path summary data loaded.")
    else:
        valid_summaries = path_summaries.dropna(subset=["SolvedInitialWithdrawal"])
        solved_w_values = valid_summaries["SolvedInitialWithdrawal"].values
        statuses = path_summaries["Status"].tolist()
        num_converged = statuses.count("Success") + statuses.count(
            "ConvergedOutsideRange"
        )
        num_bound_hits = statuses.count("HitLowerBound") + statuses.count(
            "HitUpperBound"
        )
        num_solver_errors = statuses.count("SolverError")

        print(f"Number of paths simulated: {len(path_summaries)}")
        print(f"Number converged (Success or Outside Range): {num_converged}")
        print(f"Number hitting solver bounds: {num_bound_hits}")
        print(f"Number of solver errors: {num_solver_errors}")

        if len(solved_w_values) > 0:
            print("\nDistribution of Solved Initial *Discretionary* Withdrawal (W):")
            key_percentiles_str = params.get("KeyPercentilesStored", "10,25,50,75,90")
            try:
                key_percentiles = [int(p) for p in key_percentiles_str.split(",")]
            except ValueError:
                key_percentiles = [10, 25, 50, 75, 90]
                print("Warning: Using default percentiles.")

            percentiles_w = np.percentile(solved_w_values, key_percentiles)

            # Recalculate first year expense/income from loaded streams for context
            first_year_income = 0
            if income_streams_df is not None:
                first_year_income = income_streams_df[
                    income_streams_df["StartYearOffset"] == 0
                ]["InitialAmount"].sum()
            first_year_expense = 0
            if expense_streams_df is not None:
                first_year_expense = expense_streams_df[
                    expense_streams_df["StartYearOffset"] == 0
                ]["InitialAmount"].sum()

            initial_portfolio = params["InitialPortfolioValue"]

            for p, val in zip(key_percentiles, percentiles_w):
                initial_total_spend = val + first_year_expense - first_year_income
                initial_total_spend_pct = (
                    (initial_total_spend / initial_portfolio) * 100
                    if initial_portfolio
                    else 0
                )
                w_pct = (
                    (val / initial_portfolio) * 100
                    if initial_portfolio and not np.isnan(val)
                    else np.nan
                )
                print(
                    f"  {p:3d}th Perc. Discr W: ${val:10,.0f} ({w_pct:5.2f}%) | Approx Initial Total Spend Pct: {initial_total_spend_pct:5.2f}%"
                )

            print(f"  Mean Solved Discr W:   ${np.mean(solved_w_values):10,.0f}")
            print(f"  Median Solved Discr W: ${np.median(solved_w_values):10,.0f}")
        else:
            print("\nNo valid solved withdrawal values found to calculate statistics.")

    # 5. Display Key Path Yearly Data (Interactive)
    yearly_data = run_data["key_paths_yearly"]
    if not yearly_data.empty:
        if "PercentileRank" in yearly_data.columns:
            available_percentiles = sorted(
                [int(p) for p in yearly_data["PercentileRank"].unique()]
            )
            print(
                f"\n------ Yearly Data Available for Key Percentiles: {available_percentiles} ------"
            )
            while True:
                try:
                    choice = (
                        input(
                            f"Enter percentile to view (e.g., {available_percentiles[0]}) or 'q' to quit: "
                        )
                        .strip()
                        .lower()
                    )
                    if choice == "q" or choice == "":
                        break
                    chosen_p = int(choice)
                    if chosen_p in available_percentiles:
                        print(f"\n--- Yearly Data for {chosen_p}th Percentile Path ---")
                        percentile_df = yearly_data[
                            yearly_data["PercentileRank"] == chosen_p
                        ].copy()
                        display_cols = [  # Updated columns based on V4 save
                            "YearNum",
                            "StartOfYearBalance",
                            "TotalIncome",
                            "TotalExpense",
                            "DiscretionaryWithdrawal",
                            "MarketReturn",
                            "Inflation",
                            "EndOfYearBalance",
                        ]
                        actual_cols = [
                            col for col in display_cols if col in percentile_df.columns
                        ]
                        missing_cols = [
                            col
                            for col in display_cols
                            if col not in percentile_df.columns
                        ]
                        if missing_cols:
                            print(f"Warning: Missing expected columns: {missing_cols}")
                        display_df = percentile_df[actual_cols].copy()

                        # Print table header
                        headers = {
                            "YearNum": " Year ",
                            "StartOfYearBalance": " Start Balance ",
                            "TotalIncome": " Tot Income ",
                            "TotalExpense": " Tot Expense",
                            "DiscretionaryWithdrawal": " Discr. WD ",
                            "MarketReturn": " Mkt Ret ",
                            "Inflation": " Inflation ",
                            "EndOfYearBalance": " End Balance ",
                        }
                        col_widths = {
                            "YearNum": 5,
                            "StartOfYearBalance": 13,
                            "TotalIncome": 10,
                            "TotalExpense": 10,
                            "DiscretionaryWithdrawal": 9,
                            "MarketReturn": 7,
                            "Inflation": 7,
                            "EndOfYearBalance": 13,
                        }
                        header_line = "|".join(
                            [
                                headers.get(col, col).center(col_widths.get(col, 10))
                                for col in actual_cols
                            ]
                        )
                        separator_line = "-".join(
                            ["-" * col_widths.get(col, 10) for col in actual_cols]
                        )
                        print(header_line)
                        print(separator_line)
                        # Print rows with formatting
                        for index, row in display_df.iterrows():
                            row_items = []
                            for col in actual_cols:
                                val = row.get(col)
                                width = col_widths.get(col, 10)
                                fmt_str = ""
                                try:
                                    if pd.isna(val):
                                        formatted_val = "N/A".center(width)
                                    elif col == "YearNum":
                                        formatted_val = f"{int(val):<{width}d}"
                                    elif col in ["MarketReturn", "Inflation"]:
                                        formatted_val = f"{val*100:{width-1}.2f}%"
                                    elif isinstance(val, (int, float)):
                                        formatted_val = f"${val:{width-1},.0f}"
                                    else:
                                        formatted_val = f"{str(val):<{width}}"
                                except (ValueError, TypeError):
                                    formatted_val = "FmtErr".center(width)
                                row_items.append(formatted_val)
                            print("|".join(row_items))
                        print("-" * len(separator_line))  # Footer separator
                    else:
                        print(
                            f"Invalid percentile. Choose from {available_percentiles} or 'q'."
                        )
                except ValueError:
                    print("Invalid input. Please enter a number or 'q'.")
                except KeyboardInterrupt:
                    print("\nExiting viewer.")
                    break
        else:
            print("\n'PercentileRank' column not found in yearly data.")
    else:
        print("\nNo detailed yearly path data found for this run.")


# --- Main Execution ---
if __name__ == "__main__":
    # List available runs first
    if not list_runs(DB_FILE):
        print("\nExiting - No runs found or database issue.")
    else:
        selected_run_id = None
        while selected_run_id is None:
            try:
                run_id_input = (
                    input("\nEnter the RunID you want to load (or 'q' to quit): ")
                    .strip()
                    .lower()
                )
                if run_id_input == "q":
                    break
                selected_run_id = int(run_id_input)
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nExiting.")
                selected_run_id = "q"
                break

        if selected_run_id != "q" and selected_run_id is not None:
            loaded_data = load_run_data(DB_FILE, selected_run_id)
            if loaded_data:
                import numpy as np  # Ensure numpy is available

                display_run_data(loaded_data)

    print("\nLoader finished.")
