import sqlite3
import pandas as pd  # Used for easy data handling and display
import numpy as np  # Needed for percentile calculations in display
from datetime import datetime

DB_FILE = "retirement_sim.db"  # Database file created by V3 script


def list_runs(db_file):
    """Lists available simulation runs from the database."""
    print("\n--- Available Simulation Runs ---")
    try:
        conn = sqlite3.connect(db_file)
        # Use pandas to read and display the table easily
        query = """
        SELECT RunID, Timestamp, InitialPortfolioValue, NumberOfPaths, Notes
        FROM SimulationRuns
        ORDER BY Timestamp DESC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if not df.empty:
            # Format timestamp for better readability if needed
            df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            # Format currency
            df["InitialPortfolioValue"] = df["InitialPortfolioValue"].map(
                "${:,.0f}".format
            )
            print(df.to_string(index=False))  # Print DataFrame without index column
            return True
        else:
            print(f"No simulation runs found in {db_file}.")
            return False
    except (sqlite3.Error, pd.io.sql.DatabaseError) as e:
        # Handle cases where the table might not exist or other DB errors
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
        # Use Row factory to access columns by name easily
        conn.row_factory = sqlite3.Row

        # 1. Load Parameters from SimulationRuns
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM SimulationRuns WHERE RunID = ?", (run_id,))
        params_row = cursor.fetchone()

        if params_row is None:
            print(f"Error: RunID {run_id} not found in SimulationRuns table.")
            conn.close()
            return None
        loaded_data["params"] = dict(params_row)  # Convert Row object to dictionary
        print(" Parameters loaded.")

        # 2. Load Path Summaries into a Pandas DataFrame
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

        # 3. Load Key Path Yearly Data into a Pandas DataFrame
        query_yearly = """
        SELECT pr.PercentileRank, pyd.*
        FROM PathYearlyData pyd
        JOIN PathResults pr ON pyd.ResultID = pr.ResultID
        WHERE pr.RunID = ? AND pr.IsKeyPercentilePath = 1
        ORDER BY pr.PercentileRank, pyd.YearNum
        """
        # Use ResultID from PathYearlyData table definition
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

    # Format and print parameters nicely
    param_display_order = [
        "RunID",
        "Timestamp",
        "InitialPortfolioValue",
        "SimulationHorizonYears",
        "TargetEndBalance_Base_TodayDollars",
        "TargetEndBalance_Range_TodayDollars",
        "BaseSSIncomeToday",
        "BaseLivingExpenseToday",
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
        value = params.get(
            key
        )  # Use .get to handle potentially missing keys gracefully
        if value is None:
            continue  # Skip if parameter wasn't present

        label = key.replace("_", " ").title()  # Make label readable
        # Apply specific formatting based on expected data type or name patterns
        if isinstance(value, float):
            if "Pct" in key:
                formatted_value = f"{value*100:.2f}%"
            elif (
                "Value" in key
                or "Dollars" in key
                or "Income" in key
                or "Expense" in key
            ):
                formatted_value = f"${value:,.0f}"
            else:
                formatted_value = f"{value:.4f}"  # Default float format
        elif key == "Timestamp":
            formatted_value = pd.to_datetime(value).strftime("%Y-%m-%d %H:%M:%S")
        else:
            formatted_value = str(value)

        print(f" {label:<35}: {formatted_value}")

    # 2. Display Summary Statistics (calculated from loaded path summaries)
    print("\n------ Simulation Results Summary ------")
    path_summaries = run_data["path_summaries"]
    if path_summaries.empty:
        print("No path summary data loaded.")
        return  # Can't proceed further

    valid_summaries = path_summaries.dropna(subset=["SolvedInitialWithdrawal"])
    solved_w_values = valid_summaries["SolvedInitialWithdrawal"].values

    statuses = path_summaries["Status"].tolist()
    num_converged = statuses.count("Success") + statuses.count("ConvergedOutsideRange")
    num_bound_hits = statuses.count("HitLowerBound") + statuses.count("HitUpperBound")
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
            print(
                f"Warning: Could not parse KeyPercentilesStored ('{key_percentiles_str}'). Using defaults."
            )
            key_percentiles = [10, 25, 50, 75, 90]

        percentiles_w = np.percentile(solved_w_values, key_percentiles)

        initial_portfolio = params["InitialPortfolioValue"]
        expense_today = (
            params.get("BaseLivingExpenseToday", 0) or 0
        )  # Handle None from DB
        ss_today = params.get("BaseSSIncomeToday", 0) or 0  # Handle None from DB

        # Print percentile results
        for p, val in zip(key_percentiles, percentiles_w):
            initial_total_spend = val + expense_today - ss_today
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

    # 3. Display Key Path Yearly Data (Interactive)
    yearly_data = run_data["key_paths_yearly"]
    if not yearly_data.empty:
        # Ensure PercentileRank column exists before accessing .unique()
        if "PercentileRank" in yearly_data.columns:
            available_percentiles = sorted(yearly_data["PercentileRank"].unique())
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

                        # Select and format columns for display
                        display_cols = [
                            "YearNum",
                            "StartOfYearBalance",
                            "SSIncome",
                            "LivingExpense",
                            "DiscretionaryWithdrawal",
                            "MarketReturn",
                            "Inflation",
                            "EndOfYearBalance",
                        ]
                        # Check if all expected columns exist
                        missing_cols = [
                            col
                            for col in display_cols
                            if col not in percentile_df.columns
                        ]
                        if missing_cols:
                            print(
                                f"Warning: Missing expected columns in yearly data: {missing_cols}"
                            )
                            print(
                                percentile_df.to_string(index=False)
                            )  # Print what we have
                            continue

                        display_df = percentile_df[display_cols].copy()

                        # Apply formatting for display using standard Python f-strings
                        print(" Year | Start Balance | SS Income | Living Exp| Discr. WD | Mkt Ret | Inflation | End Balance ")
                        rint("------|---------------|-----------|-----------|-----------|---------|-----------|---------------")
                         for index, row in display_df.iterrows():
                             try:
                                 # Format numerical values, explicitly check for NaN before formatting percentages
                                 ret_val = row['MarketReturn']
                                 inf_val = row['Inflation']
                                 ret_str = f"{ret_val*100:>6.2f}%" if not pd.isna(ret_val) else "  N/A  "
                                 inf_str = f"{inf_val*100:>6.2f}%" if not pd.isna(inf_val) else "  N/A  "

                                 # Use .get with default for safety, format safely
                                 start_bal_str = f"${row.get('StartOfYearBalance', 0):<13,.0f}"
                                 ss_str = f"${row.get('SSIncome', 0):<9,.0f}"
                                 exp_str = f"${row.get('LivingExpense', 0):<9,.0f}"
                                 wd_str = f"${row.get('DiscretionaryWithdrawal', 0):<9,.0f}"
                                 end_bal_str = f"${row.get('EndOfYearBalance', 0):<13,.0f}"
                                 year_str = f"{int(row.get('YearNum', 0)):<5d}"


                                 print(f" {year_str}| "
                                       f"{start_bal_str}| "
                                       f"{ss_str}| "
                                       f"{exp_str}| "
                                       f"{wd_str}| "
                                       f"{ret_str}| "
                                       f"{inf_str}| "
                                       f"{end_bal_str}")
                             except (ValueError, TypeError) as fmt_e:
                                 # Catch specific formatting errors if NaN checks weren't enough
                                 print(f" FmtErr | Row {index+1} data could not be formatted cleanly: {fmt_e}")
                                 # Optionally print raw row data for debugging: print(row.to_dict())
                         print("-" * 100) # Separator

                    else:
                        print(
                            f"Invalid percentile. Please choose from {available_percentiles} or 'q'."
                        )
                except ValueError:
                    print("Invalid input. Please enter a number or 'q'.")
                except KeyboardInterrupt:
                    print("\nExiting.")
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
                # Prompt user for input
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
                break  # Exit loop on Ctrl+C

        # Proceed if a valid RunID was entered
        if selected_run_id is not None:
            loaded_data = load_run_data(DB_FILE, selected_run_id)

            if loaded_data:
                # Need numpy for calculations inside display_run_data
                import numpy as np

                display_run_data(loaded_data)
            else:
                print(f"Failed to load data for RunID {selected_run_id}.")

    print("\nLoader finished.")
