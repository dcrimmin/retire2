# simulator_engine.py (V5.3 - Correct Total Withdrawal Logic & Standardized Keys)

import numpy as np
from scipy.optimize import brentq
import time
import sqlite3
from datetime import datetime
import pandas as pd # Needed for loading check consistency
import traceback # For better error logging

# --- Default Parameters / Constants ---
SIMULATION_HORIZON_YEARS_DEF = 34
DB_FILE = 'retirement_sim.db'
# *** Using Standardized Keys Now ***
DEFAULT_INCOME_STREAMS = [
    {'IncomeType': 'SS_User',   'StartYearOffset': 4, 'EndYearOffset': SIMULATION_HORIZON_YEARS_DEF, 'InitialAmount': 30000, 'InflationAdjustmentRule': 'General'},
    {'IncomeType': 'SS_Spouse', 'StartYearOffset': 9, 'EndYearOffset': SIMULATION_HORIZON_YEARS_DEF, 'InitialAmount': 20000, 'InflationAdjustmentRule': 'General'}
]
DEFAULT_EXPENSE_STREAMS = [
    {'ExpenseType': 'Living', 'StartYearOffset': 0, 'EndYearOffset': SIMULATION_HORIZON_YEARS_DEF, 'InitialAmount': 60000, 'InflationAdjustmentRule': 'General'},
    {'ExpenseType': 'Travel', 'StartYearOffset': 2, 'EndYearOffset': 12, 'InitialAmount': 15000, 'InflationAdjustmentRule': 'General'},
    {'ExpenseType': 'Medicare','StartYearOffset': 2, 'EndYearOffset': SIMULATION_HORIZON_YEARS_DEF, 'InitialAmount': 7000,  'InflationAdjustmentRule': 'General'} # Renamed key for consistency
]
KEY_PERCENTILES = [10, 25, 50, 75, 90]
# Using standardized keys for default bounds too
TOTAL_WITHDRAWAL_SOLVER_MIN_PCT = 0.02
TOTAL_WITHDRAWAL_SOLVER_MAX_PCT = 0.20

# --- Database Functions ---

def setup_database(db_file):
    """Creates/updates database tables."""
    conn = None
    try:
        conn = sqlite3.connect(db_file); cursor = conn.cursor()
        # SimulationRuns Table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS SimulationRuns (
            RunID INTEGER PRIMARY KEY AUTOINCREMENT, Timestamp DATETIME NOT NULL,
            InitialPortfolioValue REAL NOT NULL, SimulationHorizonYears INTEGER NOT NULL,
            TargetEndBalance_Base_TodayDollars REAL NOT NULL, TargetEndBalance_Range_TodayDollars REAL NOT NULL,
            ReturnMean REAL NOT NULL, ReturnStdDev REAL NOT NULL, InflationMean REAL NOT NULL, InflationStdDev REAL NOT NULL,
            LongtermInflationMean REAL, LongtermInflationStdDev REAL,
            WithdrawalSolverMinPct REAL NOT NULL, WithdrawalSolverMaxPct REAL NOT NULL,
            NumberOfPaths INTEGER NOT NULL, KeyPercentilesStored TEXT, Notes TEXT )''')
        # --- Add Longterm Inflation columns if they don't exist (safe to run multiple times) ---
        cursor.execute("PRAGMA table_info(SimulationRuns)")
        run_columns = [info[1] for info in cursor.fetchall()]
        if 'LongtermInflationMean' not in run_columns:
            cursor.execute("ALTER TABLE SimulationRuns ADD COLUMN LongtermInflationMean REAL")
        if 'LongtermInflationStdDev' not in run_columns:
            cursor.execute("ALTER TABLE SimulationRuns ADD COLUMN LongtermInflationStdDev REAL")
        # -------------------------------------------------------------------------------------
        # PathResults Table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS PathResults ( ResultID INTEGER PRIMARY KEY AUTOINCREMENT, RunID INTEGER NOT NULL, PathNumber INTEGER NOT NULL,
            IsKeyPercentilePath INTEGER NOT NULL DEFAULT 0, PercentileRank INTEGER, SolvedInitialWithdrawal REAL, Status TEXT NOT NULL,
            FinalBalance_Nominal REAL, CumulativeInflation REAL, FinalBalance_TodayDollars REAL,
            FOREIGN KEY (RunID) REFERENCES SimulationRuns (RunID) ON DELETE CASCADE )''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pathresults_runid ON PathResults (RunID)')
        # PathYearlyData Table Schema Update Logic
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS PathYearlyData ( YearlyDataID INTEGER PRIMARY KEY AUTOINCREMENT, ResultID INTEGER NOT NULL, YearNum INTEGER NOT NULL,
            StartOfYearBalance REAL, TotalIncome REAL, TotalExpense REAL, TotalWithdrawal REAL, MarketReturn REAL, Inflation REAL, EndOfYearBalance REAL,
            FOREIGN KEY (ResultID) REFERENCES PathResults (ResultID) ON DELETE CASCADE )''')
        cursor.execute("PRAGMA table_info(PathYearlyData)"); columns = [info[1] for info in cursor.fetchall()]
        if 'DiscretionaryWithdrawal' in columns and 'TotalWithdrawal' not in columns:
             try: cursor.execute("ALTER TABLE PathYearlyData RENAME COLUMN DiscretionaryWithdrawal TO TotalWithdrawal"); print("Renamed PathYearlyData column.")
             except sqlite3.OperationalError:
                  if 'TotalWithdrawal' not in columns: cursor.execute("ALTER TABLE PathYearlyData ADD COLUMN TotalWithdrawal REAL"); print("Added TotalWithdrawal column.")
        elif 'TotalWithdrawal' not in columns: cursor.execute("ALTER TABLE PathYearlyData ADD COLUMN TotalWithdrawal REAL")
        if 'TotalIncome' not in columns: cursor.execute("ALTER TABLE PathYearlyData ADD COLUMN TotalIncome REAL")
        if 'TotalExpense' not in columns: cursor.execute("ALTER TABLE PathYearlyData ADD COLUMN TotalExpense REAL")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pathyearlydata_resultid ON PathYearlyData (ResultID)')
        # IncomeStreams Table (Using standardized keys where possible in definition)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS IncomeStreams ( IncomeID INTEGER PRIMARY KEY AUTOINCREMENT, RunID INTEGER NOT NULL, IncomeType TEXT, StartYearOffset INTEGER, EndYearOffset INTEGER,
            InitialAmount REAL, AmountType TEXT, InflationAdjustmentRule TEXT, TaxabilityRule TEXT, OtherParameters TEXT,
            FOREIGN KEY (RunID) REFERENCES SimulationRuns (RunID) ON DELETE CASCADE )''')
        # Expenses Table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS Expenses ( ExpenseID INTEGER PRIMARY KEY AUTOINCREMENT, RunID INTEGER NOT NULL, ExpenseType TEXT, StartYearOffset INTEGER, EndYearOffset INTEGER,
            InitialAmount REAL, InflationAdjustmentRule TEXT, OtherParameters TEXT,
            FOREIGN KEY (RunID) REFERENCES SimulationRuns (RunID) ON DELETE CASCADE )''')
        conn.commit(); print(f"Database {db_file} checked/initialized.")
    except sqlite3.Error as e: print(f"Database setup error: {e}")
    finally:
        if conn: conn.close()

def save_simulation_run(db_file, params, income_streams, expense_streams, results, key_paths_indices, key_paths_yearly_data, notes=""):
    """Saves simulation - uses standardized keys for streams."""
    conn = None; run_id = -1
    try:
        conn = sqlite3.connect(db_file); cursor = conn.cursor(); run_timestamp = datetime.now()
        # 1. Insert SimulationRuns
        run_sql = '''INSERT INTO SimulationRuns ( Timestamp, InitialPortfolioValue, SimulationHorizonYears, TargetEndBalance_Base_TodayDollars, TargetEndBalance_Range_TodayDollars,
                       ReturnMean, ReturnStdDev, InflationMean, InflationStdDev, 
                       LongtermInflationMean, LongtermInflationStdDev,
                       WithdrawalSolverMinPct, WithdrawalSolverMaxPct, NumberOfPaths, KeyPercentilesStored, Notes )
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
        run_params_tuple = ( run_timestamp, params['initial_portfolio'], params['horizon'], params['target_today'], params['target_range_today'], params['ret_mean'], params['ret_sd'],
                             params['inf_mean'], params['inf_sd'],
                             params.get('longterm_inf_mean', None),
                             params.get('longterm_inf_sd', None),
                             params['total_min_pct'], params['total_max_pct'], params['num_paths'], ','.join(map(str, sorted(key_paths_indices.values()))) if key_paths_indices else "", notes )
        cursor.execute(run_sql, run_params_tuple); run_id = cursor.lastrowid; print(f"Saved RunID: {run_id}")
        # 2. Insert Income Streams (using standardized keys)
        income_sql = '''INSERT INTO IncomeStreams (RunID, IncomeType, StartYearOffset, EndYearOffset, InitialAmount, InflationAdjustmentRule) VALUES (?, ?, ?, ?, ?, ?)'''
        income_data_tuples=[(run_id, s.get('IncomeType'), s.get('StartYearOffset'), s.get('EndYearOffset'), s.get('InitialAmount'), s.get('InflationAdjustmentRule')) for s in income_streams]
        if income_data_tuples: cursor.executemany(income_sql, income_data_tuples); print(f"Inserted {len(income_data_tuples)} income streams.")
        # 3. Insert Expense Streams (using standardized keys)
        expense_sql = '''INSERT INTO Expenses (RunID, ExpenseType, StartYearOffset, EndYearOffset, InitialAmount, InflationAdjustmentRule) VALUES (?, ?, ?, ?, ?, ?)'''
        expense_data_tuples=[(run_id, s.get('ExpenseType'), s.get('StartYearOffset'), s.get('EndYearOffset'), s.get('InitialAmount'), s.get('InflationAdjustmentRule')) for s in expense_streams]
        if expense_data_tuples: cursor.executemany(expense_sql, expense_data_tuples); print(f"Inserted {len(expense_data_tuples)} expense streams.")
        # 4/5. Insert PathResults & Get Key ResultIDs (unchanged)
        path_results_data = []; key_path_result_ids = {}
        for r in results:
            path_index=r['path_index']; is_key=path_index in key_paths_indices; percentile=key_paths_indices.get(path_index, None)
            path_data_tuple = (run_id, path_index, 1 if is_key else 0, percentile, r['solved_w'], r['status'], r['final_balance_nominal'], r['cumulative_inflation'], r['final_balance_today_dollars'])
            path_results_data.append(path_data_tuple);
            if is_key: key_path_result_ids[path_index] = None
        results_sql = '''INSERT INTO PathResults (RunID, PathNumber, IsKeyPercentilePath, PercentileRank, SolvedInitialWithdrawal, Status, FinalBalance_Nominal, CumulativeInflation, FinalBalance_TodayDollars) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)'''
        cursor.executemany(results_sql, path_results_data); print(f"Inserted {len(path_results_data)} path results.")
        if key_path_result_ids:
             q_sql = "SELECT ResultID, PathNumber FROM PathResults WHERE RunID = ? AND PathNumber IN ({})"
             p_holders=','.join('?'*len(key_path_result_ids)); idx_list=list(key_path_result_ids.keys())
             cursor.execute(q_sql.format(p_holders), [run_id]+idx_list); fetched=cursor.fetchall()
             for res_id, p_num in fetched:
                  if p_num in key_path_result_ids: key_path_result_ids[p_num] = res_id
        # 6. Insert into PathYearlyData (Data keyed by percentile, inner keys are DB Col names)
        yearly_data_to_insert = []
        for percentile, yearly_data in key_paths_yearly_data.items():
            path_index_for_p = -1;
            for p_idx, p in key_paths_indices.items():
                if p == percentile: path_index_for_p = p_idx; break
            result_id = key_path_result_ids.get(path_index_for_p)
            if result_id is None: continue
            # Use DB Col names as keys when accessing lists
            num_years = len(yearly_data.get('YearNum',[]))
            for year_idx in range(num_years):
                 def get_list_val(key, default=None): lst = yearly_data.get(key, []); return lst[year_idx] if year_idx < len(lst) else default
                 yearly_row = ( result_id, get_list_val('YearNum'), get_list_val('StartOfYearBalance'), get_list_val('TotalIncome'), get_list_val('TotalExpense'),
                                get_list_val('TotalWithdrawal'), get_list_val('MarketReturn'), get_list_val('Inflation'), get_list_val('EndOfYearBalance') )
                 yearly_data_to_insert.append(yearly_row)
        if yearly_data_to_insert:
            yearly_sql = '''INSERT INTO PathYearlyData (ResultID, YearNum, StartOfYearBalance, TotalIncome, TotalExpense, TotalWithdrawal, MarketReturn, Inflation, EndOfYearBalance) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)'''
            cursor.executemany(yearly_sql, yearly_data_to_insert); print(f"Inserted {len(yearly_data_to_insert)} yearly rows.")
        conn.commit(); print(f"Successfully saved RunID {run_id} to database.")
        return run_id
    except sqlite3.Error as e: print(f"DB error save: {e}"); conn.rollback(); return None
    finally:
        if conn: conn.close()

# --- Simulation Functions ---

def generate_stochastic_data(num_paths, horizon, ret_mean, ret_sd, inf_mean, inf_sd, longterm_inf_mean, longterm_inf_sd):
    """Generates returns and inflations, using different inflation stats after year 10."""
    returns = np.random.normal(ret_mean, ret_sd, size=(num_paths, horizon))

    # Generate inflation in two stages
    short_term_horizon = min(10, horizon)
    long_term_horizon = horizon - short_term_horizon

    inflations_st = np.random.normal(inf_mean, inf_sd, size=(num_paths, short_term_horizon))

    if long_term_horizon > 0:
        inflations_lt = np.random.normal(longterm_inf_mean, longterm_inf_sd, size=(num_paths, long_term_horizon))
        inflations = np.concatenate((inflations_st, inflations_lt), axis=1)
    else:
        inflations = inflations_st

    inflations = np.maximum(inflations, -0.05) # Floor inflation at -5%
    return returns, inflations

def run_single_path(initial_total_withdrawal_W, initial_portfolio, horizon, path_returns, path_inflations,
                    income_streams, expense_streams ):
    """ V5.3: Correct Total Withdrawal logic & Standardized Keys & DB Col Name dict return """
    balance = float(initial_portfolio); cumulative_inflation_factor = 1.0
    # Use standardized keys matching input streams dict structure
    current_income_nominal = {i: s.get('InitialAmount', 0) for i, s in enumerate(income_streams)}
    current_expense_nominal = {i: s.get('InitialAmount', 0) for i, s in enumerate(expense_streams)}
    current_total_withdrawal_nominal = float(initial_total_withdrawal_W)

    # Store yearly data using DB Column Names as keys for consistency
    yearly_data = { 'YearNum': list(range(1, horizon + 1)), 'StartOfYearBalance': [0.0]*horizon, 'TotalIncome': [0.0]*horizon,
                    'TotalExpense': [0.0]*horizon, 'TotalWithdrawal': [0.0]*horizon, 'MarketReturn': [0.0]*horizon,
                    'Inflation': [0.0]*horizon, 'EndOfYearBalance': [0.0]*horizon }
    current_balance = balance

    for year in range(horizon):
        yearly_data['StartOfYearBalance'][year] = current_balance

        # 1. Determine nominal amounts for THIS year (using previous inflation)
        if year > 0:
            inf_prev = path_inflations[year-1]
            for i, s in enumerate(income_streams): # Use descriptive keys
                if s.get('InflationAdjustmentRule', 'General') == 'General': current_income_nominal[i] *= (1 + inf_prev)
            for i, s in enumerate(expense_streams): # Use descriptive keys
                 if s.get('InflationAdjustmentRule', 'General') == 'General': current_expense_nominal[i] *= (1 + inf_prev)
            current_total_withdrawal_nominal *= (1 + inf_prev)

        # Calculate total income/expense active THIS year (for tracking/reporting)
        # Use descriptive keys
        total_income_this_year = sum(max(0.0, current_income_nominal[i]) for i, s in enumerate(income_streams) if s.get('StartYearOffset', 0) <= year < s.get('EndYearOffset', horizon))
        total_expense_this_year = sum(max(0.0, current_expense_nominal[i]) for i, s in enumerate(expense_streams) if s.get('StartYearOffset', 0) <= year < s.get('EndYearOffset', horizon))
        total_withdrawal_this_year = max(0.0, current_total_withdrawal_nominal)

        yearly_data['TotalIncome'][year] = total_income_this_year
        yearly_data['TotalExpense'][year] = total_expense_this_year # Store planned expense

        # *** CORRECTED V5 LOGIC: Income -> Withdraw Total -> Apply Return ***
        start_of_year_balance = current_balance
        balance = current_balance # Start fresh for this year's calc
        balance += total_income_this_year # Add income first
        actual_total_withdrawal_taken = min(total_withdrawal_this_year, balance if balance > 0 else 0.0) # Take total withdrawal
        balance -= actual_total_withdrawal_taken; balance = max(0.0, balance)
        # Expenses are NOT subtracted from balance here, only tracked in yearly_data['TotalExpense']

        yearly_data['TotalWithdrawal'][year] = actual_total_withdrawal_taken # Store ACTUAL total withdrawn

        # 3. Apply Market Return
        market_return = path_returns[year]; effective_return = max(market_return, -1.0)
        balance *= (1 + effective_return); balance = max(0.0, balance)
        yearly_data['MarketReturn'][year] = market_return

        # 4. Update Inflation Factor & Store End Balance
        current_year_inflation = path_inflations[year]; yearly_data['Inflation'][year] = current_year_inflation
        yearly_data['EndOfYearBalance'][year] = balance # Record end balance
        current_balance = balance # Update for next year's start
        cumulative_inflation_factor *= (1 + current_year_inflation)

    final_balance = current_balance
    # Return dict keyed by DB Col Name
    return (final_balance, cumulative_inflation_factor, yearly_data)


def objective_function(trial_total_W, initial_portfolio, horizon, path_returns, path_inflations,
                       income_streams, expense_streams, target_today, target_range_today):
    # Uses corrected run_single_path V5.3
    final_balance, cumulative_inflation_factor, _ = run_single_path(
        trial_total_W, initial_portfolio, horizon, path_returns, path_inflations, income_streams, expense_streams)
    target_nominal_mid = max(0.0, target_today * cumulative_inflation_factor)
    return final_balance - target_nominal_mid

def solve_for_initial_W(path_returns, path_inflations, income_streams, expense_streams, params):
    # Uses standardized keys from params
    initial_portfolio = params['initial_portfolio']; horizon = params['horizon']; target_today = params['target_today']; target_range_today = params['target_range_today']
    min_w_pct = params.get('total_min_pct', params.get('min_pct', TOTAL_WITHDRAWAL_SOLVER_MIN_PCT)); max_w_pct = params.get('total_max_pct', params.get('max_pct', TOTAL_WITHDRAWAL_SOLVER_MAX_PCT))
    min_w = min_w_pct * initial_portfolio; max_w = max_w_pct * initial_portfolio

    try:
        result = brentq( objective_function, min_w, max_w, xtol=10.0, rtol=1e-4,
                         args=(initial_portfolio, horizon, path_returns, path_inflations, income_streams, expense_streams, target_today, target_range_today) )
        # Check if result is very close to bounds
        if np.isclose(result, min_w, atol=1.0): status = "SuccessAtLowerBound"
        elif np.isclose(result, max_w, atol=1.0): status = "SuccessAtUpperBound"
        else: status = "Success"
        return result, status
    except ValueError as e:
        # Check if the objective function has the same sign at both ends
        lower_bound_val = objective_function(min_w, initial_portfolio, horizon, path_returns, path_inflations, income_streams, expense_streams, target_today, target_range_today)
        upper_bound_val = objective_function(max_w, initial_portfolio, horizon, path_returns, path_inflations, income_streams, expense_streams, target_today, target_range_today)
        if np.sign(lower_bound_val) == np.sign(upper_bound_val):
             if abs(lower_bound_val) < abs(upper_bound_val): return min_w, "ConvergedOutsideRange" # Closest to zero at lower bound
             else: return max_w, "ConvergedOutsideRange" # Closest to zero at upper bound
        else:
            # Generic solver error (e.g., max iterations reached)
            return None, "SolverError"
    except Exception as e:
         print(f"Solver Error Path: {e}"); traceback.print_exc()
         return None, "SolverError"

def run_and_save_simulation(params, income_streams, expense_streams, notes=""):
    """Main function: Runs simulation, analyzes, saves. V5.3 Uses updated gen/save funcs."""
    print("--- Running Simulation (Engine V5.3 - Total Withdrawal) ---")
    start_time = time.time(); setup_database(DB_FILE)
    # Ensure new keys are present in params for internal use (provide defaults if missing)
    if 'longterm_inf_mean' not in params: params['longterm_inf_mean'] = params.get('inf_mean')
    if 'longterm_inf_sd' not in params: params['longterm_inf_sd'] = params.get('inf_sd')
    run_id = -1; analysis_results = {'summary_stats': {}, 'solved_w_values': [], 'percentiles_w': {}, 'key_paths_yearly_data': {}}
    params_internal = params.copy(); params_internal['total_min_pct'] = params.get('total_min_pct', params.get('min_pct', TOTAL_WITHDRAWAL_SOLVER_MIN_PCT)); params_internal['total_max_pct'] = params.get('total_max_pct', params.get('max_pct', TOTAL_WITHDRAWAL_SOLVER_MAX_PCT))
    # --- Pass new longterm params to generate_stochastic_data --- 
    returns_data, inflations_data = generate_stochastic_data(
        params_internal['num_paths'], params_internal['horizon'], 
        params_internal['ret_mean'], params_internal['ret_sd'], 
        params_internal['inf_mean'], params_internal['inf_sd'], 
        params_internal['longterm_inf_mean'], params_internal['longterm_inf_sd']
    )
    # -------------------------------------------------------------
    results = []
    print(f"Running solver for {params_internal['num_paths']} paths...")
    for i in range(params_internal['num_paths']):
        params_internal['current_path_index'] = i; path_returns = returns_data[i, :]; path_inflations = inflations_data[i, :]
        solved_total_W, status = solve_for_initial_W( path_returns, path_inflations, income_streams, expense_streams, params_internal)
        
        # --- If solver succeeded, run path again to get final details --- 
        final_bal = np.nan
        cum_inf = np.nan
        if solved_total_W is not None:
            # Rerun the single path with the solved W to get final state
            final_bal, cum_inf, _ = run_single_path(
                solved_total_W, 
                params_internal['initial_portfolio'], 
                params_internal['horizon'], 
                path_returns, 
                path_inflations, 
                income_streams, 
                expense_streams
            )
            # Recalculate final balance in today's dollars
            final_bal_today = final_bal / cum_inf if cum_inf > 0 and not pd.isna(final_bal) else np.nan
            results.append({ 
                'path_index': i, 
                'solved_w': solved_total_W, 
                'status': status, 
                'final_balance_nominal': final_bal, 
                'cumulative_inflation': cum_inf, 
                'final_balance_today_dollars': final_bal_today 
            })
        else:
             # If solver failed, still record the failure
             results.append({
                 'path_index': i, 
                 'solved_w': np.nan, # Use NaN for failed solves
                 'status': status, 
                 'final_balance_nominal': np.nan, 
                 'cumulative_inflation': np.nan, 
                 'final_balance_today_dollars': np.nan
             })
        # --------------------------------------------------------------

    print("--- Analyzing Results ---")
    # --- Filter results based on status != 'SolverError'? Or non-NaN solved_w? ---
    # Filter based on non-NaN solved_w seems safer as status might be complex
    valid_results=[r for r in results if r.get('solved_w') is not None and not np.isnan(r['solved_w'])]; 
    # -------------------------------------------------------------------------
    solved_w_total_values=np.array([r['solved_w'] for r in valid_results])
    key_paths_indices = {}; key_paths_yearly_data_by_percentile = {}; percentile_values_w = {}
    if len(solved_w_total_values) > 0:
        key_percentiles_to_calc = KEY_PERCENTILES; calc_percentile_values = np.percentile(solved_w_total_values, key_percentiles_to_calc)
        percentile_values_w = dict(zip(key_percentiles_to_calc, calc_percentile_values))
        # Regenerate yearly data for key paths
        for p, p_val in percentile_values_w.items():
            closest_path_result = min(valid_results, key=lambda r: abs(r['solved_w'] - p_val)); closest_path_index = closest_path_result['path_index']
            if closest_path_index not in key_paths_indices:
                 key_paths_indices[closest_path_index] = p; key_path_returns = returns_data[closest_path_index, :]; key_path_inflations = inflations_data[closest_path_index, :]
                 key_solved_w_total = closest_path_result['solved_w']
                 # Rerun path, get yearly_data dict keyed by DB COL NAMES
                 _, _, yearly_data_dict_for_path = run_single_path( key_solved_w_total, params_internal['initial_portfolio'], params_internal['horizon'], key_path_returns, key_path_inflations, income_streams, expense_streams)
                 # Store this dict, keyed by PERCENTILE now
                 key_paths_yearly_data_by_percentile[p] = yearly_data_dict_for_path
        # Prepare analysis results dictionary to return
        statuses = [r['status'] for r in results]
        analysis_results['summary_stats'] = { 'num_paths': params_internal['num_paths'], 'num_converged': statuses.count("Success") + statuses.count("ConvergedOutsideRange") + statuses.count("SuccessAtLowerBound") + statuses.count("SuccessAtUpperBound"), 'num_bound_hits': statuses.count("HitLowerBound") + statuses.count("HitUpperBound"), 'num_solver_errors': statuses.count("SolverError") } # Refined converged count
        analysis_results['solved_w_values'] = solved_w_total_values.tolist(); analysis_results['percentiles_w'] = percentile_values_w
        analysis_results['mean_w'] = np.mean(solved_w_total_values); analysis_results['median_w'] = np.median(solved_w_total_values)
        # Use standardized keys for first year flow calc
        analysis_results['first_year_income'] = sum(s.get('InitialAmount',0) for s in income_streams if s.get('StartYearOffset', 0) == 0)
        analysis_results['first_year_expense'] = sum(s.get('InitialAmount',0) for s in expense_streams if s.get('StartYearOffset', 0) == 0)
        analysis_results['key_paths_yearly_data'] = key_paths_yearly_data_by_percentile # Keyed by percentile, inner keys are DB Col names
    else: print("No valid paths converged.")
    print("--- Saving Run to Database ---")
    run_id = save_simulation_run(DB_FILE, params_internal, income_streams, expense_streams, results, key_paths_indices, key_paths_yearly_data_by_percentile, notes)
    analysis_results['run_id'] = run_id
    end_time = time.time(); print(f"Simulation execution finished in {end_time - start_time:.2f} seconds.")
    return analysis_results