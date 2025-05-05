# app_v4.py (Complete Corrected Version 18 - Fixed Load Error & Table Format)

import dash
from dash import dcc, html, Input, Output, State, callback, dash_table, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import json
import traceback
import pprint
import collections.abc

# --- Import Engine ---
# (Import logic unchanged)
simulator_engine_imported = False
critical_error_message = ""
def run_and_save_simulation(params, income_streams, expense_streams, notes=""): # Dummy
    print("WARN: Using dummy simulation function!")
    return {'summary_stats': {},'solved_w_values': [], 'percentiles_w': {}, 'mean_w': None, 'median_w': None, 'run_id': None,'key_paths_yearly_data': {}}
DEFAULT_INCOME_STREAMS = []
DEFAULT_EXPENSE_STREAMS = []
KEY_PERCENTILES = [10, 25, 50, 75, 90]; DB_FILE = 'retirement_sim.db'
try:
    from simulator_engine import run_and_save_simulation, DEFAULT_INCOME_STREAMS, DEFAULT_EXPENSE_STREAMS, KEY_PERCENTILES, DB_FILE
    simulator_engine_imported = True
except ImportError as e: critical_error_message = f"CRITICAL ERROR: Could not import from simulator_engine.py... {e}..."; print(critical_error_message)

# --- Initialize the Dash App ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Retirement Simulator V4"

# --- Helper Functions ---
# (create_input_row, format_run_for_dropdown unchanged)
def create_input_row(label, input_id, input_type='number', value=None, **kwargs):
    return dbc.Row([dbc.Label(label, html_for=input_id, width=6, style={'textAlign':'right', 'fontWeight':'bold'}), dbc.Col(dbc.Input(id=input_id, type=input_type, value=value, **kwargs), width=6)], className="mb-2")
def format_run_for_dropdown(run_data):
     try:
        ts = pd.to_datetime(run_data['Timestamp']).strftime('%Y-%m-%d %H:%M'); pfolio = f"${run_data['InitialPortfolioValue']:,.0f}" if run_data['InitialPortfolioValue'] is not None else "N/A"; paths_str = f"{run_data['NumberOfPaths']} paths" if run_data['NumberOfPaths'] is not None else ""; notes = run_data['Notes'] or ''; note_preview = notes[:30] + ('...' if len(notes) > 30 else '')
        return f"Run {run_data['RunID']} ({ts}) - {pfolio} ({paths_str}) - {note_preview}"
     except Exception: return f"Run {run_data.get('RunID', '?')} - Error formatting label"

def check_for_ellipsis(data_struct, path="root"): # Keep debug helper
    if data_struct is Ellipsis: print(f"!!! FOUND ELLIPSIS at path: {path}"); return True
    elif isinstance(data_struct, dict): found = False; [found := check_for_ellipsis(v, path=f"{path}[{repr(k)}]") or found for k, v in data_struct.items()]; return found
    elif isinstance(data_struct, collections.abc.Sequence) and not isinstance(data_struct, (str, bytes)): found = False; [found := check_for_ellipsis(v, path=f"{path}[{i}]") or found for i, v in enumerate(data_struct)]; return found
    return False

# --- Define App Layout ---
# (Layout unchanged)
empty_fig = go.Figure(layout={'title': {'text': 'Distribution - Run/Load Simulation'}, 'height': 400, 'xaxis':{'title':'Initial Total Withdrawal ($)'}, 'yaxis':{'title':'Number of Paths'}})
empty_timeseries_fig = go.Figure(layout={'title': {'text': 'Key Path Evolution - Run/Load Simulation'}, 'height': 450, 'xaxis':{'title':'Simulation Year'}, 'yaxis':{'title':'Amount ($)'}})

# --- Refined Table Column Definitions --- 
income_columns = [
    {'id': 'IncomeType', 'name': 'Type', 'type':'text', 'editable':True},
    {'id': 'StartYearOffset', 'name': 'Start Yr', 'type':'numeric', 'editable':True},
    {'id': 'EndYearOffset', 'name': 'End Yr', 'type':'numeric', 'editable':True},
    {'id': 'InitialAmount', 'name': 'Amount $', 'type':'text', 'editable':True},
    {'id': 'InflationAdjustmentRule', 'name': 'Inflation', 'type':'text', 'editable':True}
]
expense_columns = [
    {'id': 'ExpenseType', 'name': 'Type', 'type':'text', 'editable':True},
    {'id': 'StartYearOffset', 'name': 'Start Yr', 'type':'numeric', 'editable':True},
    {'id': 'EndYearOffset', 'name': 'End Yr', 'type':'numeric', 'editable':True},
    {'id': 'InitialAmount', 'name': 'Amount $', 'type':'text', 'editable':True},
    {'id': 'InflationAdjustmentRule', 'name': 'Inflation', 'type':'text', 'editable':True}
]

# --- Format Default Stream Amounts --- 
for stream in DEFAULT_INCOME_STREAMS:
    if isinstance(stream.get('InitialAmount'), (int, float)):
        stream['InitialAmount'] = f"${stream['InitialAmount']:,.0f}"
for stream in DEFAULT_EXPENSE_STREAMS:
    if isinstance(stream.get('InitialAmount'), (int, float)):
        stream['InitialAmount'] = f"${stream['InitialAmount']:,.0f}"
# ---------------------------------

app.layout = dbc.Container([ # Layout structure unchanged... content omitted for brevity
    dcc.Store(id='loaded-run-analysis-store'), dcc.Store(id='loaded-yearly-data-store'), dcc.Store(id='loaded-run-params-store'),
    dbc.Row(dbc.Col(html.H1("Retirement Scenario Simulator"), width=12), className="mb-4 mt-2 text-center"),
    dbc.Row([ dbc.Col([...], width=12, lg=4, className="bg-light p-4 border rounded shadow-sm"), # Input Column
              dbc.Col([...], width=12, lg=8, className="p-4") # Output Column
             ]) ], fluid=True)
# --- (Copy full layout from V16/V17 above if needed) ---
# Full Layout:
app.layout = dbc.Container([
    dcc.Store(id='loaded-run-analysis-store'), dcc.Store(id='loaded-yearly-data-store'), dcc.Store(id='loaded-run-params-store'),
    dbc.Row(dbc.Col(html.H1("Retirement Scenario Simulator"), width=12), className="mb-4 mt-2 text-center"),
    dbc.Row([
        dbc.Col([ html.H4("Load Past Run"), dbc.Row([ dbc.Col(dcc.Dropdown(id='run-selector-dropdown', placeholder='Select...'), width=9), dbc.Col(dbc.Button("Load", id='load-run-button', n_clicks=0, color="secondary"), width=3) ]), html.Hr(), html.H4("Simulation Setup"), create_input_row("Initial Portfolio ($):", 'input-initial-portfolio', value=2_500_000, step=10000, min=0), create_input_row("Planning Horizon Age:", 'input-horizon-age', value=92, step=1, min=1), create_input_row("Current Age (User):", 'input-current-age-user', value=63, step=1, min=0), create_input_row("Current Age (Spouse):", 'input-current-age-spouse', value=58, step=1, min=0), create_input_row("Target End Balance (Today $):", 'input-target-today', value=500_000, step=10000, min=0), create_input_row("Target Range (+/- $):", 'input-target-range', value=100_000, step=5000, min=0), html.Hr(), create_input_row("Return Mean (Nominal %):", 'input-return-mean', value=6.2, step=0.1), create_input_row("Return Std Dev (%):", 'input-return-sd', value=10.8, step=0.1, min=0), create_input_row("Inflation Mean (%):", 'input-inflation-mean', value=2.8, step=0.1), create_input_row("Inflation Std Dev (%):", 'input-inflation-sd', value=1.8, step=0.1, min=0), create_input_row("Longterm Inflation Mean (%):", 'input-longterm-inflation-mean', value=2.5, step=0.1), create_input_row("Longterm Inflation Std Dev (%):", 'input-longterm-inflation-sd', value=1.5, step=0.1, min=0), html.Hr(), create_input_row("Solver Min Total W (% Initial):", 'input-solver-min-pct', value=2.0, step=0.5, min=0), create_input_row("Solver Max Total W (% Initial):", 'input-solver-max-pct', value=20.0, step=0.5, min=0),
    dbc.Row([
        dbc.Label("Number of Paths:", html_for='input-num-paths', width=6, style={'textAlign':'right', 'fontWeight':'bold'}),
        dbc.Col(dbc.RadioItems(
            id='input-num-paths',
            options=[
                {'label': '1,000', 'value': 1000},
                {'label': '10,000', 'value': 10000},
                {'label': '25,000', 'value': 25000},
                {'label': '50,000', 'value': 50000},
            ],
            value=1000, # Default value
            inline=True,
            labelClassName="me-3", # Add spacing
            inputClassName="me-1"
        ), width=6)
    ], className="mb-2"),
    html.Hr(),
    html.H5("Income Streams"),
    dash_table.DataTable( id='income-streams-table', columns=income_columns, data=DEFAULT_INCOME_STREAMS, editable=True, row_deletable=True, style_table={'overflowX': 'auto'}, style_cell={'textAlign': 'left','padding':'3px'}, style_header={'fontWeight': 'bold'}),
    dbc.Button("Add Income Row", id='add-income-row-button', n_clicks=0, size="sm", className="mt-1 mb-3"),
    html.H5("Expense Streams"),
    dash_table.DataTable( id='expense-streams-table', columns=expense_columns, data=DEFAULT_EXPENSE_STREAMS, editable=True, row_deletable=True, style_table={'overflowX': 'auto'}, style_cell={'textAlign': 'left','padding':'3px'}, style_header={'fontWeight': 'bold'}),
    dbc.Button("Add Expense Row", id='add-expense-row-button', n_clicks=0, size="sm", className="mt-1 mb-3"),
    html.Hr(),
    dbc.Label("Run Notes:", html_for='input-run-notes', style={'fontWeight': 'bold'}),
    dbc.Textarea(id='input-run-notes', placeholder='Enter notes...', style={'height': '100px'}, className="mb-3"),
    dbc.Button("Run Simulation", id='run-button', n_clicks=0, color="primary", className="w-100") ], width=12, lg=4, className="bg-light p-4 border rounded shadow-sm"),
        dbc.Col([ html.H4("Results"), dbc.Spinner(id="loading-spinner", children=[ html.Div(id='loaded-run-details'), html.Hr(id='results-separator', style={'display':'none'}), html.Div(id='summary-stats-output', children="Load a past run or click 'Run Simulation'."), dcc.Graph(id='histogram-output', figure=empty_fig), html.Hr(id='plot-separator', style={'display':'none'}), html.Div(id='path-plot-section', children=[ html.H5("Key Path Evolution"), dbc.Row([ dbc.Col(dbc.Label("Show Paths:", width="auto"), className="col-auto"), dbc.Col(dbc.Checklist(id='timeseries-path-selector', options=[], value=[], inline=True)), dbc.Col(dbc.Label("Data Series:", width="auto"), className="col-auto"), dbc.Col(dbc.Checklist(
                            id='timeseries-data-selector', 
                            options=[
                                {'label':'Balance ($)', 'value':'StartOfYearBalance'}, 
                                {'label':'Total Withdrawal ($)', 'value':'TotalWithdrawal'}, 
                                {'label':'Total Income ($)', 'value':'TotalIncome'}, 
                                {'label':'Total Expense ($)', 'value':'TotalExpense'},
                                {'label':'Market Return (%)', 'value':'MarketReturn'}, 
                                {'label':'Inflation (%)', 'value':'Inflation'}
                            ], 
                            value=['StartOfYearBalance'], # Default selection
                            inline=True,
                            labelClassName="me-3", # Add some spacing between items
                            inputClassName="me-1"
                        ))
                    ], className="mb-2 align-items-center"), dcc.Graph(id='key-paths-timeseries-graph', figure=empty_timeseries_fig), html.Hr(), html.H5("Yearly Path Detail"), dbc.Row([ dbc.Col(dbc.Label("Select Percentile Path:", width="auto"), className="col-auto"), dbc.Col(dbc.Select(id='detail-table-percentile-selector', options=[], placeholder="Select..."), width=3) ], className="mb-2 align-items-center"), html.Div(dash_table.DataTable(id='key-path-detail-table', style_table={'overflowX': 'auto', 'height': '350px', 'overflowY': 'auto'}, style_cell={'textAlign':'right','padding':'5px'}, style_header={'fontWeight': 'bold', 'position': 'sticky', 'top': 0, 'backgroundColor':'white', 'zIndex':1}, fixed_rows={'headers': True}), id='detail-table-div')], style={'display':'none'})], color="primary", spinner_style={"width": "3rem", "height": "3rem"}) ], width=12, lg=8, className="p-4")
        ])
], fluid=True)


# --- Helper Function to Load Data ---
def load_data_for_run(run_id_to_load):
    """Loads all data associated with a specific RunID - V18 FIX for UnboundLocalError"""
    if run_id_to_load is None: return None
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # 1. Load Params - Ensure loaded_data is initialized *after* success
        cursor.execute("SELECT * FROM SimulationRuns WHERE RunID = ?", (run_id_to_load,))
        params_row = cursor.fetchone()
        if not params_row:
            print(f"RunID {run_id_to_load} not found in SimulationRuns.")
            conn.close()
            return None

        # Initialize dict *after* confirming params_row exists
        loaded_data = {'params': dict(params_row)}
        print(" Parameters loaded.")

        # 2. Load Streams, Results, Yearly Data (wrap each in try/except?)
        # For simplicity, keep in main try block for now. If specific loads fail, df will be empty.
        q_income = "SELECT IncomeType, StartYearOffset, EndYearOffset, InitialAmount, InflationAdjustmentRule FROM IncomeStreams WHERE RunID = ? ORDER BY StartYearOffset, IncomeType"
        loaded_data['income_streams'] = pd.read_sql_query(q_income, conn, params=(run_id_to_load,))
        q_expense = "SELECT ExpenseType, StartYearOffset, EndYearOffset, InitialAmount, InflationAdjustmentRule FROM Expenses WHERE RunID = ? ORDER BY StartYearOffset, ExpenseType"
        loaded_data['expense_streams'] = pd.read_sql_query(q_expense, conn, params=(run_id_to_load,))
        q_results = "SELECT * FROM PathResults WHERE RunID = ? ORDER BY PathNumber"
        loaded_data['path_summaries'] = pd.read_sql_query(q_results, conn, params=(run_id_to_load,))
        q_yearly="SELECT pr.PercentileRank, pyd.* FROM PathYearlyData pyd JOIN PathResults pr ON pyd.ResultID=pr.ResultID WHERE pr.RunID = ? AND pr.IsKeyPercentilePath = 1 ORDER BY pr.PercentileRank, pyd.YearNum"
        key_paths_yearly_df = pd.read_sql_query(q_yearly, conn, params=(run_id_to_load,))
        yearly_data_dict = {};
        if not key_paths_yearly_df.empty:
             for p_rank, group_df in key_paths_yearly_df.groupby('PercentileRank'): yearly_data_dict[int(p_rank)] = group_df.to_dict(orient='list'); del yearly_data_dict[int(p_rank)]['PercentileRank']
        loaded_data['key_paths_yearly_data'] = yearly_data_dict

        print(f"Data loaded successfully for RunID {run_id_to_load}.")
        return loaded_data

    except Exception as e:
        print(f"Error loading data for RunID {run_id_to_load}: {e}")
        traceback.print_exc() # Print full traceback for debugging
        # loaded_data might be partially populated or unassigned if error was early
        # Return None to indicate failure
        return None
    finally:
        if conn: conn.close()


# --- Define Callbacks ---

# Populate run selector dropdown
@callback( Output('run-selector-dropdown', 'options'), Input('run-selector-dropdown', 'id') )
def populate_run_selector(_): # (Unchanged)
    options=[{'label':'Select...','value':'', 'disabled':True}]; conn=None
    try: conn=sqlite3.connect(DB_FILE); conn.row_factory=sqlite3.Row; runs=conn.execute("SELECT RunID, Timestamp, InitialPortfolioValue, NumberOfPaths, Notes FROM SimulationRuns ORDER BY Timestamp DESC LIMIT 100").fetchall(); conn.close();
    except Exception as e: print(f"Error populating dropdown: {e}"); runs = []
    finally:
         if conn: conn.close()
    for run_row in runs: options.append({'label': format_run_for_dropdown(run_row), 'value': run_row['RunID']})
    return options

# Add Income Row
@callback( Output('income-streams-table', 'data', allow_duplicate=True), Input('add-income-row-button', 'n_clicks'), State('income-streams-table', 'data'), prevent_initial_call=True)
def add_income_row(n_clicks, rows): # (Unchanged)
    print("DEBUG: Add Income Row"); current_rows = rows if rows is not None else []
    # --- Format amount for new row ---
    current_rows.append({'IncomeType': 'New', 'StartYearOffset': 0, 'EndYearOffset': 34, 'InitialAmount': '$0', 'InflationAdjustmentRule': 'General'})
    # -------------------------------
    return current_rows

# Add Expense Row
@callback( Output('expense-streams-table', 'data', allow_duplicate=True), Input('add-expense-row-button', 'n_clicks'), State('expense-streams-table', 'data'), prevent_initial_call=True)
def add_expense_row(n_clicks, rows): # (Unchanged)
    print("DEBUG: Add Expense Row"); current_rows = rows if rows is not None else []
    # --- Format amount for new row ---
    current_rows.append({'ExpenseType': 'New', 'StartYearOffset': 0, 'EndYearOffset': 34, 'InitialAmount': '$0', 'InflationAdjustmentRule': 'General'})
    # -------------------------------
    return current_rows

# Load selected run data into stores and display areas
@callback( # Outputs unchanged
    Output('loaded-run-analysis-store', 'data'), Output('loaded-yearly-data-store', 'data'), Output('loaded-run-params-store', 'data'),
    Output('input-initial-portfolio', 'value'), Output('input-horizon-age', 'value'), Output('input-current-age-user', 'value'), Output('input-current-age-spouse', 'value'),
    Output('input-target-today', 'value'), Output('input-target-range', 'value'), Output('input-return-mean', 'value'), Output('input-return-sd', 'value'),
    Output('input-inflation-mean', 'value'), Output('input-inflation-sd', 'value'),
    Output('input-longterm-inflation-mean', 'value'),
    Output('input-longterm-inflation-sd', 'value'),
    Output('input-solver-min-pct', 'value'), Output('input-solver-max-pct', 'value'),
    Output('input-num-paths', 'value'), Output('input-run-notes', 'value'),
    Output('income-streams-table', 'data', allow_duplicate=True), Output('expense-streams-table', 'data', allow_duplicate=True),
    Output('loaded-run-details', 'children'), Output('summary-stats-output', 'children'), Output('histogram-output', 'figure'), Output('results-separator', 'style'), Output('path-plot-section', 'style'),
    Input('load-run-button', 'n_clicks'), State('run-selector-dropdown', 'value'), prevent_initial_call=True )
def load_selected_run_callback(n_clicks, selected_run_id): # (Logic mostly unchanged, relies on corrected load_data_for_run)
    ctx = dash.callback_context
    if not ctx.triggered or selected_run_id is None: return dash.no_update
    print(f"\nLoad button clicked for RunID: {selected_run_id}"); loaded_data = load_data_for_run(selected_run_id) # Call corrected helper
    no_data_msg=html.Div(f"Could not load data for RunID {selected_run_id}.", className="alert alert-danger"); no_update_inputs=[dash.no_update]*16; outputs=([None,None,None]+no_update_inputs+[dash.no_update]*2+[no_data_msg,no_data_msg,empty_fig,{'display':'none'},{'display':'none'}])
    if not loaded_data: return outputs # Return defaults if loading failed
    params=loaded_data['params']; income_df=loaded_data['income_streams']; expense_df=loaded_data['expense_streams']; paths=loaded_data['path_summaries']; yearly_dict=loaded_data['key_paths_yearly_data']; analysis_store={}
    valid_s=paths.dropna(subset=['SolvedInitialWithdrawal']); w_vals=valid_s['SolvedInitialWithdrawal'].values
    if len(w_vals)>0: # Build analysis store (logic unchanged)
        key_p_stored=params.get('KeyPercentilesStored',''); key_p=[int(p) for p in key_p_stored.split(',') if p] if key_p_stored else KEY_PERCENTILES; calc_p=np.percentile(w_vals,key_p) if key_p else []; p_w=dict(zip(key_p,calc_p)); mean_w=np.mean(w_vals); med_w=np.median(w_vals); statuses=paths['Status'].tolist()
        analysis_store['summary_stats'] = {'num_paths':len(paths),'num_converged':statuses.count("Success")+statuses.count("ConvergedOutsideRange")+statuses.count("SuccessAtLowerBound")+statuses.count("SuccessAtUpperBound"), 'num_bound_hits':statuses.count("HitLowerBound")+statuses.count("HitUpperBound"), 'num_solver_errors':statuses.count("SolverError")}
        analysis_store['solved_w_values'] = w_vals.tolist(); analysis_store['percentiles_w'] = p_w; analysis_store['mean_w'] = mean_w; analysis_store['median_w'] = med_w
        inc_df0=income_df[income_df['StartYearOffset']==0]['InitialAmount']; analysis_store['first_year_income']=pd.to_numeric(inc_df0,errors='coerce').sum()
        exp_df0=expense_df[expense_df['StartYearOffset']==0]['InitialAmount']; analysis_store['first_year_expense']=pd.to_numeric(exp_df0,errors='coerce').sum()
    else: analysis_store={'summary_stats':{}, 'solved_w_values':[], 'percentiles_w':{}}
    # *** DEBUG CHECK before JSON DUMP ***
    print("\nDEBUG LOAD: Checking loaded yearly_dict for ellipsis before json.dumps..."); yearly_store_data = None
    if yearly_dict: check_for_ellipsis(yearly_dict, path="yearly_dict_loaded");
    else: print("DEBUG LOAD: yearly_dict is empty or None.")
    try: yearly_store_data = json.dumps(yearly_dict) if yearly_dict else None # This might still fail if ellipsis is present
    except Exception as json_err: print(f"!!! JSON DUMP ERROR on LOAD: {json_err}"); traceback.print_exc()
    p = params; # Prepare input field values (unchanged)
    input_vals = [ p.get('InitialPortfolioValue'), dash.no_update, dash.no_update, dash.no_update, p.get('TargetEndBalance_Base_TodayDollars'), p.get('TargetEndBalance_Range_TodayDollars'),
                   p.get('ReturnMean')*100 if p.get('ReturnMean') is not None else None, p.get('ReturnStdDev')*100 if p.get('ReturnStdDev') is not None else None,
                   p.get('InflationMean')*100 if p.get('InflationMean') is not None else None, p.get('InflationStdDev')*100 if p.get('InflationStdDev') is not None else None,
                   p.get('longterm_inf_mean')*100 if p.get('longterm_inf_mean') is not None else None,
                   p.get('longterm_inf_sd')*100 if p.get('longterm_inf_sd') is not None else None,
                   p.get('WithdrawalSolverMinPct')*100 if p.get('WithdrawalSolverMinPct') is not None else None, p.get('WithdrawalSolverMaxPct')*100 if p.get('WithdrawalSolverMaxPct') is not None else None,
                   p.get('NumberOfPaths'), p.get('Notes', '') ]
    # --- Format Amount columns before sending to table ---
    income_table_data = income_df.to_dict('records')
    for row in income_table_data:
        if pd.notna(row.get('InitialAmount')):
            try: row['InitialAmount'] = f"${float(row['InitialAmount']):,.0f}" 
            except (ValueError, TypeError): row['InitialAmount'] = str(row['InitialAmount']) # Fallback
        else: row['InitialAmount'] = ""
    expense_table_data = expense_df.to_dict('records')
    for row in expense_table_data:
        if pd.notna(row.get('InitialAmount')):
            try: row['InitialAmount'] = f"${float(row['InitialAmount']):,.0f}" 
            except (ValueError, TypeError): row['InitialAmount'] = str(row['InitialAmount'])
        else: row['InitialAmount'] = ""
    # ----------------------------------------------------
    param_items = [dbc.Row([dbc.Col(f"{k.replace('_',' ').title()}:",width=4,className="text-end fw-bold"), dbc.Col(p.get(k),width=8)]) for k in ['RunID','Timestamp','SimulationHorizonYears','KeyPercentilesStored'] if p.get(k)]
    loaded_details = html.Div([ html.H5(f"Displaying Loaded Run ID: {selected_run_id}"), dbc.Row([dbc.Col(param_items, width=12, lg=6), dbc.Col([html.B("Notes:"), html.Blockquote(p.get('Notes') or "(No notes)", className="blockquote-footer mt-1" if p.get('Notes') else "text-muted mt-1")], width=12, lg=6)]), ], className="mt-3 mb-3 border rounded p-3 bg-light")
    summary_stats=analysis_store.get('summary_stats',{}); percentiles_w=analysis_store.get('percentiles_w',{}); mean_w=analysis_store.get('mean_w'); median_w=analysis_store.get('median_w'); summary_div=html.Div("No results.")
    if analysis_store.get('solved_w_values'): # Build summary display... (logic unchanged)
        summary_children = [ html.H5(f"Results for Loaded Run ID: {selected_run_id}"), dbc.Row([...]), dbc.Row([...]), html.Hr(), html.H6("Solved Initial Total Withdrawal (W):") ]
        summary_children = [ html.H5(f"Results for Loaded Run ID: {selected_run_id}"), dbc.Row([dbc.Col("Paths:",width="auto"),dbc.Col(f"{summary_stats.get('num_paths','N/A')}",width=True), dbc.Col("Converged:",width="auto"),dbc.Col(f"{summary_stats.get('num_converged','N/A')}",width=True)]), dbc.Row([dbc.Col("Hit Bounds:",width="auto"),dbc.Col(f"{summary_stats.get('num_bound_hits','N/A')}",width=True), dbc.Col("Errors:",width="auto"),dbc.Col(f"{summary_stats.get('num_solver_errors','N/A')}",width=True)]), html.Hr(), html.H6("Solved Initial Total Withdrawal (W):") ]
        p_list_items = []
        for perc, val in sorted(percentiles_w.items()):
              tot_spend_pct = (val / p['InitialPortfolioValue'])*100 if p.get('InitialPortfolioValue') else 0; discr_pct = ((val - analysis_store['first_year_expense'] + analysis_store['first_year_income']) / p['InitialPortfolioValue'])*100 if p.get('InitialPortfolioValue') else 0
              p_list_items.append( html.Li(f"{perc}th Perc: ${val:,.0f} ({tot_spend_pct:.2f}%) | Approx Initial Discr: {discr_pct:.2f}%") )
        summary_children.extend([html.Ul(p_list_items, style={'fontSize':'0.9em'}), html.P(f"Mean W: ${mean_w:,.0f}", className="mb-0"), html.P(f"Median W: ${median_w:,.0f}", className="mt-0")])
        summary_div = html.Div(summary_children, className="mt-4")
    fig = empty_fig
    if analysis_store.get('solved_w_values'): hist_df = pd.DataFrame(analysis_store['solved_w_values'], columns=['Solved Total W']); fig = px.histogram(hist_df, x='Solved Total W', nbins=max(10,min(50,len(analysis_store['solved_w_values'])//10)), title=f"Distribution (Loaded Run {selected_run_id})"); fig.update_layout(bargap=0.1, height=400, xaxis_title="Initial Total Withdrawal ($)", yaxis_title="Number of Paths")
    plot_section_style = {'display':'block'} if yearly_store_data else {'display':'none'}
    # Order: analysis_store, yearly_store, params_store, input_vals(16), income_table, expense_table, loaded_details, summary, histo, separator, plot_section
    return ( analysis_store, yearly_store_data, params, *input_vals, income_table_data, expense_table_data, loaded_details, summary_div, fig, {'display':'block'}, plot_section_style )

# Callback to Run Simulation
@callback( # Outputs unchanged
    Output('loaded-run-analysis-store', 'data', allow_duplicate=True), Output('loaded-yearly-data-store', 'data', allow_duplicate=True), Output('loaded-run-params-store', 'data', allow_duplicate=True),
    Output('summary-stats-output', 'children', allow_duplicate=True), Output('histogram-output', 'figure', allow_duplicate=True), Output('loaded-run-details', 'children', allow_duplicate=True),
    Output('results-separator', 'style', allow_duplicate=True), Output('path-plot-section', 'style', allow_duplicate=True),
    Input('run-button', 'n_clicks'), # States unchanged...
    # --- Added States for new longterm inflation inputs ---
    State('input-initial-portfolio', 'value'), State('input-horizon-age', 'value'), State('input-current-age-user', 'value'), State('input-current-age-spouse', 'value'), State('input-target-today', 'value'), State('input-target-range', 'value'), State('input-return-mean', 'value'), State('input-return-sd', 'value'), State('input-inflation-mean', 'value'), State('input-inflation-sd', 'value'),
    State('input-longterm-inflation-mean', 'value'), # Added
    State('input-longterm-inflation-sd', 'value'),   # Added
    State('input-solver-min-pct', 'value'), State('input-solver-max-pct', 'value'), State('input-num-paths', 'value'), State('input-run-notes', 'value'), State('income-streams-table', 'data'), State('expense-streams-table', 'data'), prevent_initial_call=True )
def run_new_simulation_callback(n_clicks, initial_portfolio, horizon_age, current_age_user, current_age_spouse, target_today, target_range, return_mean_pct, return_sd_pct, inflation_mean_pct, inflation_sd_pct,
                                longterm_inflation_mean_pct, longterm_inflation_sd_pct, # Added args
                                solver_min_pct, solver_max_pct, num_paths, run_notes, income_table_data, expense_table_data):
    # ---------------------------------------------------
    # (Added Debug Print before json.dumps, uses standardized stream keys)
    if n_clicks is None or n_clicks < 1: return dash.no_update
    if not simulator_engine_imported: return None, None, None, html.Div(html.Pre(critical_error_message),className="alert alert-danger"), empty_fig, None, {'display':'none'}, {'display':'none'}
    print("\nRun button clicked! Starting simulation..."); params = {}; error_message_div = None; analysis_results = None; income_streams = []; expense_streams = []
    try: # Input Validation (unchanged)
        # --- Updated inputs dict for validation ---
        inputs={"ip":initial_portfolio,"ha":horizon_age,"u":current_age_user,"s":current_age_spouse,"tt":target_today,"tr":target_range,"rm":return_mean_pct,"rsd":return_sd_pct,"im":inflation_mean_pct,"isd":inflation_sd_pct,
                "ltim":longterm_inflation_mean_pct, "ltisd":longterm_inflation_sd_pct, # Added
                "minp":solver_min_pct,"maxp":solver_max_pct,"np":num_paths}
        # -----------------------------------------
        none_inputs=[k for k,v in inputs.items() if v is None];
        if none_inputs: raise ValueError(f"Missing: {', '.join(none_inputs)}")
        youngest=min(int(current_age_user), int(current_age_spouse)); horizon=max(1, int(horizon_age)-youngest)
        # --- Updated params dict ---
        params = {'initial_portfolio':float(initial_portfolio), 'horizon':horizon, 'target_today':float(target_today), 'target_range_today':float(target_range), 'ret_mean':float(return_mean_pct)/100.0, 'ret_sd':float(return_sd_pct)/100.0,
                  'inf_mean':float(inflation_mean_pct)/100.0, 'inf_sd':float(inflation_sd_pct)/100.0,
                  'longterm_inf_mean': float(longterm_inflation_mean_pct)/100.0, # Added
                  'longterm_inf_sd': float(longterm_inflation_sd_pct)/100.0,     # Added
                  'total_min_pct':float(solver_min_pct)/100.0, 'total_max_pct':float(solver_max_pct)/100.0, 'num_paths':int(num_paths)}
        # ------------------------
        if params['horizon']<=0: raise ValueError("Horizon Age > youngest age.")
        
        # --- Parse Amount strings back to float when creating streams for simulation --- 
        def parse_amount(amount_str):
           if isinstance(amount_str, (int, float)): return float(amount_str) # Already numeric
           if isinstance(amount_str, str):
               try:
                   # Remove $ and , then convert to float
                   cleaned_str = amount_str.replace('$', '').replace(',', '')
                   return float(cleaned_str)
               except ValueError:
                   return 0.0 # Or raise error, or handle differently?
           return 0.0 # Default if None or unexpected type

        income_streams = [{'IncomeType': r.get('IncomeType'), 'StartYearOffset': int(r.get('StartYearOffset',0)), 'EndYearOffset': int(r.get('EndYearOffset', params['horizon'])), 'InitialAmount': parse_amount(r.get('InitialAmount')), 'InflationAdjustmentRule': r.get('InflationAdjustmentRule', 'General')} for i, r in enumerate(income_table_data or []) if r and r.get('IncomeType')]
        expense_streams = [{'ExpenseType': r.get('ExpenseType'), 'StartYearOffset': int(r.get('StartYearOffset',0)), 'EndYearOffset': int(r.get('EndYearOffset', params['horizon'])), 'InitialAmount': parse_amount(r.get('InitialAmount')), 'InflationAdjustmentRule': r.get('InflationAdjustmentRule', 'General')} for i, r in enumerate(expense_table_data or []) if r and r.get('ExpenseType')]
        # ----------------------------------------------------------------------------

    except (TypeError, ValueError, KeyError) as e: error_message_div = html.Div([html.H5("Input Error",className="text-danger"), html.P(f"{e}")], className="mt-4"); return None, None, None, error_message_div, empty_fig, None, {'display':'none'}, {'display':'none'}
    # Run Simulation
    try: analysis_results = run_and_save_simulation(params, income_streams, expense_streams, run_notes); print("Sim complete.")
    except Exception as e: print(f"ERROR: {e}"); traceback.print_exc(); error_message_div = html.Div([html.H5("Runtime Error",className="text-danger"), html.P(f"{e}")], className="mt-4"); return None, None, None, error_message_div, empty_fig, None, {'display':'none'}, {'display':'none'}
    # Prepare Outputs
    run_id=analysis_results.get('run_id'); summary_stats=analysis_results.get('summary_stats',{}); solved_w_values=analysis_results.get('solved_w_values',[])
    percentiles_w=analysis_results.get('percentiles_w',{}); mean_w=analysis_results.get('mean_w'); median_w=analysis_results.get('median_w')
    first_year_income=analysis_results.get('first_year_income',0); first_year_expense=analysis_results.get('first_year_expense',0)
    key_paths_yearly_data = analysis_results.get('key_paths_yearly_data', {})
    summary_div = html.Div("No valid results."); fig = empty_fig; analysis_store_data = analysis_results
    # *** ADD DEBUG CHECK BEFORE JSON DUMP ***
    print("\nDEBUG RUN: Checking generated yearly_dict for ellipsis before json.dumps..."); yearly_store_data = None
    if key_paths_yearly_data: check_for_ellipsis(key_paths_yearly_data, path="key_paths_yearly_data_run");
    else: print("DEBUG RUN: key_paths_yearly_data is empty or None.")
    try: yearly_store_data = json.dumps(key_paths_yearly_data) if key_paths_yearly_data else None
    except Exception as json_err: print(f"!!! JSON DUMP ERROR on RUN: {json_err}"); traceback.print_exc()

    if solved_w_values: # Build summary div and histogram fig... (logic unchanged)
        summary_children = [ html.H5(f"Run Summary (New Run ID: {run_id if run_id else 'N/A'})"), dbc.Row([...]), dbc.Row([...]), html.Hr(), html.H6("Solved Initial Total Withdrawal (W):") ]
        summary_children = [ html.H5(f"Run Summary (New Run ID: {run_id if run_id else 'N/A'})"), dbc.Row([dbc.Col("Paths:",width="auto"),dbc.Col(f"{summary_stats.get('num_paths', 'N/A')}",width=True), dbc.Col("Converged:",width="auto"),dbc.Col(f"{summary_stats.get('num_converged', 'N/A')}",width=True)]), dbc.Row([dbc.Col("Hit Bounds:",width="auto"),dbc.Col(f"{summary_stats.get('num_bound_hits', 'N/A')}",width=True), dbc.Col("Errors:",width="auto"),dbc.Col(f"{summary_stats.get('num_solver_errors', 'N/A')}",width=True)]), html.Hr(), html.H6("Solved Initial Total Withdrawal (W):") ]
        p_list_items = []
        for p, val in sorted(percentiles_w.items()):
              tot_spend_pct = (val / params['initial_portfolio']) * 100 if params['initial_portfolio'] else 0; discr_pct = ((val - first_year_expense + first_year_income) / params['initial_portfolio']) * 100 if params['initial_portfolio'] else 0
              p_list_items.append( html.Li(f"{p}th Perc: ${val:,.0f} ({tot_spend_pct:.2f}%) | Approx Initial Discr: {discr_pct:.2f}%") )
        summary_children.extend([ html.Ul(p_list_items, style={'fontSize': '0.9em'}), html.P(f"Mean W: ${mean_w:,.0f}", className="mb-0"), html.P(f"Median W: ${median_w:,.0f}", className="mt-0") ])
        summary_div = html.Div(summary_children, className="mt-4")
        hist_df = pd.DataFrame(solved_w_values, columns=['Solved Total W']); fig = px.histogram(hist_df, x='Solved Total W', nbins=max(10, min(50, len(solved_w_values)//10)), title="Distribution (New Run)"); fig.update_layout(bargap=0.1, height=400, xaxis_title="Initial Total Withdrawal ($)", yaxis_title="Number of Paths")
    plot_section_style = {'display':'block'} if key_paths_yearly_data else {'display':'none'}
    print("Outputs prepared for UI update.")
    # Order: analysis_store, yearly_store, params_store, summary, histo, loaded_details, separator, plot_section
    return ( analysis_store_data, yearly_store_data, params, summary_div, fig, None, {'display':'none'}, plot_section_style )

# --- CALLBACKS FOR PLOTS/TABLES ---
# (Callbacks update_path_selectors, update_timeseries_plot, update_detail_table remain unchanged from V16)
@callback( Output('timeseries-path-selector', 'options'), Output('timeseries-path-selector', 'value'), Output('detail-table-percentile-selector', 'options'), Input('loaded-run-analysis-store', 'data') )
def update_path_selectors(analysis_data): # (Unchanged)
    if not analysis_data or not analysis_data.get('percentiles_w'): return [], [], []
    available_percentiles = sorted(analysis_data['percentiles_w'].keys()); options = [{'label': f"{p}th", 'value': p} for p in available_percentiles]; default_checklist = [p for p in [10, 50, 90] if p in available_percentiles]
    if not default_checklist and 50 in available_percentiles: default_checklist = [50]
    elif not default_checklist and available_percentiles: default_checklist = [available_percentiles[len(available_percentiles)//2]]
    return options, default_checklist, options

# --- Rebuilt callback for multi-select checklist and dual y-axes --- 
@callback(
    Output('key-paths-timeseries-graph', 'figure'), 
    Input('loaded-yearly-data-store', 'data'), 
    Input('timeseries-path-selector', 'value'), # Checklist of percentiles
    Input('timeseries-data-selector', 'value')  # Checklist of data columns
)
def update_timeseries_plot(yearly_data_json, selected_percentiles, selected_data_cols):
    # Define which columns represent percentages
    percent_cols = ['MarketReturn', 'Inflation']
    # Define display names (could be more dynamic later)
    col_display_names = {
        'StartOfYearBalance': 'Balance',
        'TotalWithdrawal': 'Total Withdrawal',
        'TotalIncome': 'Total Income',
        'TotalExpense': 'Total Expense',
        'MarketReturn': 'Market Return',
        'Inflation': 'Inflation'
    }

    # Initialize figure with secondary y-axis
    fig = go.Figure()
    fig.update_layout(
        title_text='Key Path Evolution',
        xaxis_title='Simulation Year',
        yaxis=dict(title='Amount ($)', side='left'),
        yaxis2=dict(title='Percent (%)', side='right', overlaying='y', showgrid=False, zeroline=False),
        height=500,
        legend_title_text='Path & Series'
    )

    if not yearly_data_json or not selected_percentiles or not selected_data_cols: 
        return fig # Return empty fig with layout if no data/selection

    try:
        yearly_data_dict = json.loads(yearly_data_json)
        # Color mapping could be added here for consistency
        # E.g., colors = px.colors.qualitative.Plotly

        for p in selected_percentiles:
            p_str = str(p)
            if p_str in yearly_data_dict:
                path_data = yearly_data_dict[p_str]
                years = path_data.get('YearNum', [])
                num_years = len(years)
                if num_years == 0: continue

                for data_col in selected_data_cols:
                    if data_col in path_data:
                        data_series = path_data[data_col]
                        if len(data_series) == num_years:
                            
                            is_percent = data_col in percent_cols
                            axis = 'y2' if is_percent else 'y1' # y1 is default primary
                            display_name = col_display_names.get(data_col, data_col)
                            trace_name = f"P{p} {display_name}"

                            # Format hovertemplate based on type
                            if is_percent:
                                hovertemplate = f'<b>{trace_name}</b><br>Year: %{{x}}<br>Value: %{{y:.2%}}<extra></extra>'
                                # Convert decimal to percentage for plotting on % axis
                                data_series_plot = [d * 100 if d is not None else None for d in data_series]
                            else:
                                hovertemplate = f'<b>{trace_name}</b><br>Year: %{{x}}<br>Value: $%{{y:,.0f}}<extra></extra>'
                                data_series_plot = data_series # Plot as is

                            fig.add_trace(
                                go.Scatter(
                                    x=years,
                                    y=data_series_plot,
                                    mode='lines+markers',
                                    name=trace_name,
                                    yaxis=axis, # Explicitly set axis ('y1' or 'y2')
                                    hovertemplate=hovertemplate
                                ),
                                # No need for secondary_y=True, yaxis='y2' handles it
                            )
    except Exception as e:
        print(f"Error plotting timeseries: {e}"); traceback.print_exc()

    # Always set titles, let Plotly hide axes if unused
    fig.update_layout(
        yaxis=dict(title='Amount ($)'), 
        yaxis2=dict(title='Percent (%)')
    )

    return fig

@callback( Output('key-path-detail-table', 'data'), Output('key-path-detail-table', 'columns'), Input('loaded-yearly-data-store', 'data'), Input('detail-table-percentile-selector', 'value') )
def update_detail_table(yearly_data_json, selected_percentile): # (Expects DB Col names, Pre-formats currency)
    columns_def = [ {"name":"Year","id":"YearNum","type":"numeric"}, {"name":"Start Balance","id":"StartOfYearBalance","type":"text"}, {"name":"Total Income","id":"TotalIncome","type":"text"}, {"name":"Total Expense","id":"TotalExpense","type":"text"}, {"name":"Total Withdrawal","id":"TotalWithdrawal","type":"text"}, {"name":"Market Return","id":"MarketReturn","type":"numeric","format":dash_table.Format.Format(precision=2,scheme=dash_table.Format.Scheme.percentage_rounded)}, {"name":"Inflation","id":"Inflation","type":"numeric","format":dash_table.Format.Format(precision=2,scheme=dash_table.Format.Scheme.percentage_rounded)}, {"name":"End Balance","id":"EndOfYearBalance","type":"text"},]
    default_data = [];
    if not yearly_data_json or selected_percentile is None: return default_data, columns_def
    # print(f"\n--- update_detail_table ---"); print(f"Selected Percentile: {selected_percentile}")
    try:
        yearly_data_dict = json.loads(yearly_data_json); # print(f"DEBUG Table: Loaded keys: {list(yearly_data_dict.keys())}")
        p_str = str(selected_percentile)
        if p_str in yearly_data_dict:
             # print(f"DEBUG Table: Found data for key '{p_str}'");
             path_data = yearly_data_dict[p_str]; num_years = len(path_data.get('YearNum', []))
             if num_years == 0: return default_data, columns_def
             table_data = []
             for year_idx in range(num_years):
                 row = {'YearNum': path_data.get('YearNum',[])[year_idx] if year_idx < len(path_data.get('YearNum',[])) else year_idx+1 }
                 def get_val(key, idx, default=np.nan): return path_data.get(key,[])[idx] if idx < len(path_data.get(key,[])) else default
                 start_bal=get_val('StartOfYearBalance',year_idx); end_bal=get_val('EndOfYearBalance',year_idx); tot_inc=get_val('TotalIncome',year_idx); tot_exp=get_val('TotalExpense',year_idx); tot_wd=get_val('TotalWithdrawal',year_idx); mkt_ret=get_val('MarketReturn',year_idx); infl=get_val('Inflation',year_idx)
                 row['MarketReturn'] = mkt_ret; row['Inflation'] = infl
                 row['StartOfYearBalance'] = f"${start_bal:,.0f}" if not pd.isna(start_bal) else "N/A"; row['TotalIncome'] = f"${tot_inc:,.0f}" if not pd.isna(tot_inc) else "N/A"; row['TotalExpense'] = f"${tot_exp:,.0f}" if not pd.isna(tot_exp) else "N/A"; row['TotalWithdrawal'] = f"${tot_wd:,.0f}" if not pd.isna(tot_wd) else "N/A"; row['EndOfYearBalance'] = f"${end_bal:,.0f}" if not pd.isna(end_bal) else "N/A"
                 table_data.append(row)
             # print("DEBUG Table: Table data generated.")
             return table_data, columns_def
        else: print(f"DEBUG Table: Key '{p_str}' not found."); return default_data, columns_def
    except Exception as e: print(f"Error table: {e}"); traceback.print_exc(); return default_data, columns_def

# --- Run the App ---
if __name__ == '__main__':
    if not simulator_engine_imported: print("\nERROR: Cannot start server..."); print(critical_error_message)
    else: app.run(debug=True)