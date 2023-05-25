import csv
import os

import numpy as np
import pandas as pd
import sqlite3
import datetime
import json
import zipfile
import pytz
import glob

from collections import defaultdict
from itertools import combinations

import networkx as nx

from cryptography.fernet import Fernet
import matplotlib.pyplot as plt
import matplotlib.colors

from model import MILP
from gurobipy import *
from flask import Flask, render_template, request, url_for, redirect, session, flash

import config_sim_builder as csb

import utils as u

# Get main conf
with open('static/sims-basic-config/cfg.json', 'r') as f:
    main_cfg = json.load(f)

app = Flask(__name__)

app.config.update(
    SECRET_KEY=main_cfg['appSecretKey']
)


def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn


def get_key():
    key_file = open('key.key', 'rb')
    key = key_file.read()
    key_file.close()
    return key


def decrypt(str_data):
    key = get_key()
    f = Fernet(key)
    return f.decrypt(str_data).decode()


def encrypt(str_data):
    key = get_key()
    f = Fernet(key)
    str_data = str_data.encode()
    return f.encrypt(str_data).decode()


def check_login_data(conn, u, p):
    login_data = conn.execute("SELECT * FROM user WHERE username='%s'" % u).fetchall()
    if len(login_data) > 0 and 'username' in login_data[0].keys():
        if decrypt(login_data[0]['password']) == p:
            return login_data[0]['id'], login_data[0]['email'], login_data[0]['company']
        else:
            return False, False, False
    else:
        return False, False, False


def get_sims_data(conn):
    return conn.execute("SELECT *, datetime(created_at, 'unixepoch', 'localtime') AS created_at_dt "
                        "FROM sim WHERE id_user=%i" % session['id_user']).fetchall()


def get_single_sim_data(conn, id_sim):
    cur = conn.cursor()
    cur.execute("SELECT *, datetime(created_at, 'unixepoch', 'localtime') AS created_at_dt "
                "FROM sim WHERE id=%i" % int(id_sim))

    for row in cur.fetchall():
        return list(row)


def get_buses_models_data(conn):
    cur = conn.cursor()
    cur.execute("SELECT id, code, name, features FROM bus_model ORDER BY id ASC")
    bus_data = []
    for row in cur.fetchall():
        if row[3] is not None:
            row_dict = json.loads(row[3])
        else:
            row_dict = {}
        row_dict['id'] = row[0]
        row_dict['code'] = row[1]
        row_dict['name'] = row[2]
        bus_data.append(row_dict)
    return bus_data


def get_available_buses_models():
    bm_files = []
    for bm_file in glob.glob('static/bus-types/*.json', recursive=True):
        bm_files.append(bm_file.split(os.sep)[-1].replace('.json', ''))
    return bm_files


def get_single_bus_model_data(conn, id_bus_model):
    cur = conn.cursor()
    cur.execute("SELECT id, code, name, features FROM bus_model WHERE id=%i" % id_bus_model)
    for row in cur.fetchall():
        if row[3] is not None:
            row_dict = json.loads(row[3])
        else:
            row_dict = {}
        row_dict['id'] = row[0]
        row_dict['code'] = row[1]
        row_dict['name'] = row[2]
        break
    return row_dict


def is_logged():
    if 'username' in session.keys():
        return True
    else:
        return False


def insert_user(conn, username, email, password):
    cur = conn.cursor()
    res = cur.execute("INSERT INTO user (username, email, password) VALUES (?, ?, ?)",
                      (username, email, encrypt(password)))
    conn.commit()
    return res


def get_companies_lines_list():
    cl_list = []
    for cl_file in glob.glob('static/time-energy/*.json', recursive=True):
        cl_list.append(cl_file.split(os.sep)[-1].replace('.json', '').replace('-time-energy-', '__'))
    return cl_list


def delete_file_sim(file_path):
    try:
        os.unlink(file_path)
    except Exception as e:
        print('ERROR: Unable to delete file %s' % file_path)


def delete_sim(conn, sim_metadata):
    conn.execute('DELETE FROM sim WHERE id = ?', (sim_metadata[0],))
    conn.commit()

    created_at = sim_metadata[2]

    id_file = '%s_%s' % (session['id_user'], created_at)

    delete_file_sim('static/input-csv/%s.csv' % id_file)
    delete_file_sim('static/sim-config/%s.json' % id_file)
    delete_file_sim('static/output/%s.zip' % id_file)
    delete_file_sim('static/output-df/%s.csv' % id_file)
    delete_file_sim('static/output-bsize/%s.csv' % id_file)
    delete_file_sim('static/plot/%s.png' % id_file)


def delete_bus_model(conn, bus_model_id):
    conn.execute('DELETE FROM bus_model WHERE id = ?', (bus_model_id,))
    conn.commit()


def calc_a(i, t):
    q = 1 + i
    return (np.power(q, t) * i) / (np.power(q, t) - 1)


def calculate_economical_parameters(capex_features, opex_features):
    # Calculate the CAPEX costs
    interest_rate = float(capex_features['capex_interest_rate']) / 1e2
    a_bus = calc_a(interest_rate, float(capex_features['capex_bus_lifetime']))
    a_batt = calc_a(interest_rate, float(capex_features['capex_battery_lifetime']))
    a_char = calc_a(interest_rate, float(capex_features['capex_charger_lifetime']))

    # todo Capex_additional_fee is the cost for the connection to the grid for a charger (TBD)
    capex_cost = a_bus * float(capex_features['capex_bus_cost']) * float(capex_features['capex_number_buses']) + \
                 a_batt * float(capex_features['capex_battery_cost']) * float(capex_features['capex_number_buses']) + \
                 a_char * float(capex_features['capex_charger_cost']) * float(capex_features['capex_number_chargers']) + \
                 (float(capex_features['capex_additional_fee']) * float(capex_features['capex_number_chargers']))

    # Calculate the OPEX costs
    # todo OPEX still to check
    opex_cost = (float(opex_features['opex_buses_maintainance']) +
                 float(opex_features['opex_buses_efficiency']) * float(opex_features['opex_energy_tariff'])) * \
                float(opex_features['opex_annual_usage']) * float(capex_features['capex_number_buses'])

    return capex_cost, opex_cost


def get_lines_daytypes_from_data_file(sim_file_path):
    lines = set()
    days_types = set()
    terminals = {}

    with open(sim_file_path) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            line_id = row['line_id']
            day_type = row['day_type']
            lines.add(line_id)
            days_types.add(day_type)
            if (line_id, day_type) not in terminals:
                terminals[(line_id, day_type)] = set()
            terminals[(line_id, day_type)].add(row['starting_city'])
            terminals[(line_id, day_type)].add(row['arrival_city'])

    lines = sorted(list(lines))
    days_types = sorted(list(days_types))
    return lines, days_types


def get_terminals(sim_metadata):
    lines = []
    for k in sim_metadata.keys():
        if len(k) > 5 and k[0:5] == 'line_':
            lines.append(k[5:])
    terminals = []
    with open(sim_metadata['data_file']) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if row['line_id'] in lines:
                terminals.append(row['arrival_city'])
    return list(set(terminals))


def filter_pars(pars, filter_string):
    res = {}
    for k in pars.keys():
        if filter_string in k:
            res[k] = pars[k]
    return res


def run_sim(conn, main_cfg, pars, bus_model_data, terminals_selected, terminals_metadata, distances_matrix):
    cur = conn.cursor()

    capex_pars = filter_pars(pars, 'capex')
    opex_pars = filter_pars(pars, 'opex')

    # Prepare the line string
    str_routes = ''
    for elem in pars.keys():
        if elem[0:5] == 'line_':
            str_routes += ',' + elem[5:]
    str_routes = str_routes[1:]

    ts = pars['data_file'].replace('.csv', '').split(os.sep)[-1].split('_')[-1]

    sim_cfg_filename = csb.configuration(csv_file_path=pars['data_file'],
                                         company=session['company_user'],
                                         route_number=str_routes,
                                         charging_locations=[],
                                         day_type=pars['day_type'],
                                         t_horizon=main_cfg['simSettings']['modelTimesteps'],
                                         p_max=float(pars['p_max']),
                                         pd_max=float(pars['pd_max']),
                                         depot_charging=main_cfg['simSettings']['chargingAtDeposit'],
                                         optimize_for_each_bus=False,
                                         bus_model_data=bus_model_data,
                                         terminals_selected=terminals_selected,
                                         terminals_metadata=terminals_metadata,
                                         distances_matrix=distances_matrix)

    e = Env("gurobi.log", params={'MemLimit': 30,
                                  'PreSparsify': 1,
                                  'MIPFocus': 3,
                                  'NodefileStart': 0.5,
                                  'Heuristics': 0.3,
                                  'Presolve': 1,
                                  'NoRelHeurTime': 60,
                                  'NonConvex': 2,
                                  'MIPGap': 0.2})

    # Load data files
    with open(sim_cfg_filename, 'r') as f:
        config_file = json.load(f)

    model = MILP(config=config_file,
                 env=e,
                 opt_battery_for_each_bus=False,
                 default_assignment=True,
                 partial_assignment=[],
                 non_overlap_charging=False)

    model.optimize()
    if model.status == GRB.INFEASIBLE:
        return False

    # Define the variables to save in the database
    res_var = ['bp', 'bs', 'bi', 'Ct', 'u', 'x', 'yd', 'SOC']
    var_info = [(v.varName, v.X) for v in model.getVars() if (v.X != 0) and any([v.varName.startswith(s) for s in res_var])]

    # Define the CSV output file path
    csv_output_file = pars['data_file'].replace('input-csv', 'output')

    # Write the simulation output in a CSV file
    res_file = csv_output_file.split(os.sep)[-1]
    with open(res_file, 'w', newline='') as f:
        wr = csv.writer(f, delimiter=";")
        wr.writerows(var_info)

    # Create the plot
    ret_plot_data = schedule_plot(res_file)

    # Create the final dataframe result
    schedule_drivers(res_file)

    # Archive the CSV file in a zip
    with zipfile.ZipFile(csv_output_file.replace('.csv', '.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as zip:
        zip.write(res_file)
    os.unlink(res_file)

    # Update CAPEX and OPEX parameters with some output given by the simulation
    df_bsize_filename = 'static/output-bsize/%s_%s.csv' % (session['id_user'], ts)
    df_bsize = pd.read_csv(df_bsize_filename)
    capex_pars['capex_number_buses'] = len(df_bsize)
    # We assume that number of chargers is equal to the number of buses
    capex_pars['capex_number_chargers'] = len(df_bsize)

    cur.execute("INSERT INTO sim (id_user, created_at, company, line, day_type, battery_size, max_charging_power, "
                "elevation_deposit, elevation_starting_station, elevation_arrival_station, capex_pars, opex_pars, "
                "max_charging_powers) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (int(session['id_user']), int(ts), session['company_user'], str_routes, pars['day_type'],
                 float(bus_model_data['batt_pack_capacity']), float(pars['p_max']), 0, 0, 0, json.dumps(capex_pars),
                 json.dumps(opex_pars), json.dumps(ret_plot_data['df_charge'])))
    conn.commit()
    return True


def create_new_bus_model(pars):
    bus_name = pars['bus_name']
    del pars['bus_name']

    conn = get_db_connection()
    cur = conn.cursor()

    with open('static/bus-types/%s.json' % pars['bus_type'], 'r') as f:
        default_pars = json.load(f)
    pars.update(default_pars)

    cur.execute("INSERT INTO bus_model (code, name, features) "
                "VALUES (?, ?, ?)",
                (pars['bus_type'], bus_name, json.dumps(pars)))
    conn.commit()
    conn.close()


def update_bus_model(conn, pars):
    bus_id = pars['id']
    bus_name = pars['name']
    del pars['id']
    del pars['name']
    cur = conn.cursor()

    cur.execute("UPDATE bus_model SET name=?, features=? WHERE id=?",
                (bus_name, json.dumps(pars), bus_id))
    conn.commit()


def read_solution(solution_file_path, sep=';'):
    file = os.path.join(solution_file_path)
    df = pd.read_csv(file, sep=sep, header=None)

    df.columns = ['var', 'val']

    df[['var', 'index1', 'index2']] = df['var'].str.split(r'\[|\]', expand=True, regex=True)
    dfx = df[df['var'] == 'x'].copy()

    dfx[['n1', 'n2', 'bus']] = dfx['index1'].str.split(',', expand=True)
    dfx = dfx.reset_index()[['var', 'bus', 'n1', 'n2', 'val']]
    dfs = df[df['var'] == 'SOC'].copy()

    dfs[['bus', 'node']] = dfs['index1'].str.split(',', expand=True)
    dfs = dfs.reset_index()[['var', 'bus', 'node', 'val']]

    dfc = df[df['var'] == 'Ct'].copy()
    if not dfc.empty:
        dfc[['n1']] = dfc['index1'].str.split(',', expand=True)
        dfc = dfc.reset_index()[['var', 'n1', 'val']]
    return dfx, dfs, dfc


def read_time_windows(fname):
    # Load data files
    with open(f"{fname}", "r") as f:
        c = json.load(f)
    service_start = {i["index"]: i["t_start"] for i in c['trips_info']}
    service_end = {i["index"]: i["t_end"] for i in c['trips_info']}
    service_relocation_time = np.array(c['service_time'])
    depot_origin = {i["index"]: i["n_start"] for i in c['depot_origin']}
    depot_destination = {i["index"]: i["n_start"] for i in c['depot_destination']}
    return service_start, service_end, service_relocation_time, list(depot_origin.keys()), list(depot_destination.keys())


def read_battery_size(solution_file_path, sep=';'):
    config_name = 'static/sim-config/%s' % solution_file_path.split(os.sep)[-1].replace('.csv', '.json')
    with open(config_name, 'r') as f:
        c = json.load(f)

    df = pd.read_csv(solution_file_path, sep=sep, header=None)
    df.columns = ['var', 'val']
    df[['var', 'index1', 'index2']] = df['var'].str.split(r'\[|\]', expand=True, regex=True)
    dfb = df[df['var'] == 'bp'].copy().reset_index()
    dfb = dfb[['index1', 'val']]

    if 'optimize_for_each_bus' in c.keys() and c['optimize_for_each_bus']:
        dfb = dfb[dfb['index1'].isin(c["vehicle_ids"])]
        dfb.columns = ['Bus id', 'Battery side (kWh)']
        dfb["Battery packs"] = dfb['Battery side (kWh)'].apply(lambda x: math.ceil(x / c["Battery pack size"]))
    else:
        size = dfb.val.values[0]
        dfb = pd.DataFrame(list(zip(c["vehicle_ids"], [size for _ in c["vehicle_ids"]])), columns=['Bus id', 'Battery side (kWh)'])
        dfb["Battery packs"] = dfb['Battery side (kWh)'].apply(lambda x: math.ceil(x / c["Battery pack size"]))

    df_file_name = 'static/output-bsize/%s' % solution_file_path.split(os.sep)[-1]
    dfb.to_csv(df_file_name)
    return dfb


def schedule_plot(solution_file_path, charging_blocks=3):
    returned_data = {}
    config_name = 'static/sim-config/%s' % solution_file_path.split(os.sep)[-1].replace('.csv', '.json')
    with open(config_name, 'r') as f:
        c = json.load(f)

    df, df_soc, dfc = read_solution(solution_file_path)
    t_start, t_end, times, depot_start, depot_end = read_time_windows(config_name)
    bat_size = read_battery_size(solution_file_path)
    df_final = df_soc[df_soc.node == str(depot_end[0])]

    df['start'] = 0
    df['end'] = 0
    df['duration'] = 0

    for index, row in df.iterrows():
        if int(row.n1) in depot_start:
            n1 = int(row.n1)
            df.at[index, 'end'] = t_start[int(row.n2)]
            df.at[index, 'duration'] = times[n1, int(row.n2)]
            df.at[index, 'start'] = t_start[int(row.n2)] - times[n1, int(row.n2)]
        else:
            if int(row.n1) < c['#trips']:
                n1 = int(row.n1)
            else:
                n1 = int(row.n1) - c['#trips']
            df.at[index, 'start'] = t_start[n1]
            df.at[index, 'duration'] = times[n1, int(row.n2)]
            df.at[index, 'end'] = t_start[n1] + times[n1, int(row.n2)]

    if not dfc.empty:
        dfc['start'] = 0
        dfc['duration'] = dfc['val']
        dfc = dfc.merge(df[['n1', 'bus']], how='left', on=['n1'])

        for index, row in dfc.iterrows():
            rem = int(row.n1) // c['#trips']
            dfc.at[index, 'start'] = t_end[int(row.n1) - rem * c['#trips']]
            dfc.at[index, 'var'] = "Dt" if rem == 2 else "Ct"

        dfc['end'] = dfc['start'] + dfc['duration']
        dfc = dfc[['var', 'bus', 'start', 'val', 'duration', 'end']]

    df = df.sort_values(['bus', 'start', 'end']).reset_index().drop(['index'], axis=1)
    max_step = max(df.end)
    step_mask = {}

    for bus in set(df.bus):
        step_mask[bus] = np.ones(max_step, )

    for bus in set(df.bus):
        earliest_start = min(df[(df.bus == bus)]['start'])
        latest_finish = max(df[(df.bus == bus)]['end'])
        step_mask[bus][earliest_start: latest_finish] = 0

    df_charge = {}

    for b in set(df.bus):
        b_size = bat_size.loc[bat_size['Bus id'] == b, 'Battery packs'].values[0] * c["Battery pack size"]
        df_charge[b] = np.ceil((1 - float(df_final[df_final.bus == b]['val'].values[0])) * b_size)  # * 60 / c['max_depot_charging_power'])
    returned_data['df_charge'] = df_charge

    model = Model('charge')
    # Bus i charges at time j.
    x = model.addVars(set(df.bus), np.arange(max_step), vtype=GRB.BINARY, name='x')
    # Total charging events at time j.
    y = model.addVars(np.arange(max_step), lb=0, name='y')
    # Charging power of bus i at timestep j
    p = model.addVars(set(df.bus), np.arange(max_step), lb=1e-6, ub=150, name='p')
    z2 = model.addVar(name='z2')

    cost = model.addVar(name='cost')

    for k, v in step_mask.items():
        model.addConstrs(x[k, t] == 0 for t in range(len(v)) if v[t] == 0)

    # Every bus should charge to 100% SOC
    model.addConstrs(quicksum(x[k, t] * p[k, t] for t in np.arange(max_step)) == df_charge[k] * 60 for k in set(df.bus))

    # Total charging power at time t
    model.addConstrs(y[t] == quicksum(x[k, t] * p[k, t] for k in set(df.bus)) for t in np.arange(max_step))
    # Maximum rate of change of charging power is 1kW per second
    model.addConstrs(p[k, t] * x[k, t] - p[k, t + 1] * x[k, t + 1] <= 10 for t in np.arange(max_step - 1) for k in set(df.bus))
    model.addConstrs(p[k, t] * x[k, t] - p[k, t + 1] * x[k, t + 1] >= -10 for t in np.arange(max_step - 1) for k in set(df.bus))

    # Upper bound on the sum of charging power
    model.addConstrs(z2 >= quicksum(p[k, t] for k in set(df.bus)) for t in np.arange(max_step))

    model.addConstr(cost == z2)
    model.setObjective(cost, sense=GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible!!!!")

    sol_x = [(v.varName, v.X) for v in model.getVars() if (v.X != 0) and any([v.varName.startswith(s) for s in ['x']])]
    solution = pd.DataFrame(sol_x, columns=['var', 'val'])

    sol_p = [(v.varName, v.X) for v in model.getVars() if (v.X != 0) and any([v.varName.startswith(s) for s in ['p']])]
    power = pd.DataFrame(sol_p, columns=['var', 'val'])

    solution[['var', 'index1', 'index2']] = solution['var'].str.split(r'\[|\]', expand=True, regex=True)
    solution[['bus', 'start']] = solution['index1'].str.split(',', expand=True)
    solution = solution.reset_index()[['var', 'bus', 'start', 'val']]
    solution['start'] = solution['start'].astype('int')
    solution['duration'] = solution['val'].astype('int')
    solution['end'] = solution['start'] + solution['duration']
    solution = pd.concat([solution, dfc], axis=0, ignore_index=True)

    power[['var', 'index1', 'index2']] = power['var'].str.split(r'\[|\]', expand=True, regex=True)
    power[['bus', 'start']] = power['index1'].str.split(',', expand=True)
    power = power.reset_index()[['var', 'bus', 'start', 'val']]
    power['start'] = power['start'].astype('int')

    temp_x = pd.DataFrame(columns=df.columns)

    for index, row in df.iterrows():
        if df.at[index, 'end'] > 1440:
            bus_id = df.at[index, 'bus']

            if len(df[(df.bus == bus_id) & (df.end < 200)]) > 0:
                max_t = max(df[(df.bus == bus_id) & (df.end < 200)]['end'])
                min_t = min(df[(df.bus == bus_id) & (df.start > 200)]['start'])

                # Change a time of dead-heading from depot origin
                df.loc[(df.n1 == str(depot_start[0])) & (df.bus == bus_id), "start"] = min_t - df.loc[
                    (df.n1 == str(depot_start[0])) & (df.bus == bus_id), "duration"]

                df.loc[(df.n1 == str(depot_start[0])) & (df.bus == bus_id), "end"] = \
                    df.loc[(df.n1 == str(depot_start[0])) & (df.bus == bus_id), "start"] + df.loc[
                        (df.n1 == str(depot_start[0])) & (df.bus == bus_id), "duration"]

                # Change dead-heading time to depot
                df.loc[(df.n2 == str(depot_end[0])) & (df.bus == bus_id), "start"] = max_t
                df.loc[(df.n2 == str(depot_end[0])) & (df.bus == bus_id), "end"] = max_t + \
                                                                                   df.loc[(df.n2 == str(depot_end[0])) & (df.bus ==
                                                                                                                          bus_id),
                                                                                   "duration"]

    j = 0
    for index, row in df.iterrows():
        if df.at[index, 'end'] > 1440:
            bus_id = df.at[index, 'bus']
            rem = df.at[index, 'end'] - 1440
            df.at[index, 'end'] = 1440
            df.at[index, 'duration'] = 1440 - df.at[index, 'start'] if df.at[index, 'start'] < 1440 else 0

            temp_x.at[j, 'var'] = df.at[index, 'var']
            temp_x.at[j, 'bus'] = df.at[index, 'bus']
            temp_x.at[j, 'n1'] = df.at[index, 'n1']
            temp_x.at[j, 'n2'] = df.at[index, 'n2']
            temp_x.at[j, 'val'] = df.at[index, 'val']
            temp_x.at[j, 'start'] = 0
            temp_x.at[j, 'end'] = rem
            temp_x.at[j, 'duration'] = rem
            j += 1

    j = 0
    temp_y = pd.DataFrame(columns=solution.columns)

    for index, row in solution.iterrows():
        if solution.at[index, 'end'] > 1440:
            rem = solution.at[index, 'end'] - 1440

            if solution.at[index, 'start'] > 1440:
                solution.at[index, 'start'] = solution.at[index, 'start'] - 1440
                solution.at[index, 'end'] = solution.at[index, 'end'] - 1440
                solution.at[index, 'duration'] = solution.at[index, 'end'] - solution.at[index, 'start']
            else:
                solution.at[index, 'end'] = 1440
                solution.at[index, 'duration'] = 1440 - solution.at[index, 'start']

                temp_y.at[j, 'var'] = solution.at[index, 'var']
                temp_y.at[j, 'bus'] = solution.at[index, 'bus']
                temp_y.at[j, 'val'] = solution.at[index, 'val']
                temp_y.at[j, 'start'] = 0
                temp_y.at[j, 'end'] = rem
                temp_y.at[j, 'duration'] = rem
            j += 1

    df = pd.concat([df, temp_x], axis=0, ignore_index=True)
    df_charge = pd.concat([solution, temp_y], axis=0, ignore_index=True)
    df_depot_charge = df_charge[df_charge['var'].isin(['Dt', 'x'])]
    df_panto_charge = df_charge[df_charge['var'].isin(['Ct'])]

    df_trips = df[(df.n1 != str(depot_start[0])) & (df.n2 != str(depot_end[0])) & (~df['n1'].isna())]
    df_dead = df[((df.n1 == str(depot_start[0])) | (df.n2 == str(depot_end[0]))) & (~df['n1'].isna())]
    x_ticks = [i * 30 for i in range(49)]

    plt.figure(figsize=(12, 4.5))
    # Define the color gradient from white to red
    colors = ['#FFFFFF', '#FF0000']
    values = [0, 150]
    color_dict = {value: color for value, color in zip(values, colors)}
    colormap = matplotlib.colors.LinearSegmentedColormap.from_list('white_to_red', list(color_dict.values()))

    buses = sorted(list(set(df_depot_charge.bus)))
    for b in buses:
        for s in df_depot_charge.loc[df_depot_charge.bus == b, 'start'].values:
            for t in range(int(df_depot_charge.loc[(df_depot_charge.bus == b) & (df_depot_charge.start == s), 'duration'].values[0])):
                ci = power.loc[(power.bus == b) & (power.start == s + t), 'val'].values[0]
                plt.barh(y=b, left=s + t, width=1, height=0.8, color=colormap(ci))

    plt.barh(y=df_panto_charge.bus, left=df_panto_charge.start, width=df_panto_charge.duration, color='#FFB455',
             label='Pantograph Charging',
             height=0.8)
    plt.barh(y=df_dead.bus, left=df_dead.start, width=df_dead.duration, color='grey', label='Dead-heading', height=0.8)
    plt.barh(y=df_trips.bus, left=df_trips.start, width=df_trips.duration, color='black', label='Trips', height=0.8)
    plt.gca().invert_yaxis()
    plt.xticks(ticks=x_ticks[::3], fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=10, loc='lower left')
    plt.tick_params(axis='y', which='major', pad=6)

    scheduling_plot_file_name = 'static/plot/%s' % solution_file_path.split(os.sep)[-1].replace('.csv', '.png')
    plt.savefig(scheduling_plot_file_name, dpi=300, format=None, metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto', backend=None)

    plt.figure(figsize=(12, 4.5))
    plt.plot(power.groupby(['start'])['val'].sum().values, '-r', label='Power')
    plt.xlabel('Timestep')
    plt.ylabel("Power (kW)")
    cp_plot_file_name = scheduling_plot_file_name.replace('.png', '_charge_profile.png')
    plt.savefig(cp_plot_file_name, dpi=300, format=None, metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto', backend=None)

    return returned_data


def generate_graph(nodes_list, edges_list):
    graph = nx.Graph()

    for n0 in nodes_list:
        graph.add_node(n0)
    for k, v in edges_list.items():
        for e in v:
            graph.add_edge(k, e)
    return graph


def create_edge_list(edges, data_frame, start, end):
    e_list = defaultdict(list)
    for i, j in edges:
        bus_i = data_frame.loc[data_frame.n1 == i.split('_')[0], 'bus'].values[0]
        bus_j = data_frame.loc[data_frame.n1 == j.split('_')[0], 'bus'].values[0]

        # If (i, j) or (j, i) is in node_pairs there is no edge.
        if 'S' in i.split('_')[0]:
            tsi = data_frame.loc[data_frame.n1 == i.split('_')[0], 'start'].values[0]
            tei = data_frame.loc[data_frame.n1 == i.split('_')[0], 'end'].values[0]
        else:
            tsi = start[int(i.split('_')[0])]
            tei = end[int(i.split('_')[0])]

        if 'S' in j.split('_')[0]:
            tsj = data_frame.loc[data_frame.n1 == j.split('_')[0], 'start'].values[0]
        else:
            tsj = start[int(j.split('_')[0])]

        if 'E' in j.split('_')[-1]:
            tej = data_frame.loc[data_frame.n2 == j.split('_')[-1], 'end'].values[0]
        else:
            tej = end[int(j.split('_')[-1])]

        if (np.abs(tej - tsi) >= 270) or (tsj < tei) or ('E' in i) or ('S' in j) or (bus_i != bus_j):
            e_list[i].append(j)
    return e_list


def generate_shift_assignment_graph(c, data_frame):
    nodes = list(set(c.values()))
    earliest_start = defaultdict(list)
    latest_end = defaultdict(list)

    graph = nx.Graph()
    for n0 in nodes:
        graph.add_node(n0)

    for node, color in c.items():
        n1, n2 = node.split('_')
        earliest_start[color].append(data_frame.loc[(data_frame.n1 == n1) & (data_frame.n2 == n2), 'start'].values[0])
        latest_end[color].append(data_frame.loc[(data_frame.n1 == n1) & (data_frame.n2 == n2), 'end'].values[0])

    for p1, p2 in itertools.product(earliest_start.keys(), earliest_start.keys()):
        if p1 != p2:
            range_1 = set(range(min(earliest_start[p1]), max(latest_end[p1])))
            range_2 = range(min(earliest_start[p2]), max(latest_end[p2]))
            if (np.abs(max(latest_end[p1]) - min(earliest_start[p2])) >= 45) and not range_1.intersection(range_2):
                pass
            else:
                graph.add_edge(p1, p2)

    return graph


def schedule_drivers(solution_file_path):
    config_name = 'static/sim-config/%s' % solution_file_path.split(os.sep)[-1].replace('.csv', '.json')

    df = u.read_schedule_from_solution(solution_file_path)
    t_start, t_end, times, depot_start, depot_end = read_time_windows(config_name)
    df = u.merge_time_schedule(df, t_start, times, depot_start)

    for index, row in df.iterrows():
        if int(row.n1) in depot_start:
            df.at[index, 'n1'] = 'S' + row.bus
        if int(row.n2) in depot_end:
            df.at[index, 'n2'] = 'E' + row.bus

    # Create a list of all (n1, n2) pairs in df
    node_list = []  # These are the nodes in the graph
    for index, row in df.iterrows():
        node_list.append((row.n1 + "_" + row.n2))

    # Set up edge list of the graph
    edges = list(combinations(node_list, 2))

    # There is an edge between two nodes if those two trips cannot co-exist in the same work shift.
    edge_list = create_edge_list(edges=edges, start=t_start, end=t_end, data_frame=df)
    G = generate_graph(node_list, edge_list)
    color = nx.coloring.greedy_color(G)

    # Number of colors represent the minimum number of valid shifts required to cover all the trips.
    # Now we need to assign each trip to a driver, given there is a 45-min interval between two shifts for the same driver.
    Gd = generate_shift_assignment_graph(color, df)
    color_drivers = nx.coloring.greedy_color(Gd)

    df_res = df.copy()
    df_res['shifts'] = None
    df_res['driver'] = None

    for index, row in df_res.iterrows():
        df_res.at[index, 'shifts'] = int(color[row.n1 + "_" + row.n2])

    df_res['driver'] = df_res.shifts.apply(lambda x: color_drivers[x])
    df_res = df_res[['bus', 'n1', 'start', 'n2', 'end', 'driver']]
    df_res.columns = ['bus#', 'Terminal start', 'Start time', 'Terminal end', 'End time', 'Driver#']

    df_file_name = 'static/output-df/%s' % solution_file_path.split(os.sep)[-1]
    df_res.to_csv(df_file_name)
    print(f'Minimum number of drivers required: {len(set(df_res["Driver#"]))}')


def clean_terminals(conn):
    cur = conn.cursor()
    cur.execute("DELETE FROM terminal WHERE company='%s'" % session['company_user'])
    conn.commit()


def clean_distances(conn):
    cur = conn.cursor()
    cur.execute("DELETE FROM distance WHERE company='%s'" % session['company_user'])
    conn.commit()


def update_terminals(conn, terminals_file_path):
    df = pd.read_csv(terminals_file_path)
    df = df.assign(id=df.index)
    df = df.assign(company=np.array([session['company_user'] for _ in range(len(df.index))]))
    df = df.rename(columns={'terminal_station': 'name'})
    df = df.reindex(columns=['id', 'name', 'company', 'elevation_m', 'is_charging_station'])
    df.to_sql('terminal', conn, if_exists='replace', index=False)

    terms = get_terminals_metadata(conn)
    terms_dict = {}
    for term in terms:
        terms_dict[term[1]] = term[0]
    return terms_dict


def update_distances(conn, distances_file_path, terms_dict):
    df = pd.read_csv(distances_file_path)
    starting_stations_ids = []
    arrival_stations_ids = []
    for _, row in df.iterrows():
        starting_stations_ids.append(terms_dict[row['starting_station']])
        arrival_stations_ids.append(terms_dict[row['arrival_station']])

    df = df.assign(id_starting_station=starting_stations_ids)
    df = df.assign(id_arrival_station=arrival_stations_ids)
    df = df.assign(company=np.array([session['company_user'] for _ in range(len(df.index))]))
    df = df.drop(['starting_station', 'arrival_station'], axis=1)
    df = df.reindex(columns=['id_starting_station', 'id_arrival_station', 'distance_km', 'avg_travel_time_min', 'company'])
    df.to_sql('distance', conn, if_exists='replace', index=False)


def get_terminals_metadata(conn):
    return conn.execute("SELECT * FROM terminal WHERE company='%s'" % session['company_user']).fetchall()


def get_distances_matrix(conn):
    return conn.execute("SELECT * FROM distance WHERE company='%s'" % session['company_user']).fetchall()


def get_terminals_metadata_dict(conn):
    terms = get_terminals_metadata(conn)
    terms_dict = {}
    for term in terms:
        terms_dict[term[1]] = {'id': term[0], 'elevation': term[3], 'is_charging_station': term[4]}
    return terms_dict


def get_distances_matrix_dict(conn):
    distances = get_distances_matrix(conn)
    distances_dict = {}
    for d in distances:
        distances_dict[d[0], d[1]] = {'distance_km': d[2], 'avg_travel_time_min': d[3]}
    return distances_dict


# Route for handling the login page logic
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        conn = get_db_connection()
        id_user, email_user, company_user = check_login_data(conn, request.form['username'], request.form['password'])
        conn.close()
        if id_user is False:
            error = 'Invalid Credentials. Please try again.'
        else:
            session['id_user'] = id_user
            session['email_user'] = email_user
            session['company_user'] = company_user
            session['username'] = request.form['username']
            session['password'] = request.form['password']
            return redirect(url_for('index'))
    return render_template('login.html', error=error)


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.pop('username', None)
    session.pop('password', None)
    return redirect(url_for('index'))


# Route for handling the login page logic
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        conn = get_db_connection()
        insert_user(conn, request.form['username'], request.form['email'], request.form['password'])
        conn.close()
        return redirect(url_for('login'))
    return render_template('signup.html', error=error)


@app.route('/', methods=('GET', 'POST'))
def index():
    if is_logged():
        conn = get_db_connection()
        if 'del' in request.args.keys() and 'id' in request.args.keys():
            sim_metadata = get_single_sim_data(conn, int(request.args.to_dict()['id']))
            delete_sim(conn, sim_metadata)

        # Get simulation data
        sims = get_sims_data(conn)
        conn.close()

        return render_template('index.html', sims=sims)
    else:
        return redirect(url_for('login'))


@app.route('/detail', methods=('GET', 'POST'))
def detail():
    if is_logged():
        conn = get_db_connection()
        sim_metadata = get_single_sim_data(conn, int(request.args.to_dict()['id']))
        conn.close()

        df_bsize_filename = 'static/output-bsize/%s_%i.csv' % (session['id_user'], sim_metadata[2])
        df_bsize = pd.read_csv(df_bsize_filename)

        df_data_filename = 'static/output-df/%s_%i.csv' % (session['id_user'], sim_metadata[2])
        df_data = pd.read_csv(df_data_filename)

        data = dict(request.args)
        data['min_num_drivers'] = len(set(df_data["Driver#"]))
        data['df_bsize'] = df_bsize

        # Calculate CAPEX and OPEX costs
        capex_features = json.loads(sim_metadata[11])
        opex_features = json.loads(sim_metadata[12])
        capex, opex = calculate_economical_parameters(capex_features=capex_features, opex_features=opex_features)
        capex_features['capex_cost'] = int(capex / 1e3)
        opex_features['opex_cost'] = int(opex / 1e3)

        return render_template('detail.html', sim_metadata=sim_metadata, data=data, capex_features=capex_features,
                               opex_features=opex_features)
    else:
        return redirect(url_for('login'))


@app.route('/new_sim_step1/', methods=('GET', 'POST'))
def new_sim_step1():
    if is_logged():
        conn = get_db_connection()
        if request.method == 'POST':
            target = 'static/input-csv'

            ts = (datetime.datetime.now(tz=pytz.UTC).timestamp())
            if len(request.files['data_file'].filename) > 0:
                file_name = '%i_%i.csv' % (session['id_user'], ts)
                data_file = '/'.join([target, file_name])
                request.files['data_file'].save(data_file)

                id_bus_model = int(request.form.to_dict()['id_bus_model'])
                lines, days_types = get_lines_daytypes_from_data_file(data_file)
                conn.close()
                return redirect(url_for('new_sim_step2', data_file=data_file, lines=lines, days_types=days_types,
                                        id_bus_model=id_bus_model))
            else:
                buses_models = get_buses_models_data(conn)
                conn.close()
                return render_template('new_sim_step1.html', error='No file uploaded', buses_models=buses_models)
        else:
            buses_models = get_buses_models_data(conn)
            conn.close()
            return render_template('new_sim_step1.html', buses_models=buses_models)
    else:
        return redirect(url_for('login'))


@app.route('/new_sim_step2/', methods=('GET', 'POST'))
def new_sim_step2():
    if is_logged():
        conn = get_db_connection()
        if request.method == 'POST':
            try:
                # Run the simulation and save the output in the DB
                sim_pars = request.form.to_dict()
                run_sim(conn=conn, main_cfg=main_cfg, pars=sim_pars,
                        bus_model_data=get_single_bus_model_data(conn, int(sim_pars['bus_model_id'])),
                        terminals_selected=get_terminals(sim_pars),
                        terminals_metadata=get_terminals_metadata_dict(conn),
                        distances_matrix=get_distances_matrix_dict(conn))
                conn.close()
                return redirect(url_for('index'))
            except Exception as e:
                print('EXCEPTION: %s' % str(e))
                conn = get_db_connection()
                req_dict = request.args.to_dict()
                lines, days_types = get_lines_daytypes_from_data_file(req_dict['data_file'])
                bus_model_data = get_single_bus_model_data(conn, int(req_dict['id_bus_model']))
                conn.close()
                return render_template('new_sim_step2.html',
                                       error='Data file has a wrong format! The simulation cannot be run',
                                       data_file=req_dict['data_file'], lines=lines, days_types=days_types,
                                       bus_model_data=bus_model_data)
        else:
            req_dict = request.args.to_dict()
            lines, days_types = get_lines_daytypes_from_data_file(req_dict['data_file'])
            bus_model_data = get_single_bus_model_data(conn, int(req_dict['id_bus_model']))
            conn.close()
            return render_template('new_sim_step2.html', data_file=req_dict['data_file'], lines=lines,
                                   days_types=days_types, bus_model_data=bus_model_data)
    else:
        return redirect(url_for('login'))


@app.route('/new_bus_model', methods=('GET', 'POST'))
def new_bus_model():
    if is_logged():
        if request.method == 'POST':
            create_new_bus_model(request.form.to_dict())
            return redirect(url_for('company_manager'))
        else:
            available_buses_models = get_available_buses_models()
            return render_template('new_bus_model.html', available_buses_models=available_buses_models)
    else:
        return redirect(url_for('login'))


@app.route('/edit_bus_model', methods=('GET', 'POST'))
def edit_bus_model():
    if is_logged():
        conn = get_db_connection()
        if request.method == 'POST':
            # Update data on db
            new_pars = request.form.to_dict()
            bus_model_data = get_single_bus_model_data(conn, int(request.args.to_dict()['id_bus_model']))
            for k in bus_model_data.keys():
                if k in new_pars.keys():
                    bus_model_data[k] = new_pars[k]
            update_bus_model(conn, bus_model_data)

            # Get data of bus model and pass them to the page
            bus_model_data = get_single_bus_model_data(conn, int(request.args.to_dict()['id_bus_model']))
            conn.close()
            return render_template('edit_bus_model.html', bus_model_data=bus_model_data)
        else:
            # Get data of bus model and pass them to the page
            bus_model_data = get_single_bus_model_data(conn, int(request.args.to_dict()['id_bus_model']))
            conn.close()
            return render_template('edit_bus_model.html', bus_model_data=bus_model_data)
    else:
        return redirect(url_for('login'))


@app.route('/buses_models_list', methods=('GET', 'POST'))
def buses_models_list():
    if is_logged():
        conn = get_db_connection()
        if 'del' in request.args.keys() and 'id_bus_model' in request.args.keys():
            delete_bus_model(conn, request.args['id_bus_model'])

        # Get bus models data
        buses_models = get_buses_models_data(conn)
        conn.close()
        return render_template('buses_models_list.html', buses_models=buses_models)
    else:
        return redirect(url_for('login')) \


@app.route('/company_manager', methods=('GET', 'POST'))
def company_manager():
    if is_logged():
        conn = get_db_connection()
        buses_models = get_buses_models_data(conn)
        if request.method == 'POST':
            target = 'static/tmp'
            terminals_file_path = '/'.join([target, '%i_terminals.csv' % session['id_user']])
            request.files['terminals_file'].save(terminals_file_path)
            distances_file_path = '/'.join([target, '%i_distances.csv' % session['id_user']])
            request.files['distances_file'].save(distances_file_path)

            clean_distances(conn)
            clean_terminals(conn)

            terms_dict = update_terminals(conn, terminals_file_path)
            update_distances(conn, distances_file_path, terms_dict)

            os.unlink(terminals_file_path)
            os.unlink(distances_file_path)

        return render_template('company_manager.html', buses_models=buses_models)
    else:
        return redirect(url_for('login'))


if __name__ == "__main__":
    app.run(debug=True)
