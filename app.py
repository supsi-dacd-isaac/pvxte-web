import csv
import numpy as np
import pandas as pd
import sqlite3
import datetime
import json
import zipfile
import pytz

from collections import defaultdict
from itertools import combinations

import networkx as nx

from cryptography.fernet import Fernet
import matplotlib.pyplot as plt

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
            return login_data[0]['id']
        else:
            return False
    else:
        return False


def get_sims_data(conn):
    return conn.execute("SELECT *, datetime(created_at, 'unixepoch', 'localtime') AS created_at_dt "
                        "FROM sim WHERE id_user=%i" % session['id_user']).fetchall()


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


def delete_sim(conn, created_at):
    conn.execute('DELETE FROM sim WHERE id_user = ? AND created_at = ?', (session['id_user'], created_at))
    conn.commit()

    id_file = '%s_%s' % (session['id_user'], created_at)
    os.unlink('static/input-csv/%s.csv' % id_file)
    os.unlink('static/sim-config/%s.json' % id_file)
    os.unlink('static/output/%s.zip' % id_file)
    os.unlink('static/output-df/%s.csv' % id_file)
    os.unlink('static/output-bsize/%s.csv' % id_file)
    os.unlink('static/plot/%s.png' % id_file)


def run_sim(sim_file_path, main_cfg, pars):
    conn = get_db_connection()
    cur = conn.cursor()

    # Set properly the parameters
    (company, line) = pars['comp_line'].split('__')
    bsize = float(pars['bsize'])
    max_charging_power = int(pars['max_charge_power'])

    # Get user id and timestamp
    (id_user, ts) = sim_file_path.replace('.csv', '').split(os.sep)[-1].split('_')

    # todo opt_all_flag variable should be configurable
    opt_all_flag = False

    sim_cfg_filename = csb.configuration(sim_file_path, company, line, bsize, [],
                                         main_cfg['simSettings']['modelTimesteps'], max_charging_power,
                                         main_cfg['simSettings']['chargingAtDeposit'], opt_all_flag)

    e = Env("gurobi.log", params={'MemLimit': 30,
                                  'PreSparsify': 1,
                                  'MIPFocus': 3,
                                  'NodefileStart': 0.5,
                                  'Heuristics': 0.3,
                                  'Presolve': 1,
                                  'NoRelHeurTime': 60,
                                  'NonConvex': 2,
                                  'MIPGap': 0.01})

    start = datetime.datetime.now()
    print("Generated configuration. Setting up optimization model: ", start)

    # Load data files
    with open(sim_cfg_filename, 'r') as f:
        config_file = json.load(f)

    model = MILP(config=config_file,
                 env=e,
                 charging_power=max_charging_power,
                 opt_battery_for_each_bus=False,
                 default_assignment=True,
                 partial_assignment=[],
                 non_overlap_charging=False)

    model.optimize()

    if model.status == GRB.INFEASIBLE:
        return False

    # Define the variables to save in the database
    res_var = ['bp', 'bs', 'bi', 'Ct', 'u', 'x', 'yd', 'SOC']
    varInfo = [(v.varName, v.X) for v in model.getVars() if (v.X != 0) and any([v.varName.startswith(s) for s in res_var])]

    # Define the CSV output file path
    csv_output_file = sim_file_path.replace('input-csv', 'output')

    # Write the simulation output in a CSV file
    res_file = csv_output_file.split(os.sep)[-1]
    with open(res_file, 'w', newline='') as f:
        wr = csv.writer(f, delimiter=";")
        wr.writerows(varInfo)

    # Read the battery size
    battery_size = read_battery_size(res_file, sep=';')

    # Create the plot
    schedule_plot(res_file, battery_size)

    # Create the final dataframe result
    schedule_drivers(res_file)

    # Archive the CSV file in a zip
    with zipfile.ZipFile(csv_output_file.replace('.csv', '.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as zip:
        zip.write(res_file)
    os.unlink(res_file)

    end = datetime.datetime.now()
    print("End time: ", end)
    print('Solution time (seconds): ', (end - start).total_seconds())

    cur.execute("INSERT INTO sim (id_user, created_at, company, line, day_type, battery_size, max_charging_power) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (int(id_user), int(ts), company, line, pars['day_type'], bsize, max_charging_power))
    conn.commit()
    conn.close()
    return True

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
    return dfx, dfs

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
        dfb = pd.DataFrame(list(zip(c["vehicle_ids"], [size for _ in c["vehicle_ids"]])), columns =['Bus id', 'Battery side (kWh)'])
        dfb["Battery packs"] = dfb['Battery side (kWh)'].apply(lambda x: math.ceil(x / c["Battery pack size"]))

    df_file_name = 'static/output-bsize/%s' % solution_file_path.split(os.sep)[-1]
    dfb.to_csv(df_file_name)
    return dfb

def schedule_plot(solution_file_path, bat_size):
    config_name = 'static/sim-config/%s' % solution_file_path.split(os.sep)[-1].replace('.csv', '.json')
    with open(config_name, 'r') as f:
        c = json.load(f)

    df, df_soc = read_solution(solution_file_path)
    t_start, t_end, times, depot_start, depot_end = read_time_windows(config_name)
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
            n1 = int(row.n1)
            df.at[index, 'start'] = t_start[n1]
            df.at[index, 'duration'] = times[n1, int(row.n2)]
            df.at[index, 'end'] = t_start[n1] + times[n1, int(row.n2)]

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
        df_charge[b] = np.ceil((1 - float(df_final[df_final.bus == b]['val'].values[0])) * b_size * 60 / 150)

    model = Model('charge')
    x = model.addVars(set(df.bus), np.arange(max_step), vtype=GRB.BINARY, name='x')
    y = model.addVars(np.arange(max_step), lb=0, name='y')
    z2 = model.addVar(name='z2')

    cost = model.addVar(name='cost')

    for k, v in step_mask.items():
        model.addConstrs(x[k, t] == 0 for t in range(len(v)) if v[t] == 0)

    model.addConstrs(quicksum(x[k, t] for t in np.arange(max_step)) <= df_charge[k] for k in set(df.bus))
    model.addConstrs(quicksum(x[k, t] for t in np.arange(max_step)) >= df_charge[k] for k in set(df.bus))
    model.addConstrs(y[t] == quicksum(x[k, t] for k in set(df.bus)) for t in np.arange(max_step))

    model.addConstr(z2 == max_(y))

    model.addConstr(cost == z2)
    model.setObjective(cost, sense=GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible!!!!")

    res_var = ['x']
    varInfo = [(v.varName, v.X) for v in model.getVars() if (v.X != 0) and any([v.varName.startswith(s) for s in res_var])]
    solution = pd.DataFrame(varInfo, columns=['var', 'val'])

    solution[['var', 'index1', 'index2']] = solution['var'].str.split(r'\[|\]', expand=True, regex=True)
    solution[['bus', 'start']] = solution['index1'].str.split(',', expand=True)
    solution = solution.reset_index()[['var', 'bus', 'start', 'val']]
    solution['start'] = solution['start'].astype('int')
    solution['duration'] = solution['val'].astype('int')
    solution['end'] = solution['start'] + solution['duration']

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

    df_trips = df[(df.n1 != str(depot_start[0])) & (df.n2 != str(depot_end[0])) & ~df['n1'].isna()]
    df_dead = df[(df.n1 == str(depot_start[0])) | (df.n2 == str(depot_end[0])) & ~df['n1'].isna()]
    x_ticks = [i * 30 for i in range(49)]

    plt.figure(figsize=(12, 4.5))
    plt.barh(y=df_charge.bus, left=df_charge.start, width=df_charge.duration, color='#F5CBA7', label='Charging', height=0.8)
    plt.barh(y=df_dead.bus, left=df_dead.start, width=df_dead.duration, color='#C1C1C1', label='Dead-heading', height=0.8)
    plt.barh(y=df_trips.bus, left=df_trips.start, width=df_trips.duration, color='black', label='Trips', height=0.8)
    plt.gca().invert_yaxis()
    plt.xticks(ticks=x_ticks[::3], fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=10, loc='lower left')
    plt.tick_params(axis='y', which='major', pad=6)

    plot_file_name = 'static/plot/%s' % solution_file_path.split(os.sep)[-1].replace('.csv', '.png')
    plt.savefig(plot_file_name, dpi=300, format=None, metadata=None, bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto', backend=None)

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

        if (np.abs(tej - tsi) >= 270) or (tsj < tei) or ('E' in i) or ('S' in j):
            e_list[i].append(j)
    return e_list


def generate_shift_assignment_graph(c, data_frame):
    nodes = list(set(c.values()))
    earliest_start = defaultdict(list)
    latest_end = defaultdict(list)

    graph = nx.Graph()
    for n0 in nodes:
        graph.add_node(n0)

    for x, y in c.items():
        n1, n2 = x.split('_')
        earliest_start[y].append(data_frame.loc[(data_frame.n1 == n1) & (data_frame.n2 == n2), 'start'].values[0])
        latest_end[y].append(data_frame.loc[(data_frame.n1 == n1) & (data_frame.n2 == n2), 'end'].values[0])

    for p1, p2 in itertools.product(earliest_start.keys(), earliest_start.keys()):
        if p1 != p2:
            if np.abs(max(latest_end[p1]) - min(earliest_start[p2])) >= 45:
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


# Route for handling the login page logic
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        conn = get_db_connection()
        login_result = check_login_data(conn, request.form['username'], request.form['password'])
        conn.close()
        if login_result is False:
            error = 'Invalid Credentials. Please try again.'
        else:
            session['id_user'] = login_result
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
        if 'del' in request.args.keys() and 'created_at' in request.args.keys():
            delete_sim(conn, request.args['created_at'])

        # Get simulation data
        sims = get_sims_data(conn)
        conn.close()

        return render_template('index.html', sims=sims)
    else:
        return redirect(url_for('login'))

@app.route('/detail', methods=('GET', 'POST'))
def detail():
    if is_logged():
        df_bsize_filename = 'static/output-bsize/%s_%s.csv' % (session['id_user'], request.args['created_at'])
        df_bsize = pd.read_csv(df_bsize_filename)

        df_data_filename = 'static/output-df/%s_%s.csv' % (session['id_user'], request.args['created_at'])
        df_data = pd.read_csv(df_data_filename)

        data = dict(request.args)
        data['min_num_drivers'] = len(set(df_data["Driver#"]))
        data['df_bsize'] = df_bsize
        return render_template('detail.html', data=data)
    else:
        return redirect(url_for('login'))

@app.route('/new_sim/', methods=('GET', 'POST'))
def new_sim():
    if is_logged():
        if request.method == 'POST':
            target = 'static/input-csv'

            ts = (datetime.datetime.now(tz=pytz.UTC).timestamp())
            if len(request.files['data_file'].filename) > 0:
                file_name = '%i_%i.csv' % (session['id_user'], ts)
                file_destination = '/'.join([target, file_name])
                request.files['data_file'].save(file_destination)

                # Run the simulation and save the output in the DB
                try:
                    run_sim(sim_file_path=file_destination, main_cfg=main_cfg, pars=request.form)
                except Exception as e:
                    print('EXCEPTION: %s' % str(e))
                    return render_template('new_sim.html',
                                           error='Data file has a wrong format! The simulation cannot be run',
                                           comp_lines=get_companies_lines_list())

                return redirect(url_for('index'))
            else:
                return render_template('new_sim.html', error='No file uploaded', comp_lines=get_companies_lines_list())
        return render_template('new_sim.html', comp_lines=get_companies_lines_list())
    else:
        return redirect(url_for('login'))


# @app.route('/<int:id_user>/<int:ts>/delete/', methods=('POST',))
# def delete(id_user, ts):
#     conn = get_db_connection()
#     conn.execute('DELETE FROM sim WHERE id_user = ? AND created_at = ?', (id_user, ts))
#     conn.commit()
#     conn.close()
#     flash('Successfully deleted!')
#     return redirect(url_for('index'))

# @app.route('/<int:id>/edit/', methods=('GET', 'POST'))
# def edit(id):
#
#     if request.method == 'POST':
#         title = request.form['title']
#         content = request.form['content']
#
#         if not title:
#             flash('Title is required!')
#
#         elif not content:
#             flash('Content is required!')
#
#         else:
#             conn = get_db_connection()
#             conn.execute('UPDATE posts SET title = ?, content = ?'
#                          ' WHERE id = ?',
#                          (title, content, id))
#             conn.commit()
#             conn.close()
#             return redirect(url_for('index'))
#
#     return render_template('edit.html', post=post)
#
# @app.route('/<int:id>/delete/', methods=('POST',))
# def delete(id):
#     conn = get_db_connection()
#     conn.execute('DELETE FROM posts WHERE id = ?', (id,))
#     conn.commit()
#     conn.close()
#     flash('"{}" was successfully deleted!'.format(post['title']))
#     return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)