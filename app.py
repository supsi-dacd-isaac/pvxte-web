import copy
import csv
import http
import os
import sqlite3
import datetime
import json

import babel
import pytz
import glob
import requests

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objs as go

from collections import defaultdict
from cryptography.fernet import Fernet
from flask import Flask, render_template, request, url_for, redirect, session, flash
from flask_babel import Babel, gettext
from flask_wtf import FlaskForm
from flask_wtf.recaptcha import RecaptchaField
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length, Regexp, EqualTo


# Get main conf
with open('static/sims-basic-config/cfg.json', 'r') as f:
    main_cfg = json.load(f)

# Get private conf
with open('static/sims-basic-config/private.json', 'r') as f:
    private_cfg = json.load(f)

# Get key conf
with open('static/sims-basic-config/key.json', 'r') as f:
    key_cfg = json.load(f)# Get key conf

with open('static/sims-basic-config/captcha_keys.json', 'r') as f:
    captcha_keys_cfg = json.load(f)

app = Flask(__name__)

app.config.update(
    SECRET_KEY=key_cfg['appSecretKey'],
    RECAPTCHA_PUBLIC_KEY=captcha_keys_cfg['siteKey'],
    RECAPTCHA_PRIVATE_KEY=captcha_keys_cfg['secretKey']
)

# Set the Babel object for the translations

def get_locale():
    if 'language' in session.keys():
        return session['language']
    else:
        return 'it'

babel = Babel(app)
babel.init_app(app, locale_selector=get_locale)

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def get_not_depo_charging_terminals_ids(terminals_metadata, terminals_selected):
    ids = []
    for terminal_selected in terminals_selected:
        if terminals_metadata[terminal_selected]['is_charging_station'] == 'not_depo_charger':
            ids.append(terminals_metadata[terminal_selected]['id'])
    return ids

def get_all_not_depo_charging_terminals_ids(terminals_metadata):
    no_depo_chargers = {}
    for k in terminals_metadata.keys():
        if terminals_metadata[k]['is_charging_station'] == 'not_depo_charger':
            no_depo_chargers[k] = terminals_metadata[k]
    return no_depo_chargers

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
            return login_data[0]['id'], login_data[0]['email'], login_data[0]['company'], login_data[0]['language']
        else:
            return False, False, False, False
    else:
        return False, False, False, False


def check_company_setup(conn):
    terminal_data = conn.execute("SELECT * FROM terminal WHERE id_user='%s'" % session['id_user']).fetchall()
    distance_data = conn.execute("SELECT * FROM distance WHERE id_user='%s'" % session['id_user']).fetchall()
    bus_model_data = conn.execute("SELECT * FROM bus_model WHERE id_user='%s'" % session['id_user']).fetchall()

    if len(terminal_data) > 0 and len(distance_data) > 0 and len(bus_model_data):
        return True
    else:
        return False


def get_sims_data(conn):
    return conn.execute("SELECT *, datetime(created_at, 'unixepoch', 'localtime') AS created_at_dt "
                        "FROM sim WHERE id_user=%i" % session['id_user']).fetchall()


def get_companies_names(conn):
    names = []
    for data in conn.execute("select DISTINCT(company) from user").fetchall():
        names.append(data[0])
    return names


def get_single_sim_data(conn, id_sim):
    cur = conn.cursor()
    cur.execute("SELECT *, datetime(created_at, 'unixepoch', 'localtime') AS created_at_dt "
                "FROM sim WHERE id=%i" % int(id_sim))

    for row in cur.fetchall():
        return list(row)


def get_user_data(conn):
    cur = conn.cursor()
    cur.execute("SELECT * FROM user WHERE id=%i" % int(session['id_user']))
    row = cur.fetchone()
    column_names = [d[0] for d in cur.description]
    return dict(zip(column_names, row))


def update_user_data(conn, new_data):
    cur = conn.cursor()
    if new_data['company'] == 'new':
        company = new_data['new_company']
    else:
        company = new_data['company']
    session['company_user'] = company

    cur.execute("UPDATE user SET username=?, email=?, company=?, language=? WHERE id=?",
                (new_data['username'], new_data['email'], company, new_data['language'], session['id_user']))
    session['username'] = new_data['username']
    session['email'] = new_data['email']
    session['language'] = new_data['language']
    conn.commit()

def change_user_password(conn, new_data):
    if new_data['password'] == new_data['confirm']:
        cur = conn.cursor()
        cur.execute("UPDATE user SET password=? WHERE id=?",
                    (encrypt(new_data['password']), session['id_user']))
        conn.commit()
        return True
    else:
        return False

def get_buses_models_data(conn):
    cur = conn.cursor()
    cur.execute("SELECT id, code, name, features FROM bus_model WHERE id_user=%i ORDER BY name ASC" % session['id_user'])
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

def check_id_bus_model(buses_models, id_bus):
    for bm in buses_models:
        if id_bus == int(bm['id']):
            return True
    return False

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


def insert_user(conn, username, email, password, company, language):
    cur = conn.cursor()
    res = cur.execute("INSERT INTO user (username, email, password, company, language) VALUES (?, ?, ?, ?, ?)",
                      (username, email, encrypt(password), company, language))
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

def calculate_emissions(rates, kms):
    return {
        'CO2':  { 'value': rates['CO2']  * 1e-6 * float(kms), 'unit': 'ton'},
        'NOx':  { 'value': rates['NOx']  * 1e-6 * float(kms), 'unit': 'ton'},
        'PM10': { 'value': rates['PM10'] * 1e-3 * float(kms), 'unit': 'kg'}
    }

def calc_a(i, t):
    q = 1 + i
    return (np.power(q, t) * i) / (np.power(q, t) - 1)

def get_total_number_batteries_packs(capex_number_batt_packs):
    tot_capacity = 0
    for k in capex_number_batt_packs.keys():
        tot_capacity += capex_number_batt_packs[k]
    return tot_capacity

def calculate_economical_parameters(main_cfg, capex_features, opex_features, input_pars, num_pantographs,
                                    input_bus_model_data):
    # 1) CAPEX SECTION

    # Calculate the annualization parameter
    interest_rate = float(capex_features['capex_interest_rate']) / 1e2
    a_bus = calc_a(interest_rate, float(capex_features['capex_bus_lifetime']))
    a_batt = calc_a(interest_rate, float(capex_features['capex_battery_lifetime']))
    a_char = calc_a(interest_rate, float(capex_features['capex_charger_lifetime']))
    a_panto = calc_a(interest_rate, float(capex_features['capex_panto_lifetime']))
    a_add_fee = calc_a(interest_rate, float(capex_features['capex_additional_fee_lifetime']))

    # CAPEX cost of the buses without the batteries (cost in CHF/bus)
    capex_bus_cost_ann = a_bus * float(capex_features['capex_bus_cost']) * float(capex_features['capex_number_buses'])

    # CAPEX cost of the batteries installed in the buses (cost in CHF/battery pack)
    num_battery_packs = get_total_number_batteries_packs(capex_features['capex_number_batt_packs'])
    capex_batt_cost_ann = a_batt * float(capex_features['capex_battery_cost']) * float(num_battery_packs)

    # CAPEX cost of the deposit charger (cost in CHF/bus because chargers=buses)
    capex_char_cost_ann = a_char * float(capex_features['capex_charger_cost']) * float(capex_features['capex_number_chargers'])

    # CAPEX cost of other pantograph/plugin chargers (not deposit) (cost in CHF/kW)
    # Currently it is not possible to differentiate between pantographs and plugin charging stations
    capex_panto_cost_ann = a_panto * float(capex_features['capex_panto_cost']) * float(input_pars['p_max']) * num_pantographs

    # CAPEX cost of fees (basically the connection one) (cost in CHF/kW)
    # Maximum power of the deposit (power calculated by the simulation)
    max_power_deposit = float(capex_features['capex_maximum_power_at_deposit'])
    # Maximum power of the (not deposit) chargers (power inputted by the user before the simulation (default 450 kW)
    max_power_panto = float(input_pars['p_max']) * num_pantographs
    capex_add_fee_ann = a_add_fee * float(capex_features['capex_additional_fee']) * (max_power_deposit + max_power_panto)

    # Calculate the total investment multiplying the annualized costs (in CHF/y) for the lifetime of the different components
    capex_investment_cost = capex_bus_cost_ann * float(capex_features['capex_bus_lifetime'])
    capex_investment_cost += capex_batt_cost_ann * float(capex_features['capex_battery_lifetime'])
    capex_investment_cost += capex_char_cost_ann * float(capex_features['capex_charger_lifetime'])
    capex_investment_cost += capex_panto_cost_ann * float(capex_features['capex_panto_lifetime'])

    # Get the total annualized capex
    capex_cost_ann = capex_bus_cost_ann + capex_batt_cost_ann + capex_char_cost_ann + capex_panto_cost_ann + capex_add_fee_ann

    # Get the total cost for the diesel
    diesel_cost_single_bus = (main_cfg['defaultCosts']['diesel']['investment']['slope'] * float(input_bus_model_data['length']) +
                              main_cfg['defaultCosts']['diesel']['investment']['intercept'])
    capex_cost_ann_diesel = a_bus * diesel_cost_single_bus * float(capex_features['capex_number_buses'])
    capex_investment_cost_diesel = capex_cost_ann_diesel * float(capex_features['capex_bus_lifetime'])

    # 2) OPEX SECTION

    # Calculate the OPEX costs for the electrical buses
    opex_cost_consumption = float(opex_features['opex_buses_maintainance']) * float(opex_features['opex_annual_usage']) * float(capex_features['capex_number_buses'])
    opex_cost_maintenance = float(opex_features['opex_bus_efficiency_sim']) * float(opex_features['opex_energy_tariff']) * float(opex_features['opex_annual_usage']) * float(capex_features['capex_number_buses'])
    opex_cost = opex_cost_maintenance + opex_cost_consumption

    # Calculate the OPEX costs for the diesel buses
    # CHF/y = (0.02 x [length bus] + 0.1918) x [total km per year] x 1.6 [CHF/liter, user editable]
    m_cons = main_cfg['defaultCosts']['diesel']['opex']['consumption']['m']
    q_cons = main_cfg['defaultCosts']['diesel']['opex']['consumption']['q']
    opex_cost_consumption_diesel = (m_cons * float(input_bus_model_data['length']) + q_cons) * float(input_pars['opex_annual_usage']) * float(input_pars['opex_diesel_cost_per_liter'])
    opex_cost_consumption_diesel *= float(capex_features['capex_number_buses'])
    # CHF/y = (0.02 x [length bus] + 0.14) x [total km per year]
    m_mant = main_cfg['defaultCosts']['diesel']['opex']['maintenance']['m']
    q_mant = main_cfg['defaultCosts']['diesel']['opex']['maintenance']['q']
    opex_cost_maintenance_diesel = (m_mant * float(input_bus_model_data['length']) + q_mant) * float(input_pars['opex_annual_usage'])
    opex_cost_maintenance_diesel *= float(capex_features['capex_number_buses'])
    opex_cost_diesel = opex_cost_consumption_diesel + opex_cost_maintenance_diesel

    capex_opex_years = []
    capex_opex_cost_at_year = []
    capex_opex_cost_at_year_diesel = []

    capex_investment_cost_batt = capex_batt_cost_ann * float(capex_features['capex_battery_lifetime'])
    capex_investment_cost_char = capex_char_cost_ann * float(capex_features['capex_charger_lifetime'])
    capex_investment_cost_panto = capex_panto_cost_ann * float(capex_features['capex_panto_lifetime'])
    for i in range(0, main_cfg['defaultCosts']['investmentPeriod']+1):
        offset = 0
        if main_cfg['breakPointAdvancedGraph'] is True:
            if i > float(capex_features['capex_battery_lifetime']):
                offset += capex_investment_cost_batt
            if i > float(capex_features['capex_charger_lifetime']):
                offset += capex_investment_cost_char
            if i > float(capex_features['capex_panto_lifetime']):
                offset += capex_investment_cost_panto

        ye = opex_cost * i + capex_investment_cost + offset
        yd = opex_cost_diesel * i + capex_investment_cost_diesel
        capex_opex_years.append(i)
        capex_opex_cost_at_year.append(round(ye/1e6, 2))
        capex_opex_cost_at_year_diesel.append(round(yd/1e6, 2))


    capex_opex_costs = {
        "capex_bus_cost":  capex_bus_cost_ann/1e3,
        "capex_batt_cost": capex_batt_cost_ann/1e3,
        "capex_depo_charger_cost": capex_char_cost_ann/1e3,
        "capex_not_depo_charger_cost": capex_panto_cost_ann/1e3,
        "capex_add_fee": capex_add_fee_ann/1e3,
        "capex_investment_cost": capex_investment_cost/1e3,
        "capex_investment_cost_diesel": capex_investment_cost_diesel/1e3,
        "capex_cost_ann": capex_cost_ann/1e3,
        "capex_cost_ann_diesel": capex_cost_ann_diesel/1e3,
        "opex_cost": opex_cost/1e3,
        "opex_cost_consumption": opex_cost_consumption/1e3,
        "opex_cost_maintenance": opex_cost_maintenance/1e3,
        "opex_cost_diesel": opex_cost_diesel/1e3,
        "opex_cost_consumption_diesel": opex_cost_consumption_diesel/1e3,
        "opex_cost_maintenance_diesel": opex_cost_maintenance_diesel/1e3,
        "capex_opex_years": capex_opex_years,
        "capex_opex_cost_at_year": capex_opex_cost_at_year,
        "capex_opex_cost_at_year_diesel": capex_opex_cost_at_year_diesel
    }
    return capex_opex_costs


def create_costs_bargraph(capex_opex_costs):
    categories = [gettext('Electrical'), gettext('Diesel')]
    vals_bus = [round(capex_opex_costs['capex_bus_cost']), round(capex_opex_costs['capex_cost_ann_diesel'])]
    vals_batt = [round(capex_opex_costs['capex_batt_cost']), 0]
    vals_depo_char = [round(capex_opex_costs['capex_depo_charger_cost']), 0]
    vals_no_depo_char = [round(capex_opex_costs['capex_not_depo_charger_cost']), 0]
    vals_main = [round(capex_opex_costs['opex_cost_maintenance']),
                 round(capex_opex_costs['opex_cost_maintenance_diesel'])]
    vals_cons = [round(capex_opex_costs['opex_cost_consumption']),
                 round(capex_opex_costs['opex_cost_consumption_diesel'])]

    # Create traces for each set of values
    trace_bus = go.Bar(x=categories, y=vals_bus, name='CAPEX - %s' % gettext('Buses'),
                       marker=dict(color=['rgba(0, 0, 139, 0.7)', 'rgba(0, 0, 139, 0.7)',
                                          'rgba(0, 0, 139, 0.7)', 'rgba(0, 0, 139, 0.7)']))

    trace_batt = go.Bar(x=categories, y=vals_batt, name='CAPEX - %s' % gettext('Batteries'),
                        marker=dict(color=['rgba(0, 0, 255, 0.7)', 'rgba(0, 0, 255, 0.7)',
                                           'rgba(0, 0, 25, 0.7)', 'rgba(0, 0, 255, 0.7)']))

    trace_depo_char = go.Bar(x=categories, y=vals_depo_char, name='CAPEX - %s' % gettext('Deposit charger'),
                             marker=dict(color=['rgba(30, 144, 255, 0.7)', 'rgba(30, 144, 255, 0.7)',
                                                'rgba(30, 144, 255, 0.7)', 'rgba(30, 144, 255, 0.7)']))

    trace_no_depo_char = go.Bar(x=categories, y=vals_no_depo_char,
                                name='CAPEX - %s' % gettext('Other chargers (e.g. pantographs)'),
                                marker=dict(color=['rgba(135, 206, 230, 0.7)', 'rgba(135, 206, 230, 0.7)',
                                                   'rgba(135, 206, 230, 0.7)', 'rgba(135, 206, 230, 0.7)']))

    trace_main = go.Bar(x=categories, y=vals_main, name='OPEX - %s' % gettext('Maintenance'),
                        marker=dict(color=['rgba(0, 100, 0, 0.7)', 'rgba(0, 100, 0, 0.7)',
                                           'rgba(0, 100, 0, 0.7)', 'rgba(0, 100, 0, 0.7)']))

    trace_cons = go.Bar(x=categories, y=vals_cons, name='OPEX - %s' % gettext('Consumption'),
                        marker=dict(color=['rgba(0, 150, 0, 0.7)', 'rgba(0, 150, 0, 0.7)',
                                           'rgba(0, 150, 0, 0.7)', 'rgba(0, 150, 0, 0.7)']))

    layout = go.Layout(
        title=gettext('Annual expenditure and operative costs'),
        xaxis=dict(title=''),
        yaxis=dict(title='kCHF/%s' % gettext('year')),
        barmode='stack',
        plot_bgcolor='rgba(255, 255, 255, 0.8)',
        paper_bgcolor='rgba(255, 255, 255, 0.8)'
    )

    fig = go.Figure(data=[trace_bus, trace_batt, trace_depo_char, trace_no_depo_char, trace_main, trace_cons],
                    layout=layout)
    plot_div = fig.to_html(full_html=False)
    return plot_div

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


def calculate_default_costs(input_data):
    dict_main_input = json.loads(input_data['main_input_data'].replace('\'', '\"'))
    step = 1e3
    default_costs =  {
        "charger": step*(round((main_cfg['defaultCosts']['charger']['slope'] * float(dict_main_input['deposit_nominal_power']) + main_cfg['defaultCosts']['charger']['intercept'])/step, 0)),
        "pantograph": main_cfg['defaultCosts']['pantograph']
    }
    return default_costs


def launch_sim_instance(conn, main_cfg, private_cfg, pars):
    cur = conn.cursor()

    ts = pars['data_file'].replace('.csv', '').split(os.sep)[-1].split('_')[-1]

    bus_model_data = get_single_bus_model_data(conn, int(pars['bus_model_id']))
    terminals_selected = get_terminals(pars)
    terminals_metadata = get_terminals_metadata_dict(conn)
    distances_matrix = get_distances_matrix_dict(conn)

    distances_matrix = {str(key): value for key, value in distances_matrix.items()}
    form_data = {
        "id": session['id_user'],
        "company": session['company_user'],
        "main_cfg": main_cfg,
        "pars": pars,
        "bus_model_data": bus_model_data,
        "terminals_selected": terminals_selected,
        "terminals_metadata": terminals_metadata,
        "distances_matrix": distances_matrix
    }

    # Create the lines string
    str_lines = ''
    for par in pars:
        if 'line_' in par:
            str_lines = '%s%s,' % (str_lines, par.replace('line_', ''))
    str_lines = str_lines[:-1]

    input_pars_file = pars['data_file'].replace('input-csv', 'json-input-pars').replace('csv', 'json')
    with open(input_pars_file, "w") as fw:
        json.dump(form_data, fw)

    files_to_send = {
        "input_pars_file": open(input_pars_file, 'rb'),
        "input_data_file": open(pars['data_file'], 'rb')
    }

    # Send the POST request
    response = requests.post(private_cfg['simulatorUrl'], files=files_to_send)
    data_response = json.loads(response.text)

    if response.status_code == http.HTTPStatus.OK and data_response['error'] is False:
        cur.execute("INSERT INTO sim (id_user, created_at, company, line, day_type, battery_size, max_charging_power, "
                    "elevation_deposit, elevation_starting_station, elevation_arrival_station, name, "
                    "input_terminals_selected, input_terminals_metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (int(session['id_user']), int(ts), session['company_user'], str_lines, pars['day_type'],
                     float(bus_model_data['batt_pack_capacity']), float(pars['p_max']), 0, 0, 0, pars['sim_name'],
                     json.dumps(terminals_selected), json.dumps(terminals_metadata)))
        conn.commit()
    return True


def create_new_bus_model(conn, pars):
    bus_name = pars['bus_name']
    del pars['bus_name']

    cur = conn.cursor()

    with open('static/bus-types/%s.json' % pars['bus_type'], 'r') as f:
        default_pars = json.load(f)

    # Update the default with the input values
    default_pars.update(pars)
    default_pars['capex_battery_cost'] = (main_cfg['defaultCosts']['battery']['slope'] * float(pars['batt_pack_capacity']) +
                                          main_cfg['defaultCosts']['battery']['intercept'])
    default_pars['batt_pack_weight'] = main_cfg['weights']['battery'] * float(pars['batt_pack_capacity'])


    cur.execute("INSERT INTO bus_model (id_user, code, name, features) "
                "VALUES (?, ?, ?, ?)",
                (session['id_user'], pars['bus_type'], bus_name, json.dumps(default_pars)))
    conn.commit()
    conn.close()


def update_bus_model(conn, pars):
    bus_id = pars['id']
    bus_name = pars['name']
    del pars['id']
    del pars['name']
    cur = conn.cursor()

    pars['capex_battery_cost'] = (main_cfg['defaultCosts']['battery']['slope'] * float(pars['batt_pack_capacity']) +
                                  main_cfg['defaultCosts']['battery']['intercept'])
    pars['batt_pack_weight'] = main_cfg['weights']['battery'] * float(pars['batt_pack_capacity'])

    cur.execute("UPDATE bus_model SET name=?, features=? WHERE id=?",
                (bus_name, json.dumps(pars), bus_id))
    conn.commit()

def assemble_sequences(df_start, df_rest, depot_end):
    seq = defaultdict(list)
    n1_values = df_start['n1'].values
    n2_values = df_start['n2'].values
    df_rest = df_rest.set_index('n1')

    depot_end = set(depot_end)

    for index, (n1, n2) in enumerate(zip(n1_values, n2_values)):
        trip = (n1, n2)
        seq[index].append(trip)

        while True:
            n3 = df_rest.loc[n2, 'n2']
            seq[index].append((n2, n3))
            n2 = n3

            if n3 in depot_end:
                break
    seq = {k: [t[1] for t in v] for k, v in seq.items()}
    return seq


def generate_graph(nodes_list, edges_list):
    graph = nx.Graph()

    for n0 in nodes_list:
        graph.add_node(n0)
    for k, v in edges_list.items():
        for e in v:
            graph.add_edge(k, e)
    return graph

def clean_terminals(conn):
    conn.execute("DELETE FROM terminal WHERE id_user = ?", (session['id_user'],))
    conn.commit()


def clean_distances(conn):
    conn.execute("DELETE FROM distance WHERE id_user = ?", (session['id_user'],))
    conn.commit()


def update_terminals(conn, terminals_file_path):
    df = pd.read_csv(terminals_file_path)
    df = df.assign(id=df.index)
    df = df.assign(id_user=np.array([session['id_user'] for _ in range(len(df.index))]))
    df = df.assign(company=np.array([session['company_user'] for _ in range(len(df.index))]))
    df = df.rename(columns={'terminal_station': 'name'})
    df = df.reindex(columns=['id_user', 'company', 'name', 'elevation_m', 'is_charging_station'])
    df.to_sql('terminal', conn, if_exists='append', index=False)

    terms = get_terminals_metadata(conn)
    terms_dict = {}
    for term in terms:
        terms_dict[term[2]] = term[0]
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
    df = df.assign(id_user=np.array([session['id_user'] for _ in range(len(df.index))]))
    df = df.drop(['starting_station', 'arrival_station'], axis=1)
    df = df.reindex(columns=['id_user', 'id_starting_station', 'id_arrival_station', 'distance_km', 'avg_travel_time_min', 'company'])
    df.to_sql('distance', conn, if_exists='append', index=False)


def get_terminals_metadata(conn):
    return conn.execute("SELECT * FROM terminal WHERE id_user='%s' ORDER BY name" % session['id_user']).fetchall()

def handle_terminals_metadata(raw_data):
    res_data = []
    for rd in raw_data:
        res_data.append({'name': rd[2], 'company': rd[3], 'elevation': rd[4], 'is_charging_station': rd[5]})
    return res_data

def get_distances_matrix(conn):
    return conn.execute("SELECT * FROM distance WHERE id_user='%s'" % session['id_user']).fetchall()


def get_terminals_metadata_dict(conn):
    terms = get_terminals_metadata(conn)
    terms_dict = {}
    for term in terms:
        terms_dict[term[2]] = {'id': term[0], 'elevation': term[4], 'is_charging_station': term[5]}
    return terms_dict


def get_distances_matrix_dict(conn):
    distances = get_distances_matrix(conn)
    distances_dict = {}
    for d in distances:
        distances_dict[d[1], d[2]] = {'distance_km': d[3], 'avg_travel_time_min': d[4]}
    return distances_dict


# Route for handling the login page logic
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if 'lang' in request.args.keys():
        session['language'] = request.args['lang']
    else:
        session['language'] = 'it'

    if request.method == 'POST':
        conn = get_db_connection()
        id_user, email_user, company_user, language = check_login_data(conn, request.form['username'],
                                                                       request.form['password'])
        conn.close()
        if id_user is False:
            error = 'Invalid Credentials. Please try again.'
        else:
            session['id_user'] = id_user
            session['email_user'] = email_user
            session['company_user'] = company_user
            session['language'] = language
            session['username'] = request.form['username']
            session['password'] = request.form['password']
            return redirect(url_for('index'))
    return render_template('login.html', error=error)


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.pop('username', None)
    session.pop('password', None)
    session.pop('language', None)
    session.pop('id_user', None)
    session.pop('email', None)
    session.pop('company_user', None)
    session.pop('email_user', None)
    return redirect(url_for('index'))

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[
        DataRequired(),
        Length(min=12, message=gettext('Password must be at least 12 characters long')),
        Regexp(r'(?=.*\d)', message=gettext('Password must contain a number')),
        Regexp(r'(?=.*[A-Z])', message=gettext('Password must contain an uppercase letter')),
        Regexp(r'(?=.*[a-z])', message=gettext('Password must contain a lowercase letter'))
    ])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(),
        EqualTo('password', message=gettext('Passwords must match'))
    ])
    recaptcha = RecaptchaField()
    submit = SubmitField('Register')

# Route for handling the login page logic
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    session['language'] = 'en'
    if 'lang' in request.args.keys():
        session['language'] = request.args['lang']

    if request.method == 'POST':
        session['language'] = request.form['language']

    conn = get_db_connection()
    companies = get_companies_names(conn)

    form = RegistrationForm()
    if form.validate_on_submit():

        if request.method == 'POST':
            if request.form['company'] == 'new':
                insert_user(conn, request.form['username'], request.form['email'], request.form['password'],
                            request.form['new_company'], request.form['language'])
            else:
                insert_user(conn, request.form['username'], request.form['email'], request.form['password'],
                            request.form['company'], request.form['language'])
            conn.close()
            return redirect(url_for('login'))
        else:
            conn.close()
            return render_template('signup.html', language=session['language'], error=error,
                                   companies=companies, form=form)
    else:
        conn.close()
        return render_template('signup.html', language=session['language'], companies=companies,
                               form=form)


@app.route('/', methods=('GET', 'POST'))
def index():
    if is_logged():
        conn = get_db_connection()
        if 'del' in request.args.keys() and 'id' in request.args.keys():
            id_sim_to_delete = int(request.args.to_dict()['id'])
            sim_metadata = get_single_sim_data(conn, id_sim_to_delete)
            try:
                delete_sim(conn, sim_metadata)
            except Exception as e:
                print('EXCEPTION: %s' % str(e))
                print('Error deleting simulation %i: it does now exist' % id_sim_to_delete)

        flag_company_setup = check_company_setup(conn)

        # Get simulation data
        sims = get_sims_data(conn)
        conn.close()

        # Create a list of dictionary
        sims_list = []
        for sim in sims:
            data_sim = {
                'id': sim['id'],
                'name': sim['name'],
                'created_at_dt': sim['created_at_dt'][:-3],
                'company': sim['company'],
                'line': sim['line'],
                'day_type': sim['day_type'],
                'battery_size': sim['battery_size'],
                'capex_pars': sim['capex_pars'],
                'max_charging_powers': sim['max_charging_powers']
            }

            if sim['input_pars'] is not None and sim['input_bus_model_data'] is not None:
                data_sim['input_pars'] = json.loads(sim['input_pars'])
                data_sim['input_bus_model_data'] = json.loads(sim['input_bus_model_data'])
            else:
                data_sim['input_pars'] = None
                data_sim['input_bus_model_data'] = None
            sims_list.append(data_sim)

        return render_template('index.html', sims_list=sims_list, flag_company_setup=flag_company_setup)
    else:
        return render_template('index.html', sims_list=[], flag_company_setup=False, language='it')


@app.route('/detail', methods=('GET', 'POST'))
def detail():
    if is_logged():
        conn = get_db_connection()
        flag_company_setup = check_company_setup(conn)
        id_sim = int(request.args.to_dict()['id'])
        sim_metadata = get_single_sim_data(conn, id_sim)
        conn.close()
        try:
            input_pars = json.loads(sim_metadata[14])

            df_bsize_filename = 'static/output-bsize/%s_%i.csv' % (session['id_user'], sim_metadata[2])
            df_bsize = pd.read_csv(df_bsize_filename)

            # df_data_filename = 'static/output-df/%s_%i.csv' % (session['id_user'], sim_metadata[2])
            # df_data = pd.read_csv(df_data_filename)

            data = dict(request.args)
            # data['min_num_drivers'] = len(set(df_data["Driver#"]))
            data['df_bsize'] = df_bsize

            bus_data = {
                'number': len(df_bsize),
                'battery_packs_capacity': df_bsize.iloc[0]['Battery size (kWh)'],
                'battery_packs_number': int(df_bsize.iloc[0]['Battery packs'])
            }

            # Calculate CAPEX and OPEX costs
            capex_features = json.loads(sim_metadata[11])
            opex_features = json.loads(sim_metadata[12])
            input_bus_model_data = json.loads(sim_metadata[15])
            sim_name = sim_metadata[16]
            terminals_selected = json.loads(sim_metadata[17])
            terminals_metadata = json.loads(sim_metadata[18])
            panto_ids = get_not_depo_charging_terminals_ids(terminals_metadata, terminals_selected)

            if 'opex_bus_efficiency_sim' not in opex_features.keys():
                opex_features['opex_bus_efficiency_sim'] = float(opex_features['opex_buses_efficiency'])

            # Calculate the cost of a single not depo charging station and update costs
            if len(panto_ids) > 0:
                capex_features['capex_single_not_depo_charger_investment'] = float(capex_features['capex_panto_cost']) * float(input_pars['p_max'])/1e3
            else:
                capex_features['capex_single_not_depo_charger_investment'] = 0
            input_pars['capex_bus_cost'] = float(input_pars['capex_bus_cost']) / 1e3
            input_pars['capex_charger_cost'] = float(input_pars['capex_charger_cost']) / 1e3

            capex_opex_costs = calculate_economical_parameters(main_cfg=main_cfg, capex_features=capex_features,
                                                               opex_features=opex_features, input_pars=input_pars,
                                                               num_pantographs=len(panto_ids),
                                                               input_bus_model_data=input_bus_model_data)

            # Calculate emissions
            ems = calculate_emissions(main_cfg['emissionsRates'], opex_features['opex_annual_usage'])

            # Create bargraph object for the costs
            plot_div = create_costs_bargraph(capex_opex_costs)

            return render_template('detail.html', sim_metadata=sim_metadata, data=data, bus_data=bus_data,
                                   capex_features=capex_features, opex_features=opex_features, sim_name=sim_name,
                                   capex_opex_costs=capex_opex_costs, flag_company_setup=flag_company_setup, emissions=ems,
                                   input_pars=input_pars, input_bus_model_data=input_bus_model_data, plot_div=plot_div)
        except Exception as e:
            print('EXCEPTION: %s' % str(e))
            print('Error showing results of simulation %i: it does now exist' % id_sim)
            return redirect(url_for('index'))
    else:
        return redirect(url_for('login'))


@app.route('/new_sim_step1/', methods=('GET', 'POST'))
def new_sim_step1():
    if is_logged():
        conn = get_db_connection()
        flag_company_setup = check_company_setup(conn)

        if request.method == 'POST':
            target = 'static/input-csv'

            ts = (datetime.datetime.now(tz=pytz.UTC).timestamp())
            if len(request.files['data_file'].filename) > 0:
                try:
                    file_name = '%i_%i.csv' % (session['id_user'], ts)
                    data_file = '/'.join([target, file_name])
                    request.files['data_file'].save(data_file)

                    main_input_data = request.form.to_dict()
                    lines, days_types = get_lines_daytypes_from_data_file(data_file)
                    conn.close()
                    return redirect(url_for('new_sim_step2', data_file=data_file, lines=lines, days_types=days_types,
                                            id_bus_model=main_input_data['id_bus_model'],
                                            main_input_data=main_input_data,
                                            flag_company_setup=flag_company_setup))
                except Exception as e:
                    buses_models = get_buses_models_data(conn)
                    tm = get_terminals_metadata_dict(conn)
                    not_depo_ch_terms = get_all_not_depo_charging_terminals_ids(tm)
                    conn.close()
                    return render_template('new_sim_step1.html', error='Input file with wrong format, please check the example',
                                           buses_models=buses_models, flag_company_setup=flag_company_setup,
                                           main_cfg=main_cfg, num_not_depo_chargers=len(not_depo_ch_terms.keys()))
            else:
                buses_models = get_buses_models_data(conn)
                tm = get_terminals_metadata_dict(conn)
                not_depo_ch_terms = get_all_not_depo_charging_terminals_ids(tm)
                conn.close()
                return render_template('new_sim_step1.html', error='No input file uploaded, please check the example',
                                       buses_models=buses_models, flag_company_setup=flag_company_setup,
                                       main_cfg=main_cfg, num_not_depo_chargers=len(not_depo_ch_terms.keys()))
        else:
            buses_models = get_buses_models_data(conn)
            tm = get_terminals_metadata_dict(conn)
            not_depo_ch_terms = get_all_not_depo_charging_terminals_ids(tm)
            conn.close()
            return render_template('new_sim_step1.html', buses_models=buses_models,
                                   flag_company_setup=flag_company_setup, main_cfg=main_cfg,
                                   num_not_depo_chargers=len(not_depo_ch_terms.keys()))
    else:
        return redirect(url_for('login'))


@app.route('/new_sim_step2/', methods=('GET', 'POST'))
def new_sim_step2():
    if is_logged():
        conn = get_db_connection()
        flag_company_setup = check_company_setup(conn)

        if request.method == 'POST':
            try:
                # Run the simulation and save the output in the DB
                sim_pars = request.form.to_dict()
                launch_sim_instance(conn=conn, main_cfg=main_cfg, private_cfg=private_cfg, pars=sim_pars)
                conn.close()
                return redirect(url_for('index'))
            except Exception as e:
                print('EXCEPTION: %s' % str(e))
                conn = get_db_connection()
                req_dict = request.args.to_dict()
                defaults_costs = calculate_default_costs(req_dict)
                step1_data = json.loads(req_dict['main_input_data'].replace('\'', '\"'))
                lines, days_types = get_lines_daytypes_from_data_file(req_dict['data_file'])
                bus_model_data = get_single_bus_model_data(conn, int(req_dict['id_bus_model']))
                conn.close()
                return render_template('new_sim_step2.html',
                                       error='Data file has a wrong format! The simulation cannot be run',
                                       data_file=req_dict['data_file'], lines=lines, days_types=days_types,
                                       bus_model_data=bus_model_data, step1_data=step1_data,
                                       flag_company_setup=flag_company_setup, main_cfg=main_cfg,
                                       defaults_costs=defaults_costs)
        else:
            req_dict = request.args.to_dict()
            step1_data = json.loads(req_dict['main_input_data'].replace('\'', '\"'))
            lines, days_types = get_lines_daytypes_from_data_file(req_dict['data_file'])
            bus_model_data = get_single_bus_model_data(conn, int(req_dict['id_bus_model']))
            defaults_costs = calculate_default_costs(req_dict)
            conn.close()
            return render_template('new_sim_step2.html', data_file=req_dict['data_file'], lines=lines,
                                   days_types=days_types, bus_model_data=bus_model_data, step1_data=step1_data,
                                   flag_company_setup=flag_company_setup, main_cfg=main_cfg,
                                   defaults_costs=defaults_costs)
    else:
        return redirect(url_for('login'))


@app.route('/new_bus_model', methods=('GET', 'POST'))
def new_bus_model():
    if is_logged():
        conn = get_db_connection()
        flag_company_setup = check_company_setup(conn)
        if request.method == 'POST':
            create_new_bus_model(conn, request.form.to_dict())
            return redirect(url_for('company_manager'))
        else:
            available_buses_models = get_available_buses_models()
            args = request.args.to_dict()
            # Get default bus parameters
            try:
                with open('static/bus-types/%s.json' % args['l'], 'r') as f:
                    bus_default_pars = json.load(f)
                bus_default_pars['bus_type'] = args['l']
            except Exception as e:
                print('EXCEPTION: %s' % str(e))
                with open('static/bus-types/12m.json', 'r') as f:
                    bus_default_pars = json.load(f)
                bus_default_pars['bus_type'] = '12m'

            bus_default_pars['default_cost'] = main_cfg['defaultCosts']['bus'][args['l']]
            return render_template('new_bus_model.html',
                                   available_buses_models=available_buses_models, main_cfg=main_cfg,
                                   flag_company_setup=flag_company_setup, bus_default_pars=bus_default_pars)
    else:
        return redirect(url_for('login'))


@app.route('/edit_bus_model', methods=('GET', 'POST'))
def edit_bus_model():
    if is_logged():
        conn = get_db_connection()
        flag_company_setup = check_company_setup(conn)
        try:
            id_bus_model = int(request.args.to_dict()['id_bus_model'])
            buses_models = get_buses_models_data(conn)

            # Check if the bus model is owned by the user
            if check_id_bus_model(buses_models, id_bus_model) is True:
                if request.method == 'POST':
                    # Update data on db
                    new_pars = request.form.to_dict()
                    bus_model_data = get_single_bus_model_data(conn, id_bus_model)
                    for k in bus_model_data.keys():
                        if k in new_pars.keys():
                            bus_model_data[k] = new_pars[k]
                    update_bus_model(conn, bus_model_data)

                    # Get data of bus model and pass them to the page
                    try:
                        bus_model_data = get_single_bus_model_data(conn, id_bus_model)
                        conn.close()
                    except Exception as e:
                        print('EXCEPTION: %s' % str(e))
                        print('Error editing features of bus %i: it does now exist' % id_bus_model)
                        conn.close()
                        return redirect(url_for('company_manager'))

                    return render_template('edit_bus_model.html', bus_model_data=bus_model_data,
                                           flag_company_setup=flag_company_setup)
                else:
                    # Get data of bus model and pass them to the page
                    try:
                        bus_model_data = get_single_bus_model_data(conn, id_bus_model)
                        conn.close()
                    except Exception as e:
                        print('EXCEPTION: %s' % str(e))
                        print('Error editing features of bus %i: it does now exist' % id_bus_model)
                        conn.close()
                        return redirect(url_for('company_manager'))
            else:
                return redirect(url_for('company_manager'))
            return render_template('edit_bus_model.html', bus_model_data=bus_model_data,
                                   flag_company_setup=flag_company_setup)
        except Exception as e:
            print('EXCEPTION: %s' % str(e))
            print('Error: identifier of bus is not provided')
            conn.close()
            return redirect(url_for('company_manager'))
    else:
        return redirect(url_for('login'))


@app.route('/edit_user', methods=('GET', 'POST'))
def edit_user():
    if is_logged():
        err = None
        conn = get_db_connection()
        flag_company_setup = check_company_setup(conn)

        companies = get_companies_names(conn)
        if request.method == 'POST':
            form_data = request.form.to_dict()
            if form_data['type'] == 'change_settings':
                # Update main settings
                update_user_data(conn, form_data)
                companies = get_companies_names(conn)
            elif form_data['type'] == 'change_pwd':
                # Update password
                res = change_user_password(conn, form_data)
                if res is False:
                    err = 'New password failed the retype checking'
            # Get data from DB
            user_data = get_user_data(conn)
            conn.close()
            return render_template('edit_user.html', user_data=user_data, error=err,
                                   languages=main_cfg['languages'], flag_company_setup=flag_company_setup,
                                   companies=companies)
        else:
            # Get data from DB
            user_data = get_user_data(conn)
            conn.close()
            return render_template('edit_user.html', user_data=user_data,
                                   languages=main_cfg['languages'], flag_company_setup=flag_company_setup,
                                   companies=companies)
    else:
        return redirect(url_for('login'))


@app.route('/company_manager', methods=('GET', 'POST'))
def company_manager():
    if is_logged():
        conn = get_db_connection()
        flag_company_setup = check_company_setup(conn)
        buses_models = get_buses_models_data(conn)

        err = None
        if request.method == 'POST':
            try:
                target = 'static/tmp'
                terminals_file_path = '/'.join([target, '%i_terminals.csv' % session['id_user']])
                request.files['terminals_file'].save(terminals_file_path)
                distances_file_path = '/'.join([target, '%i_distances.csv' % session['id_user']])
                request.files['distances_file'].save(distances_file_path)

                clean_distances(conn)
                clean_terminals(conn)

                # Update the DB
                terms_dict = update_terminals(conn, terminals_file_path)
                update_distances(conn, distances_file_path, terms_dict)

                # Delete the uploaded files
                os.unlink(terminals_file_path)
                os.unlink(distances_file_path)

                # Get the new data from the DB
                terminals_raw_data = get_terminals_metadata(conn)
                terminals_data = handle_terminals_metadata(terminals_raw_data)

                # Update the company setup
                flag_company_setup = check_company_setup(conn)
            except Exception as e:
                # Delete the uploaded files
                os.unlink(terminals_file_path)
                os.unlink(distances_file_path)

                # Get the new data from the DB
                terminals_raw_data = get_terminals_metadata(conn)
                terminals_data = handle_terminals_metadata(terminals_raw_data)

                print('Exception: %s' % str(e))
                err = 'The uploaded files have not the right folder, please check the examples files and upload them again'

        elif request.method == 'GET':
            if 'del' in request.args.keys() and 'id_bus_model' in request.args.keys():
                id_bus_model_to_delete = int(request.args['id_bus_model'])
                # Check if the bus model is owned by the user
                if check_id_bus_model(buses_models, id_bus_model_to_delete) is True:
                    delete_bus_model(conn, id_bus_model_to_delete)
                    buses_models = get_buses_models_data(conn)

                    # Update the company setup
                    flag_company_setup = check_company_setup(conn)

            terminals_raw_data = get_terminals_metadata(conn)
            terminals_data = handle_terminals_metadata(terminals_raw_data)

        return render_template('company_manager.html', buses_models=buses_models,
                               company=session['company_user'], terminals_data=terminals_data,
                               flag_company_setup=flag_company_setup, error=err)
    else:
        return redirect(url_for('login'))


if __name__ == "__main__":
    app.run(debug=True)
