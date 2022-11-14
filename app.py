import csv
import os
import sqlite3
import datetime
import json
import zipfile
import pytz

from cryptography.fernet import Fernet

from model import MILP
from gurobipy import *
from flask import Flask, render_template, request, url_for, redirect, session, flash

import config_sim_builder as csb

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


def run_sim(sim_file_path, main_cfg, pars):
    conn = get_db_connection()
    cur = conn.cursor()

    # Set properly the parameters
    (company, line) = pars['comp_line'].split('__')
    bsize = float(pars['bsize'])
    max_charging_power = int(pars['max_charge_power'])

    # Get user id and timestamp
    (id_user, ts) = sim_file_path.replace('.csv', '').split(os.sep)[-1].split('_')

    sim_cfg_filename = csb.configuration(sim_file_path, company, line, bsize, [],
                                         main_cfg['simSettings']['modelTimesteps'], max_charging_power,
                                         main_cfg['simSettings']['chargingAtDeposit'])

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
                 default_assignment=True,
                 partial_assignment=[],
                 non_overlap_charging=False)

    print('Starting optimizer...', datetime.datetime.now())
    model.optimize()

    if model.status == GRB.INFEASIBLE:
        return False

    # Define the variables to save in the database
    res_var = ['bp', 'bs', 'bi', 'Ct', 'u', 'x', 'yd', 'SOC']
    varInfo = [(v.varName, v.X) for v in model.getVars() if (v.X != 0) and any([v.varName.startswith(s) for s in res_var])]

    # Write the CSV output and archive it in a zip
    print(sim_file_path)
    res_file = sim_file_path.split(os.sep)[-1]
    with open(res_file, 'w', newline='') as f:
        wr = csv.writer(f, delimiter=";")
        wr.writerows(varInfo)

    with zipfile.ZipFile(sim_file_path.replace('input-csv', 'output').replace('.csv', '.zip'), 'w',
                         compression=zipfile.ZIP_DEFLATED) as zip:
        zip.write(res_file)
    os.unlink(res_file)

    print('')
    end = datetime.datetime.now()
    print("End time: ", end)
    print('Solution time (seconds): ', (end - start).total_seconds())

    cur.execute("INSERT INTO sim (id_user, created_at, company, line, day_type, battery_size, max_charging_power) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (int(id_user), int(ts), company, line, pars['day_type'], bsize, max_charging_power))
    conn.commit()
    conn.close()
    return True


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