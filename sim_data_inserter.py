import argparse
import json
import os
import shutil
import zipfile
import sqlite3
import logging
import sys

from tendo.singleton import SingleInstance

def get_user_email(conn, user_id):
    data = conn.execute("SELECT * FROM user WHERE id='%s'" % user_id).fetchall()
    if len(data) > 0:
        return data[0]['email']
    else:
        return False

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    try:
        singleton = SingleInstance()
    except:
        sys.exit()

    # --------------------------------------------------------------------------- #
    # Configuration file
    # --------------------------------------------------------------------------- #
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-i", help="input folder")
    arg_parser.add_argument("-l", help="log file (optional, if empty log redirected on stdout)")
    args = arg_parser.parse_args()

    # --------------------------------------------------------------------------- #
    # Set logging object
    # --------------------------------------------------------------------------- #
    if not args.l:
        log_file = None
    else:
        log_file = args.l

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO,
                        filename=log_file)

    # Open Db connection
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    input_folder = '%s/input' % args.i
    tmp_folder = '%s/tmp' % args.i
    for sim_results_file in os.listdir(input_folder):
        shutil.copy('%s/%s' % (input_folder, sim_results_file), '%s/%s' % (tmp_folder, sim_results_file))
        sim_id = sim_results_file.replace('.zip', '')
        target_folder = '%s/%s' % (tmp_folder, sim_id)

        logger.info('Simulation %s: Data management starting' % sim_id)

        # Extract the data in the tmp folder
        os.mkdir(target_folder)
        with zipfile.ZipFile('%s/%s' % (tmp_folder, sim_results_file), 'r') as zip_ref:
            zip_ref.extractall(target_folder)

        # Copy the files results
        logger.info('Simulation %s: Files copy' % sim_id)
        shutil.copy('%s/input-csv/%s.csv' % (target_folder, sim_id), 'static/input-csv')
        shutil.copy('%s/input-csv/%s.csv' % (target_folder, sim_id), 'static/json-input-pars')
        shutil.copy('%s/output/%s.zip' % (target_folder, sim_id), 'static/output')
        shutil.copy('%s/output-bsize/%s.csv' % (target_folder, sim_id), 'static/output-bsize')
        shutil.copy('%s/output-df/%s.csv' % (target_folder, sim_id), 'static/output-df')
        shutil.copy('%s/plot/%s.png' % (target_folder, sim_id), 'static/plot')
        shutil.copy('%s/plot/%s_charge_profile.png' % (target_folder, sim_id), 'static/plot')
        shutil.copy('%s/sim-config/%s.json' % (target_folder, sim_id), 'static/sim-config')

        # Insert the main results in the database
        logger.info('Simulation %s: DB update' % sim_id)
        main_results = json.loads(open('%s/main_results.json' % target_folder).read())

        id_user, ts = sim_id.split('_')
        update_query = ("UPDATE sim SET line = ?, capex_pars = ?, opex_pars = ?, max_charging_powers = ? "
                        "WHERE id_user = ? and  created_at = ?;")
        data_to_update = (main_results['strRoutes'],
                          json.dumps(main_results['capexPars']),
                          json.dumps(main_results['opexPars']),
                          json.dumps(main_results['chargingData']),
                          int(id_user),
                          int(ts))
        cursor.execute(update_query, data_to_update)
        conn.commit()

        # Delete the files and folders from tmp and input directories
        shutil.rmtree(target_folder)
        os.unlink('%s/%s' % (input_folder, sim_results_file))
        os.unlink('%s/%s' % (tmp_folder, sim_results_file))

        # todo Send the notification email to the user that launched the simulation
        user_email = get_user_email(conn, int(id_user))
        logger.info('Simulation %s: Send email notification to %s' % (sim_id, user_email))

        logger.info('Simulation %s: Data management ending' % sim_id)

    conn.close()