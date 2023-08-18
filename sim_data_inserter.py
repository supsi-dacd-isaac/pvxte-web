import argparse
import json
import os
import shutil
import zipfile
import sqlite3
import logging
import sys
import smtplib
import datetime

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
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
    arg_parser.add_argument("-c", help="configuration file")
    arg_parser.add_argument("-l", help="log file (optional, if empty log redirected on stdout)")
    args = arg_parser.parse_args()

    # Load the main parameters
    config_file = args.c
    if os.path.isfile(config_file) is False:
        print('\nATTENTION! Unable to open configuration file %s\n' % config_file)
        sys.exit(1)

    cfg = json.loads(open(args.c).read())

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

    logger.info('Start program')

    input_folder = '%s/input' % cfg['inputFolder']
    tmp_folder = '%s/tmp' % cfg['inputFolder']
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

        # Send the notification email to the user that launched the simulation
        user_email = get_user_email(conn, int(id_user))
        logger.info('Simulation %s: Send email notification to %s '
                    '(server: %s; port: %i)' % (sim_id, user_email, cfg['email']['host'], cfg['email']['port']))

        ts = int(sim_id.split('_')[1])
        dt = datetime.datetime.fromtimestamp(ts)
        dt_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        text = ('Dear user,<br>the simulation you launched at %s is now completed and its results available at '
                '<a href="https://pvxte.isaac.supsi.ch">PVXTE web tool</a>') % dt_str

        try:
            smtp_server = smtplib.SMTP(cfg['email']['host'], cfg['email']['port'])
            smtp_server.starttls()
            smtp_server.login(cfg['email']['user'], cfg['email']['password'])

            message = MIMEMultipart()
            message['From'] = 'PVXTE-email-notifier'
            message['To'] = user_email
            message['Subject'] = 'PVXTE web tool: Simulation launched at %s completed' % dt_str
            message.attach(MIMEText(text, 'html'))
            res = smtp_server.sendmail(cfg['email']['user'], user_email, message.as_string())
        except Exception as e:
            logger.error("An error occurred: %s", str(e))
        finally:
            smtp_server.quit()  # Close the connection

        logger.info('Simulation %s: Data management ending' % sim_id)

    logger.info('End program')
    conn.close()
    # print(cfg['email'])