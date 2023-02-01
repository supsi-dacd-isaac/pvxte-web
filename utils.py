import json
import os

import numpy as np
import pandas as pd


def read_schedule_from_solution(path, sep=";"):
    file = os.path.join(path)
    df = pd.read_csv(file, sep=sep, header=None)
    df.columns = ['var', 'val']

    df[['var', 'index1', 'index2']] = df['var'].str.split(r'\[|\]', expand=True, regex=True)
    dfx = df[df['var'] == 'x'].copy()
    dfx[['n1', 'n2', 'bus']] = dfx['index1'].str.split(',', expand=True)
    dfx = dfx.reset_index()[['var', 'bus', 'n1', 'n2', 'val']]
    return dfx


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


def merge_time_schedule(df, t_start, times, depot_start):
    df['start'] = 0
    df['end'] = 0
    df['duration'] = 0
    num_trips = len(t_start.keys())

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

    df = df.sort_values(['bus', 'start']).reset_index().drop(['index'], axis=1)
    return df
