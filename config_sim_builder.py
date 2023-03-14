import json
from collections import defaultdict

import numpy as np
import pandas as pd
from gurobipy import *
from scipy.sparse import csr_matrix


def generate_sparse_matrix(node_times: list, max_index: int) -> csr_matrix:
    """
    param:
    :param node_times: list of node travel time tuples of the form (x: x-index, y: y-index, t: time in hours)
    :param max_index: maximum index of the distance matrix
    :return: sparse distance matrix
    """
    size = max_index
    row = np.array([t[0] for t in node_times])
    col = np.array([t[1] for t in node_times])
    data = np.array([t[2] for t in node_times])
    return csr_matrix((data, (row, col)), shape=(size, size)).toarray()


def select_routes(dataset, ids, scn):
    ids = list(map(int, ids))
    return dataset[dataset.line_id.isin(ids) & dataset.day_type.isin(scn)]


def get_key_from_value(d, val):
    keys = [k for k, v in d.items() if v == val and k[0] > 0]
    if keys:
        return keys[0]
    return None


def num_buses_at_time(df):
    result = defaultdict(int)
    for index, row in df.iterrows():
        ts = row.departure_timestep
        te = row.arrival_timestep
        for s in range(ts, te + 1):
            result[s] += 1
    return result


def configuration(csv_file_path, company, route_number, battery_size, charging_locations, day_type, t_horizon, p_max, pd_max,
                  depot_charging, optimize_for_each_bus):
    # todo: battery size is read from the bus profile. So this line is redundant. Remove it and other related statements!
    if battery_size == -1:
        battery_size = 704

    # Load data files
    route_num_split = route_number.split(',')
    route_number = '-'.join(route_num_split)
    trips_times = pd.read_csv(csv_file_path)
    trips_times = select_routes(trips_times, route_num_split, day_type.split(','))
    trips_times = trips_times.reset_index().drop(['index'], axis=1)

    vehicle_ids = list(set(trips_times.bus_id.to_list()))

    with open(f'static/node-map/{company}-node-map.json', 'r') as file:
        node_map = json.load(file)

    trips_times['departure_node'] = trips_times['starting_city'].apply(lambda x: node_map[x])
    trips_times['arrival_node'] = trips_times['arrival_city'].apply(lambda x: node_map[x])

    with open(f'static/time-energy/{company}-time-energy.json', 'r') as f:
        data = json.load(f)

    with open(f'static/elevations/{company}-elevation.json', 'r') as f:
        elevation = json.load(f)

    trips_times['starting_city_ele'] = trips_times["starting_city"].apply(lambda loc: elevation[loc])
    trips_times['arrival_city_ele'] = trips_times["arrival_city"].apply(lambda loc: elevation[loc])

    txy, exy, num_nodes = data[0], data[1], data[2]
    txy = {eval(k): v for k, v in txy.items()}
    exy = {eval(k): v for k, v in exy.items()}

    if not charging_locations:
        charging_locations = []
    else:
        charging_locations = charging_locations.split(',')
        charging_locations = list(map(int, charging_locations))

    config = {"route_id": f"{route_number}",
              "company": f"{company}",
              "#vehicles": len(vehicle_ids),
              "#depot": 1,
              "charging_locations": charging_locations,
              "#trips": len(trips_times),
              "#timesteps": t_horizon,
              'battery_size': battery_size,
              'soc_start': 1.0,
              "max_charging_power": p_max,
              "max_depot_charging_power": pd_max,
              "trips_info": [],
              "depot_origin": [],
              "depot_destination": [],
              "charge_info": [],
              "vehicle_info": [],
              "trip_ids": [],
              "depot_origin_ids": [],
              "depot_destination_ids": [],
              "charge_ids": [],
              "depot_charge_ids": [],
              "vehicle_ids": [],
              "service_time": [],
              "service_energy": [],
              "charging_cost": [],
              "optimize_for_each_bus": optimize_for_each_bus,
              "Battery pack size": 88
              }
    trips = []
    charge = []

    num_trips = config["#trips"]
    num_depots = config["#depot"]
    max_index = num_depots + 3 * num_trips

    # Trip energy is in the E_total column of the dataframe.
    for index, row in trips_times.iterrows():
        trips.append({'index': index,
                      't_start': row.departure_timestep,
                      't_end': row.arrival_timestep,
                      'n_start': row.departure_node,
                      'n_end': row.arrival_node,
                      'energy': row.E_total if row.E_total is not np.nan else 0,
                      'vehicle_id': row.bus_id,
                      'distance': row.distance,
                      'start_elevation': row.starting_city_ele,
                      'end_elevation': row.arrival_city_ele,
                      'alpha': math.atan((row.arrival_city_ele - row.starting_city_ele) / row.distance)})

    config["trips_info"] = trips

    for t in config["trips_info"]:
        config["trip_ids"].append(t['index'])
        if depot_charging:
            config["depot_charge_ids"].append(t['index'] + 2 * num_trips)

        if t['n_end'] in charging_locations:
            config["charge_ids"].append(t['index'] + num_trips)

    for t in range(3 * num_trips, max_index):
        config["depot_origin"].append({'index': t, 'n_start': 0})
        config["depot_destination"].append({'index': t + config["#depot"], 'n_start': 0})

    for t in config["depot_origin"]:
        config["depot_origin_ids"].append(t['index'])
    for t in config["depot_destination"]:
        config["depot_destination_ids"].append(t['index'])

    # Set up cost matrix for trip nodes
    t1 = []
    for x, y in list(itertools.product(config["trip_ids"], config["trip_ids"])):
        if x == y:
            # Large penalty for self-loops
            t1.append((x, y, 1000))
        else:
            service_time_x = config["trips_info"][x]['t_end'] - config["trips_info"][x]['t_start']
            relocation_time = txy[config["trips_info"][x]['n_end'], config["trips_info"][y]['n_start']]
            time = service_time_x + relocation_time
            t1.append((x, y, time))

    # Set up cost matrix for traveling between a depot and a trip node
    t2 = []
    combi = list(itertools.product(config["depot_origin_ids"], config["trip_ids"])) + \
            list(itertools.product(config["trip_ids"], config["depot_destination_ids"]))

    for x, y in combi:
        if x in config["depot_origin_ids"]:
            time = txy[config["depot_origin"][x - 3 * num_trips]['n_start'], config["trips_info"][y]['n_start']]
            t2.append((x, y, time))
        else:
            service_time_x = config["trips_info"][x]['t_end'] - config["trips_info"][x]['t_start']
            relocation_time = txy[config["trips_info"][x]['n_end'], config["depot_destination"][y - 3 * num_trips - 1]['n_start']]
            t2.append((x, y, service_time_x + relocation_time))

    # Set up cost matrix between a node and the corresponding charging node.
    t3 = []
    combi = list(itertools.product(config["trip_ids"], config["charge_ids"]))

    for x, y in combi:
        # Large penalty for arcs between a node (x) and a charging node (y) of a different node.
        # Cost of going from a charging node to a different node is the travel time between the destination node and the
        # trip node corresponding to the charging node.
        if (x + num_trips != y) and (x in config["trip_ids"]):
            t3.append((x, y, 1000))
            service_time_x = config["trips_info"][y - num_trips]['t_end'] - config["trips_info"][y - num_trips]['t_start']
            relocation_time = txy[config["trips_info"][y - num_trips]['n_end'], config["trips_info"][x]['n_start']]
            t3.append((y, x, service_time_x + relocation_time))
        # The cost of an arc between a node and it's charging node is 0. Going from charging back to node is penalized.
        elif (x + num_trips == y) and (x in config["trip_ids"]):
            # service_time_x = config["trips_info"][x]['t_end'] - config["trips_info"][x]['t_start']
            t3.append((x, y, 0))
            t3.append((y, x, 1000))

    # Set up cost matrix between a charging node and the destination node.
    t4 = []
    combi = list(itertools.product(config["charge_ids"], config["depot_destination_ids"]))
    for x, y in combi:
        t4.append((x, y, txy[config["trips_info"][x - num_trips]['n_end'], config["depot_destination"][y - 3 * num_trips - 1]['n_start']]))

    # Set up cost matrix between a node and the corresponding depot_charging node.
    # Service time includes the time to service the trip node
    if depot_charging:
        t5 = []
        combi = list(itertools.product(config["trip_ids"], config["depot_charge_ids"]))

        for x, y in combi:
            # y is the depot charging node
            if (x + 2 * num_trips == y) and (x in config["trip_ids"]):
                # y is the corresponding depot charging node of x. Cost: time to go n_end to depot (s1) + time to service x (s0)
                s0 = config["trips_info"][x]['t_end'] - config["trips_info"][x]['t_start']
                s1 = txy[config["trips_info"][x]['n_end'], config["depot_destination"][0]['n_start']]
                t5.append((x, y, s0 + s1))
                t5.append((y, x, 1000))
            else:
                t5.append((x, y, 1000))
                # TODO: Fix this to make more sense
                s2 = txy[config["depot_destination"][0]['n_start'], config["trips_info"][x]['n_start']]
                t5.append((y, x, s2))

    if depot_charging:
        csr_mat = generate_sparse_matrix(t1 + t2 + t3 + t4 + t5, max_index + 1)
    else:
        csr_mat = generate_sparse_matrix(t1 + t2 + t3 + t4, max_index + 1)
    config["service_time"] = csr_mat.tolist()

    for v in vehicle_ids:
        config["vehicle_ids"].append(str(v))
        config["vehicle_info"].append({'index': str(v), 'soc_start': 1.0, 'battery_capacity': battery_size})

    # Set up the energy matrix. From node to charging node represents charging and other arcs represent discharging.
    c1 = []
    for x, y in list(itertools.product(config["trip_ids"], config["trip_ids"])):
        if x == y:
            # Self-loops, zero energy
            c1.append((x, y, 0))
        else:
            # Energy required to traverse arc (i, j) is the energy needed to service trip i and then reach the starting
            # point of the trip j. Negative sign indicates discharge.
            service_energy_x = config["trips_info"][x]['energy']
            relocation_energy = exy[config["trips_info"][x]['n_end'], config["trips_info"][y]['n_start']]
            en = service_energy_x + relocation_energy
            c1.append((x, y, -en))

    # Set up energy matrix for traveling between a depot and a trip node.
    c2 = []
    co1 = list(itertools.product(config["depot_origin_ids"], config["trip_ids"]))
    co2 = list(itertools.product(config["trip_ids"], config["depot_destination_ids"]))

    combi = co1 + co2
    for x, y in combi:
        if x in config["depot_origin_ids"]:
            en = exy[config["depot_origin"][x - 3 * num_trips]['n_start'], config["trips_info"][y]['n_start']]
            c2.append((x, y, -en))
        else:
            en = exy[config["trips_info"][x]['n_end'], config["depot_destination"][y - 3 * num_trips - 1]['n_start']]
            c2.append((x, y, -en))

    c3 = []
    combi = list(itertools.product(config["trip_ids"], config["charge_ids"]))

    for x, y in combi:
        # Large energy penalty for arcs between a node (x) and a charging node (y) of a different node.
        # Cost of going from a charging node to a different node is the energy between the destination node and the
        # trip node corresponding to the charging node.
        if (x + num_trips != y) and (x in config["trip_ids"]):
            c3.append((x, y, -0))
            service_energy = config["trips_info"][x]['energy']
            relocation_energy = exy[config["trips_info"][y - num_trips]['n_end'], config["trips_info"][x]['n_start']]
            c3.append((y, x, -service_energy - relocation_energy))
        # The cost of an arc between a node and it's charging node is 0. Going from charging back to node is penalized.
        elif x + num_trips == y:
            c3.append((x, y, -0))
            c3.append((y, x, -0))

    if depot_charging:
        c5 = []
        combi = list(itertools.product(config["trip_ids"], config["depot_charge_ids"]))

        for x, y in combi:
            if (x + 2 * num_trips == y) and (x in config["trip_ids"]):
                # y is the corresponding depot charging node of x. Cost: energy to go n_end to depot (s1) + energy to service x (s0)
                e0 = config["trips_info"][x]['energy']
                e1 = exy[config["trips_info"][x]['n_end'], config["depot_destination"][0]['n_start']]
                c5.append((x, y, -e0 - e1))
                c5.append((y, x, -1000))
            else:
                c5.append((x, y, -1000))
                e2 = exy[config["depot_destination"][0]['n_start'], config["trips_info"][x]['n_start']]
                c5.append((y, x, -e2))

    if depot_charging:
        csr_mat = generate_sparse_matrix(c1 + c2 + c3 + c5, max_index + 1)
    else:
        csr_mat = generate_sparse_matrix(c1 + c2 + c3, max_index + 1)
    config["service_energy"] = csr_mat.tolist()
    config["num_buses_at_time"] = num_buses_at_time(trips_times)

    if not charging_locations:
        csub = ['na']
    else:
        csub = [str(i) for i in charging_locations]

    cfg_file_name = 'static/sim-config/%s' % csv_file_path.split(os.sep)[-1].replace('.csv', '.json')
    with open(cfg_file_name, 'w') as outfile:
        json.dump(config, outfile, indent=4)

    return cfg_file_name
