import json
import csv
import argparse

import numpy as np
from gurobipy import *
from collections import defaultdict
from datetime import datetime


def MILP(env,
         config,
         charging_power,
         default_assignment,
         partial_assignment,
         non_overlap_charging,
         max_charge_time=120,
         big_m=10000):
    depot_origin = config["depot_origin_ids"]
    depot_destination = config["depot_destination_ids"]
    trip_indices = config["trip_ids"]
    driver_indices = config["driver_ids"]
    charge_indices = config["charge_ids"]
    depot_charge_indices = config["depot_charge_ids"]
    slack_v = [f"slack_{x}" for x in range(config["#slack_vehicles"])]
    vehicle_indices = config["vehicle_ids"] + slack_v

    depot_charging = True if depot_charge_indices else False
    assert len(depot_charge_indices + charge_indices) > 0, "ERROR! Charging is allowed neither at any node nor at the depot during the day"

    battery_size = config['battery_size']
    soc_start = config['soc_start']
    P_max = charging_power
    nodes = trip_indices + charge_indices + depot_origin + depot_destination + depot_charge_indices

    service_start = {i["index"]: i["t_start"] for i in config['trips_info']}
    bus_mapping = {i["index"]: i["vehicle_id"] for i in config['trips_info']}

    # time[i, j] = service time of trip i + relocation time to node j.
    time = np.array(config['service_time'])
    E = np.array(config["service_energy"])
    n_timesteps = config["#timesteps"]
    num_buses_at_time = defaultdict(int, config["num_buses_at_time"])
    scale_factor = 1.0
    driver_constraints = False

    print('')

    if partial_assignment:
        mapping_new = {}
        lb, ub = partial_assignment[0], partial_assignment[1]
        print(f"Partial assignment with lb {lb} and ub {ub}...")

        list_mapping = defaultdict(list)
        for k, v in bus_mapping.items():
            list_mapping[v].append(k)

        for k, v in list_mapping.items():
            for i in range(len(v) - 1):
                t1 = v[i]
                t2 = v[i + 1]
                if (-service_start[t1] - time[t1, t2] + service_start[t2] >= lb) and \
                        (-service_start[t1] - time[t1, t2] + service_start[t2] <= ub):
                    mapping_new[t1] = k
                    mapping_new[t2] = k
        print(f"Partial assignment: {len(mapping_new)} of {len(trip_indices)} trips assigned.")
    else:
        print("No partial assignment parameter given. Continuing with default vehicle assignment.")

    print(f"Charging overlap constraint {non_overlap_charging}")
    print(f"Maximum charging time {max_charge_time}")
    print(f"Maximum charging power {P_max} kW")
    print(f"Maximum battery size {battery_size} kWh")
    print(f'Depot charging {bool(depot_charge_indices)}')
    print(f'Node charging {bool(charge_indices)}')
    print('')

    model = Model("vrp", env)

    """ Trip scheduling related variables"""
    # If the arc (i, j) is traversed by bus k.
    x = model.addVars(nodes, nodes, vehicle_indices, vtype=GRB.BINARY, name="x")
    # Trip i is completed by bus k.
    mx = model.addVars(trip_indices, vehicle_indices, vtype=GRB.BINARY, name="map")
    # Vehicle k is in use
    u = model.addVars(vehicle_indices, vtype=GRB.BINARY, name="u")

    """ Bus status variables"""
    if driver_constraints:
        # Vehicle k is active at timestep t
        ut = model.addVars(vehicle_indices, np.arange(n_timesteps), vtype=GRB.BINARY, name="ut")

    """ Driver related variables"""
    if driver_constraints:
        # Driver d is assigned to vehicle k at timestep t.
        vt = model.addVars(driver_indices, vehicle_indices, np.arange(n_timesteps), vtype=GRB.BINARY, name="vt")
        # Driver is starting a new shift at time t. A new shift begins after a break of at least 45 mins.
        st = model.addVars(driver_indices, np.arange(n_timesteps), vtype=GRB.BINARY, name="st")

    """ Charging related variables """
    # Charging time after trip i until trip j.
    Ct = model.addVars(charge_indices + depot_charge_indices, vtype=GRB.INTEGER, lb=0, ub=max_charge_time, name="Ct")
    # Charging node corresponding to trip node i is active.
    yc = model.addVars(trip_indices, vtype=GRB.BINARY, name="yc")
    # Depot charging node corresponding to trip node i is active.
    yd = model.addVars(trip_indices, vtype=GRB.BINARY, name="yd")

    #  SOC of vehicle k at node i.
    SOC = model.addVars(vehicle_indices, nodes, lb=0, ub=1.0, name="SOC")
    #  SOC change of vehicle k at node i.
    DSOC = model.addVars(vehicle_indices, trip_indices + depot_charge_indices, lb=0, ub=1.0, name="DSOC")

    """ Variables used for planning """
    # Battery size.
    b = model.addVar(name="b")
    bp = model.addVar(lb=1, ub=battery_size, name="bp")
    bs = model.addVar(name="bs")
    bi = model.addVar(name="bi")

    """ Objective function."""
    obj = model.addVar(name="objective")

    """ Auxiliary variables """
    # Auxiliary variables used to linearize the trip lag time constraints.
    if depot_charge_indices:
        s1 = model.addVars(depot_charge_indices, trip_indices, lb=-big_m, ub=big_m, name="s1")
        s5 = model.addVars(depot_charge_indices, trip_indices, vehicle_indices, lb=-1, ub=1, name="s5")
        s6 = model.addVars(depot_charge_indices, trip_indices, vehicle_indices, lb=-1, ub=1, name="s6")

    if charge_indices:
        s2 = model.addVars(charge_indices, trip_indices, lb=-big_m, ub=big_m, name="s2")
        s4 = model.addVars(charge_indices, trip_indices + depot_destination, vehicle_indices, lb=-1, ub=1, name="s4")

    # Auxiliary variables used to linearize the discharging constraints.
    s3 = model.addVars(trip_indices + depot_origin, trip_indices + depot_destination, vehicle_indices, lb=-1, ub=1, name="s3")

    """=============== CONSTRAINTS ==============="""
    model.addConstr(obj == 0
                    - 10 * quicksum(SOC[k, n] for k in vehicle_indices for n in depot_destination)
                    + 1000 * quicksum(u[k] for k in slack_v)
                    + bp
                    + 10000 * bs
                    )

    # Invalid arcs
    # Self loops are not allowed.
    model.addConstrs(x[i, i, k] == 0 for k in vehicle_indices for i in nodes)
    # You cannot enter depot origin node from anywhere else.
    model.addConstrs(x[i, j, k] == 0 for i in nodes for j in depot_origin for k in vehicle_indices)
    # You cannot leave depot destination node, it must be the final node.
    model.addConstrs(x[i, j, k] == 0 for i in depot_destination for j in nodes for k in vehicle_indices)
    # You cannot reach any charging node from the depot origin.
    model.addConstrs(x[i, j, k] == 0 for i in depot_origin for j in charge_indices + depot_charge_indices for k in vehicle_indices)
    # You cannot enter one charging node from another charging node.
    model.addConstrs(x[i, j, k] == 0 for i in charge_indices + depot_charge_indices for j in charge_indices + depot_charge_indices
                     for k in vehicle_indices)
    # There is no direct edge between depot origin and depot destination.
    model.addConstrs(x[i, j, k] == 0 for i in depot_origin for j in depot_destination for k in vehicle_indices)

    if charge_indices:
        # You cannot go back to the trip node from its charging node, the edge is not bidirectional.
        model.addConstrs(x[i, i - len(trip_indices), k] == 0 for i in charge_indices for k in vehicle_indices)

    if depot_charge_indices:
        # You cannot go back to the trip node from its depot charging node, the edge is not bidirectional.
        model.addConstrs(x[i, i - 2 * len(trip_indices), k] == 0 for i in depot_charge_indices for k in vehicle_indices)
        # There is no edge between depot charging nodes and the depot destination node.
        model.addConstrs(x[i, j, k] == 0 for i in depot_charge_indices for j in depot_destination for k in vehicle_indices)
        # There is no edge between depot origin and the depot charging nodes.
        model.addConstrs(x[i, j, k] == 0 for i in depot_origin for j in depot_charge_indices for k in vehicle_indices)

    # Remove arcs that violate the time-window constraint.
    for i in trip_indices:
        for j in trip_indices:
            if (service_start[i] + time[i, j] - service_start[j]) > 0:
                model.addConstrs(x[i, j, k] == 0 for k in vehicle_indices)

    if depot_charging:
        # Remove edges with depot charging nodes that violate time window constraint
        for i in trip_indices:
            for j in trip_indices:
                if (service_start[i] + time[i, i + 2 * len(trip_indices)] + time[i + 2 * len(trip_indices), j] - service_start[j]) > 0:
                    model.addConstrs(x[i + 2 * len(trip_indices), j, k] == 0 for k in vehicle_indices)

    # For checking the feasibility of a given schedule, assign each trip to a bus based on the schedule.
    if default_assignment:
        for i, k in bus_mapping.items():
            model.addConstr(mx[i, str(k)] == 1)
            model.addConstr(u[str(k)] == 1)

    # For optimizing partial schedule, assign the trips you want fixed and leave the rest as decision variables.
    if partial_assignment:
        for i, k in mapping_new.items():
            model.addConstr(mx[i, str(k)] == 1)
            model.addConstr(u[str(k)] == 1)

    # Bus k serves at most one trip j that is either follows trip i, origin depot, or one of the charging nodes.
    # At most condition is set by mx being a binary variable.
    model.addConstrs(mx[j, k] == quicksum(x[i, j, k] for i in trip_indices) +
                     quicksum(x[i, j, k] for i in depot_origin) +
                     quicksum(x[i, j, k] for i in charge_indices) for j in trip_indices for k in vehicle_indices)

    # Total number of vehicles should be less than the available number.
    model.addConstr(quicksum(u[k] for k in vehicle_indices) <= config["#vehicles"])

    # If arc (i, j) is served by vehicle k, it must be active.
    model.addConstrs(u[k] >= x[i, j, k]
                     for i in nodes
                     for j in nodes
                     for k in vehicle_indices)

    # Each trip node is visited exactly once.
    # A trip node can be visited from depot origin and depot destination node can be visited from any trip node.
    # Charge nodes can be visited trip nodes.
    if charge_indices and depot_charge_indices:
        model.addConstrs(quicksum(x[i, j, k] for i in depot_origin for k in vehicle_indices) +
                         quicksum(x[i, j, k] for i in trip_indices for k in vehicle_indices) +
                         quicksum(x[i, j, k] for i in charge_indices for k in vehicle_indices if i + len(trip_indices) != j) +
                         quicksum(x[i, j, k] for i in depot_charge_indices for k in vehicle_indices if i - 2 * len(trip_indices) != j) == 1
                         for j in trip_indices)

        model.addConstrs(quicksum(x[j - len(trip_indices), i, k] for i in depot_destination for k in vehicle_indices) +
                         quicksum(x[j - len(trip_indices), i, k] for i in trip_indices for k in vehicle_indices) +
                         quicksum(x[j - len(trip_indices), j, k] for k in vehicle_indices) +
                         quicksum(x[j - len(trip_indices), j + len(trip_indices), k] for k in vehicle_indices) == 1
                         for j in charge_indices)

    if depot_charge_indices and not charge_indices:
        model.addConstrs(quicksum(x[i, j, k] for i in depot_origin for k in vehicle_indices) +
                         quicksum(x[i, j, k] for i in trip_indices for k in vehicle_indices) +
                         quicksum(x[i, j, k] for i in depot_charge_indices for k in vehicle_indices if i - 2 * len(trip_indices) != j) == 1
                         for j in trip_indices)

        model.addConstrs(quicksum(x[j - 2 * len(trip_indices), i, k] for i in depot_destination for k in vehicle_indices) +
                         quicksum(x[j - 2 * len(trip_indices), i, k] for i in trip_indices for k in vehicle_indices) +
                         quicksum(x[j - 2 * len(trip_indices), j, k] for k in vehicle_indices) == 1
                         for j in depot_charge_indices)

    if not depot_charge_indices and charge_indices:
        model.addConstrs(quicksum(x[i, j, k] for i in depot_origin for k in vehicle_indices) +
                         quicksum(x[i, j, k] for i in trip_indices for k in vehicle_indices) +
                         quicksum(x[i, j, k] for i in charge_indices for k in vehicle_indices if i - len(trip_indices) != j) == 1
                         for j in trip_indices)

        model.addConstrs(quicksum(x[j - len(trip_indices), i, k] for i in depot_destination for k in vehicle_indices) +
                         quicksum(x[j - len(trip_indices), i, k] for i in trip_indices for k in vehicle_indices) +
                         quicksum(x[j - len(trip_indices), j, k] for k in vehicle_indices) == 1
                         for j in charge_indices)

    # If vehicle k is in use, it has to start from the depot and end at the depot.
    model.addConstrs(u[k] == quicksum(x[i, j, k] for j in trip_indices) for i in depot_origin for k in vehicle_indices)
    model.addConstrs(u[k] == quicksum(x[i, j, k] for i in trip_indices + charge_indices)
                     for k in vehicle_indices
                     for j in depot_destination)

    # From an active charge node, you can only reach a trip node or the destination node.
    if charge_indices:
        model.addConstrs(quicksum(x[i, j, k] for j in trip_indices for k in vehicle_indices) +
                         quicksum(x[i, j, k] for j in depot_destination for k in vehicle_indices) == yc[i - len(trip_indices)]
                         for i in charge_indices)

    if depot_charge_indices:
        model.addConstrs(quicksum(x[i, j, k] for j in trip_indices for k in vehicle_indices) == yd[i - 2 * len(trip_indices)]
                         for i in depot_charge_indices)

    # Consecutive arcs must be served by the same vehicle.
    model.addConstrs(quicksum(x[i, j, k] for i in trip_indices + charge_indices + depot_origin + depot_charge_indices) ==
                     quicksum(x[j, p, k] for p in nodes)
                     for j in trip_indices + charge_indices + depot_charge_indices
                     for k in vehicle_indices)

    if depot_charge_indices:
        # Depot charging time must be less than or equal to the time window between two trips.
        model.addConstrs(service_start[i - 2 * len(trip_indices)] + time[i - 2 * len(trip_indices), i] + time[i, j] +
                         Ct[i] - service_start[j] + s1[i, j] <= 0
                         for i in depot_charge_indices
                         for j in trip_indices if i - 2 * len(trip_indices) != j)

        model.addConstrs(s1[i, j] <= big_m * (1 - quicksum(x[i, j, k] for k in vehicle_indices))
                         for i in depot_charge_indices
                         for j in trip_indices if i - 2 * len(trip_indices) != j)

        model.addConstrs(s1[i, j] >= -big_m * (1 - quicksum(x[i, j, k] for k in vehicle_indices))
                         for i in depot_charge_indices
                         for j in trip_indices if i - 2 * len(trip_indices) != j)

    # Charging time must be less than or equal to the time window between two trips. .
    if charge_indices:
        model.addConstrs(service_start[i - len(trip_indices)] + time[i - len(trip_indices), j] +
                         Ct[i] - service_start[j] + s2[i, j] <= 0
                         for i in charge_indices
                         for j in trip_indices if i - len(trip_indices) != j)

        model.addConstrs(s2[i, j] <= big_m * (1 - quicksum(x[i, j, k] for k in vehicle_indices))
                         for i in charge_indices
                         for j in trip_indices if i - len(trip_indices) != j)

        model.addConstrs(s2[i, j] >= -big_m * (1 - quicksum(x[i, j, k] for k in vehicle_indices))
                         for i in charge_indices
                         for j in trip_indices if i - len(trip_indices) != j)

    # Charging node is active if there is an active arc between a node, and it's charging node.
    if charge_indices:
        model.addConstrs(yc[i - len(trip_indices)] == quicksum(x[i - len(trip_indices), i, k] for k in vehicle_indices)
                         for i in charge_indices)

    if depot_charge_indices:
        model.addConstrs(yd[i - 2 * len(trip_indices)] == quicksum(x[i - 2 * len(trip_indices), i, k] for k in vehicle_indices)
                         for i in depot_charge_indices)

    # Initialize SOC for all vehicles
    model.addConstrs(SOC[k, i] == soc_start for k in vehicle_indices for i in depot_origin)
    model.addConstrs(SOC[k, i] >= 0.2 for k in vehicle_indices for i in nodes)

    # Discharging dynamics.
    model.addConstrs(SOC[k, j] - (SOC[k, i] + scale_factor * E[i, j] * b) + s3[i, j, k] == 0
                     for i in trip_indices + depot_origin
                     for j in trip_indices + depot_destination
                     for k in vehicle_indices)

    model.addConstrs(s3[i, j, k] <= 1 * (1 - x[i, j, k]) for i in trip_indices + depot_origin
                     for j in trip_indices + depot_destination for k in vehicle_indices if i != j)

    model.addConstrs(s3[i, j, k] >= -1 * (1 - x[i, j, k]) for i in trip_indices + depot_origin
                     for j in trip_indices + depot_destination for k in vehicle_indices if i != j)

    # Discharging dynamics involving charge nodes.
    # Going from a charging node to another node costs the service energy of the corresponding trip node + relocation energy to the next
    # node.
    if charge_indices:
        model.addConstrs(SOC[k, j] - (SOC[k, i] + scale_factor * E[i - len(trip_indices), j] * b) + s4[i, j, k] == 0
                         for i in charge_indices
                         for j in trip_indices + depot_destination
                         for k in vehicle_indices)

        model.addConstrs(s4[i, j, k] <= 1 * (1 - x[i, j, k]) for i in charge_indices
                         for j in trip_indices + depot_destination for k in vehicle_indices if i != j)
        model.addConstrs(s4[i, j, k] >= -1 * (1 - x[i, j, k]) for i in charge_indices
                         for j in trip_indices + depot_destination for k in vehicle_indices if i != j)

    if depot_charge_indices:
        # Discharging dynamics at a depot charging node.
        # Leaving a depot charging node costs relocation energy to the next node.
        model.addConstrs(SOC[k, j] - (SOC[k, i] + scale_factor * E[i - 2 * len(trip_indices), j] * b) + s6[i, j, k] == 0
                         for i in depot_charge_indices
                         for j in trip_indices
                         for k in vehicle_indices)

        model.addConstrs(s6[i, j, k] <= 1 * (1 - x[i, j, k]) for i in depot_charge_indices for j in trip_indices for k in vehicle_indices)
        model.addConstrs(s6[i, j, k] >= -1 * (1 - x[i, j, k]) for i in depot_charge_indices for j in trip_indices for k in vehicle_indices)

    # Charging dynamics
    # Charging involving depot charging nodes.
    # Arrival to a depot charging node costs the service energy at the corresponding trip node + relocation energy to depot.
    # Arrival to a depot charging node adds energy corresponding to DSOC.
    if depot_charge_indices:
        model.addConstrs(SOC[k, i] - (SOC[k, i - 2 * len(trip_indices)] + scale_factor * E[i - 2 * len(trip_indices), i] * b) +
                         DSOC[k, i] + s5[i, i - 2 * len(trip_indices), k] == 0
                         for i in depot_charge_indices
                         for k in vehicle_indices)

        model.addConstrs(s5[i, i - 2 * len(trip_indices), k] <= 1 * (1 - x[i - 2 * len(trip_indices), i, k])
                         for i in depot_charge_indices for k in vehicle_indices)
        model.addConstrs(s5[i, i - 2 * len(trip_indices), k] >= -1 * (1 - x[i - 2 * len(trip_indices), i, k])
                         for i in depot_charge_indices for k in vehicle_indices)

    # Coming into a charging node from the corresponding trip node models the charging action.
    if charge_indices:
        model.addConstrs(SOC[k, j] == SOC[k, j - len(trip_indices)] + DSOC[k, j - len(trip_indices)] * x[j - len(trip_indices), j, k]
                         for j in charge_indices
                         for k in vehicle_indices)

    # Charge time is calculated from the SOC difference and charging power.
    if charge_indices:
        model.addConstrs(Ct[i] * b >= quicksum(DSOC[k, i - len(trip_indices)]
                                               for k in vehicle_indices) * 60 / P_max
                         for i in charge_indices)

    if depot_charge_indices:
        model.addConstrs(Ct[i] * b >= quicksum(DSOC[k, i] for k in vehicle_indices) * 60 / P_max
                         for i in depot_charge_indices)

    model.addConstr(b * bi == 1)
    model.addConstr(bp + bs == bi)

    if charge_indices:
        model.addConstrs((yc[i - len(trip_indices)] == 1) >> (Ct[i] >= 1) for i in charge_indices)
        model.addConstrs((yc[i - len(trip_indices)] == 0) >> (Ct[i] == 0) for i in charge_indices)

    if depot_charge_indices:
        model.addConstrs((yd[i - 2 * len(trip_indices)] == 1) >> (Ct[i] >= 1) for i in depot_charge_indices)
        model.addConstrs((yd[i - 2 * len(trip_indices)] == 0) >> (Ct[i] == 0) for i in depot_charge_indices)

    # Non-overlapping constraint for charging. At depot there is no overlapping constraint.
    if charge_indices:
        if non_overlap_charging:
            for i in charge_indices:
                model.addConstrs(service_start[j] - service_start[i - len(trip_indices)] >= Ct[i]
                                 for j in trip_indices
                                 if i - len(trip_indices) != j and service_start[j] > service_start[i - len(trip_indices)])

    """ Driver related constraints """
    if driver_constraints:
        # Set values for ut.
        model.addConstrs((x[i, j, k] == 1) >> (ut[k, t] == 1)
                         for i in trip_indices
                         for j in trip_indices + depot_destination
                         for k in vehicle_indices
                         for t in range(service_start[i], service_start[i] + time[i, j])
                         if i != j and service_start[i] + time[i, j] < n_timesteps)

        model.addConstrs((x[i - 2 * len(trip_indices), i, k] == 1) >> (ut[k, t] == 1) for i in depot_charge_indices
                         for k in vehicle_indices
                         for t in range(service_start[i - 2 * len(trip_indices)],
                                        service_start[i - 2 * len(trip_indices)] + time[i - 2 * len(trip_indices), i])
                         if service_start[i - 2 * len(trip_indices)] + time[i - 2 * len(trip_indices), i] < n_timesteps)

        model.addConstrs((x[j, i, k] == 1) >> (ut[k, t] == 1)
                         for i in trip_indices
                         for j in depot_charge_indices
                         for k in vehicle_indices
                         for t in range(service_start[j - 2 * len(trip_indices)],
                                        service_start[j - 2 * len(trip_indices)] + time[j - 2 * len(trip_indices), i])
                         if (j - 2 * len(trip_indices) != i) and
                         (service_start[j - 2 * len(trip_indices)] + time[j - 2 * len(trip_indices), i] < n_timesteps))

        model.addConstrs((x[i, j, k] == 1) >> (ut[k, t] == 1)
                         for i in depot_origin
                         for j in trip_indices
                         for k in vehicle_indices
                         for t in range(service_start[j] - time[i, j], service_start[j]))

        # An active bus must have a driver.
        model.addConstrs(quicksum(vt[d, v, t] for d in driver_indices) == ut[v, t] for v in vehicle_indices for t in range(n_timesteps))

        # If a driver starts a new shift at time t, he must have been idle for the previous 45 mins.
        model.addConstrs(st[d, t] == quicksum(vt[d, v, k] for v in vehicle_indices for k in range(max(0, t - 45), t))
                         for d in driver_indices
                         for t in range(n_timesteps))

        # Number of drivers working on shift must be (at least) equal to the number of buses active at that time.
        model.addConstrs(num_buses_at_time[t] <= quicksum(st[d, n] for d in driver_indices for n in range(max(0, t - 270), t))
                         for t in range(n_timesteps))

    model.setObjective(obj, GRB.MINIMIZE)
    model.update()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the optimization model with given arguments.')

    parser.add_argument('-c', '--config_type', required=True, type=str, help='Config file (without file extension).')
    parser.add_argument('-v', '--partial_assignment', required=False, type=str, default=[], help='Cutoffs for partial vehicle assignment.')
    parser.add_argument('-o', '--non_overlap_charging', required=False, action='store_true', help='Use non-overlapped charging constraint.')
    parser.add_argument('-p', '--power', required=False, default=150, type=int, help='Hybrid charging power.')

    args = parser.parse_args()

    config_type = args.config_type  # <route_ids>-v<num_vehicles>-c<charging_nodes>, e. g., '301-v10-c12'
    partial = args.partial_assignment
    non_overlap = args.non_overlap_charging
    power = args.power

    print('')
    start = datetime.now()
    print("Start time: ", start)

    # Load data files
    with open(f"./data/config/{config_type}.json", "r") as f:
        config_file = json.load(f)

    if partial:
        partial = partial.split(',')
        partial = list(map(int, partial))

    default = False if partial else True

    e = Env("gurobi.log", params={'MemLimit': 30,
                                  'PreSparsify': 1,
                                  'MIPFocus': 3,
                                  'NodefileStart': 0.5,
                                  'Heuristics': 0.3,
                                  'Presolve': 1,
                                  'NoRelHeurTime': 60,
                                  'NonConvex': 2,
                                  'MIPGap': 0.01})

    model = MILP(config=config_file,
                 env=e,
                 charging_power=power,
                 default_assignment=default,
                 partial_assignment=partial,
                 non_overlap_charging=non_overlap)

    # print('Writing model file')
    # model.write('model_mps_format.mps')
    print('')
    print('Starting optimizer...', datetime.now())
    model.optimize()

    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible!!!!")

    res_var = ['bp', 'bs', 'bi', 'Ct', 'u', 'x', 'yd', 'SOC']
    varInfo = [(v.varName, v.X) for v in model.getVars() if (v.X != 0) and any([v.varName.startswith(s) for s in res_var])]

    # Write to input-csv
    if partial:
        sub = str(int(default)) + "-" + str(partial[0]) + "-" + str(partial[1]) + "-" + str(int(non_overlap))
    else:
        sub = str(int(default)) + "-default-" + str(int(non_overlap))

    with open(f'./sol/{config_file["route_id"]}/{config_type}-{sub}-{power}.csv', 'w', newline='') as f:
        wr = csv.writer(f, delimiter=";")
        wr.writerows(varInfo)

    print('')
    end = datetime.now()
    print("End time: ", end)
    print('Solution time (seconds): ', (end - start).total_seconds())
