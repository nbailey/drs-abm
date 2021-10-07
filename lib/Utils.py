"""
utility functions are found here
"""

import csv
import copy
import datetime
import numpy as np

from lib.RoutingEngine import *
from lib.Agents import *
from lib.Demand import *
from lib.Constants import *
from lib.ModeChoice import *
from lib.ModeChoiceAlt import *

if IS_ANIMATION:
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import matplotlib.image as mpimg


def initialize_simulation(router, step, demand_filepath, fleet_specs, seeds=None):
    # frames record the states of the AMoD model for animation purpose
    frames = []

    iter_seeds = [None] * (len(fleet_specs)+2)
    if not USE_SEEDS:
        print('Step {}', file=open('output/seeds.txt', 'a+'))
    elif seeds is None:
        iter_seeds = AMOD_SEEDS[step]
    else:
        iter_seeds = seeds

    # Initialize fleets and model
    fleets = dict()
    for i, spec in enumerate(fleet_specs):
        print("Creating {}th fleet - name: {}".format(i, spec["name"]))
        f = Fleet(router, spec["fare"], spec["fleet_size"], spec["veh_capacity"], spec["assign_method"], spec["window_method"], spec["rebl_method"], seed=iter_seeds[i])
        fleets[spec["name"]] = f

    print("Initializing demand with data from {}".format(demand_filepath))
    demand = DemandUpdater(seed=iter_seeds[i+1])
    demand.addTripsFromCsv(demand_filepath)

    print("Creating model with specified fleets and demand...")
    model = Model(demand, fleets, seed=iter_seeds[i+2])

    return model


def run_iteration(model, router):
    frames = list()
    stime = time.time()

    # start time
    stime = time.time()
    # dispatch the system for T_TOTAL seconds, at the interval of INT_ASSIGN
    for T in range(0, T_TOTAL, INT_ASSIGN):
        model.dispatch_at_time(router, T)
        if IS_ANIMATION:
            frames.append(copy.deepcopy(model.vehs))
    # end time
    etime = time.time()
    # run time of this simulation
    runtime = etime - stime

    return runtime, frames


def postprocess_iteration(model, run):
    req_data = dict()

    # For each fleet in the model
    for fleet_name in model.fleets.keys():
        fleet = model.fleets[fleet_name]

        # Get request data
        fleet_req_data = save_fleet_request_data(fleet)

        save_fleet_data(fleet_name, fleet, run, "./output/{}_FleetResults_{}.csv".format(EXP_NAME, fleet_name))

        req_data[fleet_name] = fleet_req_data

    # Synthesize the request data from each fleet into one combined output file
    comb_req_data = combine_request_data(req_data, "./output/{}_ReqData_Combined_Iter{}.csv".format(EXP_NAME, run))

    return req_data


def run_full_simulation(router, num_iters, demand_filepath, fleet_specs, initial_kernel_data, seeds=None):
    print("Initializing fleets with pre-generated kernel data....")
    # TODO: Create some default kernel generating data to use or something
    kernel_generating_data = initial_kernel_data
    run_weights = [1,]

    old_models = list()

    all_req_data = {fleet["name"]: list() for fleet in fleet_specs}

    for step in range(num_iters):
        model = initialize_simulation(router, step, demand_filepath, fleet_specs, seeds[step])

        for fleet_name in model.fleets.keys():
            fleet = model.fleets[fleet_name]

            print("Generating travel time kernels for fleet {}...".format(fleet_name))
            fleet_req_data = kernel_generating_data[fleet_name]
            fleet.generate_historical_kernels(fleet_req_data, run_weights)

        print("\nBeginning iteration {}...\n".format(step))
        runtime, frames = run_iteration(model, router)

        # generate, show and save the animation of this simulation
        if IS_ANIMATION:
            anime = anim(frames)
            anime.save('output/anim.mp4', dpi=300, fps=None, extra_args=['-vcodec', 'libx264'])
            plt.show()

        old_models.append(model)
        print("\nPostprocessing data from iteration {}...\n".format(fleet_name))
        iter_req_data = postprocess_iteration(model, step)

        for fleet in fleet_specs:
            fleet_name = fleet["name"]
            all_req_data[fleet_name].append(iter_req_data[fleet_name])

        kernel_generating_data = all_req_data
        # Each run's weight gets halved when the next run is added
        run_weights = [2**((-1.0)*(i-1)) for i in range(step+1, 0, -1)]
        print("\tWeights updated for historical kernels, now: {}".format(run_weights))

        old_models.append(model)

    return old_models


def save_fleet_request_data(fleet, out_filename=None):
    reqDF = pd.DataFrame(columns=['rid', 'OLat', 'OLng', 'DLat', 'DLng', 'T_req_pickup', 'T_pickup', 'T_dropoff',
                                  'Is_OnDemand', 'Veh_ID', 'New_OE', 'Num_Pickups', 'Num_Dropoffs', 'PWT', 'RWT', 'AWT', 'WT_AbsDev',
                                  'PVT', 'RVT', 'AVT', 'VT_AbsDev', 'T_Dir_Veh', 'T_Dir_Wait'])
    req_keys = list(reqDF.columns)

    # analyze requests that were requested within the period of study
    for req in fleet.reqs:
        if req.Tr >= T_WARM_UP and req.Cep <= T_WARM_UP+T_STUDY and not req.DR:
            req_data = [req.id, req.olat, req.olng, req.dlat, req.dlng, req.Cep,
                        req.Tp, req.Td, req.OnD, req.assigned_veh, req.new_occ, req.NP, req.ND,
                        req.fpwt, req.rewt, req.Tp-req.Cep, req.wtdev, req.fpvt, req.revt,
                        req.Td-req.Tp, req.vtdev, req.Ts, req.Rs]
            req_row = {req_keys[i]: req_data[i] for i in range(len(req_keys))}
            reqDF = reqDF.append(req_row, ignore_index=True)

    print("")
    if out_filename is not None:
        reqDF.to_csv(out_filename)

    return reqDF

def combine_request_data(request_data, out_filename=None):
    all_req_data = pd.DataFrame()

    for fleet_name in request_data.keys():
        fleet_data = request_data[fleet_name]

        all_req_data = all_req_data.append(fleet_data, ignore_index=True)

    if out_filename is not None:
        all_req_data.to_csv(out_filename)

    return all_req_data

def save_fleet_data(fleet_name, fleet, step, out_filename=None):
    count_reqs = 0
    count_reqs_ond = 0
    count_reqs_adv = 0
    count_served = 0
    count_served_ond = 0
    count_served_adv = 0
    count_distance_rejs = 0
    wait_time = 0.0
    wait_time_adj = 0.0
    wait_time_ond = 0.0
    wait_time_adv = 0.0
    in_veh_time = 0.0
    detour_factor = 0.0
    simulated_reqs = 0

    for req in fleet.reqs:
        simulated_reqs += 1
        if req.Tr >= T_WARM_UP and req.Cep <= T_WARM_UP+T_STUDY and not req.DR:
            count_reqs += 1
            count_reqs_ond += 1 if req.OnD else 0
            count_reqs_adv += 0 if req.OnD else 1
            # count as "served" only when the request is complete, i.e. the dropoff time is not -1
            if not np.isclose(req.Td, -1.0):
                count_served += 1
                count_served_ond += 1 if req.OnD else 0
                count_served_adv += 0 if req.OnD else 1
                wait_time_ond += (req.Tp - req.Cep) if req.OnD else 0
                wait_time_adv += 0 if req.OnD else (req.Tp - req.Cep)
                in_veh_time += (req.Td - req.Tp)
                detour_factor += req.D
        elif req.DR and req.Cep >= T_WARM_UP and req.Cep <= T_WARM_UP+T_STUDY:
            count_distance_rejs += 1

    if not count_served == 0:
        in_veh_time /= count_served
        detour_factor /= count_served
        wait_time = (wait_time_ond + wait_time_adv)/count_served
    if not count_served_ond == 0:
        wait_time_ond /= count_served_ond
    if not count_served_adv == 0:
        wait_time_adv /= count_served_a

    # service rate
    service_rate = 0.0
    service_rate_ond = 0.0
    service_rate_adv = 0.0
    if not count_reqs == 0:
        service_rate = 100.0 * count_served / count_reqs
        wait_time_adj = (wait_time * service_rate + 2*MAX_WAIT * (100-service_rate))/100
    if not count_reqs_ond == 0:
        service_rate_ond = 100.0 * count_served_ond / count_reqs_ond
    if not count_reqs_adv == 0:
        service_rate_adv = 100.0 * count_served_adv / count_reqs_adv

    # vehicle performance
    veh_service_dist = 0.0
    veh_service_time = 0.0
    veh_pickup_dist = 0.0
    veh_pickup_time = 0.0
    veh_rebl_dist = 0.0
    veh_rebl_time = 0.0
    veh_load_by_dist = 0.0
    veh_load_by_time = 0.0
    veh_occupancy_events = 0.0
    veh_trips_per_occupancy = 0.0
    cost = 0.0

    for veh in fleet.vehs:
        veh_service_dist += veh.Ds
        veh_service_time += veh.Ts
        veh_pickup_dist += veh.Dp
        veh_pickup_time += veh.Tp
        veh_rebl_dist += veh.Dr
        veh_rebl_time += veh.Tr
        if not veh.Ds + veh.Dp + veh.Dr == 0:
            veh_load_by_dist += veh.Ld / (veh.Ds + veh.Dp + veh.Dr)
        veh_load_by_time += veh.Lt / T_STUDY
        veh_occupancy_events += veh.oes
        cost += COST_BASE + COST_MIN * T_STUDY/60 + COST_KM/1000 * (veh.Ds + veh.Dp + veh.Dr)
    veh_service_dist /= fleet.V
    veh_service_time /= fleet.V
    veh_service_time_percent = 100.0 * veh_service_time / T_STUDY
    veh_pickup_dist /= fleet.V
    veh_pickup_time /= fleet.V
    veh_pickup_time_percent = 100.0 * veh_pickup_time / T_STUDY
    veh_rebl_dist /= fleet.V
    veh_rebl_time /= fleet.V
    veh_rebl_time_percent = 100.0 * veh_rebl_time / T_STUDY
    veh_load_by_dist /= fleet.V
    veh_load_by_time /= fleet.V

    if not count_served == 0:
        veh_trips_per_occupancy = count_served / veh_occupancy_events

    if out_filename is not None:
        with open(out_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [fleet_name, step, fleet.assign_method, fleet.window_method, fleet.rebl_method, T_STUDY, fleet.V, fleet.K,
                    service_rate, count_served, count_reqs, service_rate_ond, count_served_ond, count_reqs_ond, service_rate_adv, count_served_adv, count_reqs_adv,
                    wait_time, wait_time_adj, wait_time_ond, wait_time_adv, in_veh_time, detour_factor, veh_service_dist, veh_service_time, veh_service_time_percent,
                    veh_pickup_dist, veh_pickup_time, veh_pickup_time_percent,
                    veh_rebl_dist, veh_rebl_time, veh_rebl_time_percent, veh_load_by_dist, veh_load_by_time, veh_occupancy_events, veh_trips_per_occupancy,
                    simulated_reqs]
            writer.writerow(row)


# print and save results
def print_results(model, step, runtime, fare, logsum_w_AVPT, logsum_wout_AVPT,fleet_size,fare_multiplier, df_diffprob, filename, asc_name=ASC_NAME, requests_filename=None):

    print(requests_filename)

    count_reqs = 0
    count_reqs_ond = 0
    count_reqs_adv = 0
    count_served = 0
    count_served_ond = 0
    count_served_adv = 0
    count_distance_rejs = 0
    wait_time = 0.0
    wait_time_adj = 0.0
    wait_time_ond = 0.0
    wait_time_adv = 0.0
    in_veh_time = 0.0
    detour_factor = 0.0
    benefit = 0.0
    simulated_reqs = 0

    df_OD_LOS = pd.DataFrame(pd.np.empty((1057, 6)) * pd.np.nan,columns=['m_id', 'summed_wait_time', 'summed_detour_factor', 'number_of_occurrences','wait_time', 'detour_factor'])
    df_OD_LOS['m_id'] = df_OD_LOS.index
    df_OD_LOS['number_of_occurrences'] = 0
    df_OD_LOS['summed_wait_time'] = 0
    df_OD_LOS['summed_detour_factor'] = 0

    reqDF = pd.DataFrame(columns=['rid', 'OLat', 'OLng', 'DLat', 'DLng', 'T_req_pickup', 'T_pickup', 'T_dropoff',
                                  'Is_OnDemand', 'Veh_ID', 'New_OE', 'Num_Pickups', 'Num_Dropoffs', 'PWT', 'RWT', 'AWT', 'WT_AbsDev',
                                  'PVT', 'RVT', 'AVT', 'VT_AbsDev', 'T_Dir_Veh', 'T_Dir_Wait'])
    req_keys = list(reqDF.columns)

    od_request_list = list()

    # analyze requests that were requested within the period of study
    for req in model.reqs:
        simulated_reqs += 1
        if req.Tr >= T_WARM_UP and req.Cep <= T_WARM_UP+T_STUDY and not req.DR:
            req_data = [req.id, req.olat, req.olng, req.dlat, req.dlng, req.Cep,
                        req.Tp, req.Td, req.OnD, req.assigned_veh, req.new_occ, req.NP, req.ND,
                        req.fpwt, req.rewt, req.Tp-req.Cep, req.wtdev, req.fpvt, req.revt,
                        req.Td-req.Tp, req.vtdev, req.Ts, req.Rs]
            req_row = {req_keys[i]: req_data[i] for i in range(len(req_keys))}
            reqDF = reqDF.append(req_row, ignore_index=True)

            count_reqs += 1
            count_reqs_ond += 1 if req.OnD else 0
            count_reqs_adv += 0 if req.OnD else 1
            # count as "served" only when the request is complete, i.e. the dropoff time is not -1
            if not np.isclose(req.Td, -1.0):
                count_served += 1
                count_served_ond += 1 if req.OnD else 0
                count_served_adv += 0 if req.OnD else 1
                wait_time_ond += (req.Tp - req.Cep) if req.OnD else 0
                wait_time_adv += 0 if req.OnD else (req.Tp - req.Cep)
                in_veh_time += (req.Td - req.Tp)
                detour_factor += req.D
                # assume transit connection discount is absorbed by transit agency
                tempfare = (fare[0] + fare[1]/60 * req.Ts + fare[2]/1000 * req.Ds) * fare[3]
                if tempfare < fare[5]:
                    benefit += fare[5]
                else:
                    benefit += tempfare

                number_of_occurrences = df_OD_LOS.get_value(req.m_id, 'number_of_occurrences')
                od_request_list.append([req.m_id,req.olng,req.olat,req.dlng,req.dlat,req.D,(req.Tp-req.Cep),req.Ts,req.Ds])

                if number_of_occurrences == 0: # if entries for this OD doesn't exist
                    # update wt detour factor
                    df_OD_LOS.iloc[req.m_id, df_OD_LOS.columns.get_loc('wait_time')] = (req.Tp - req.Cep)
                    df_OD_LOS.iloc[req.m_id, df_OD_LOS.columns.get_loc('summed_wait_time')] += (req.Tp - req.Cep)
                    df_OD_LOS.iloc[req.m_id, df_OD_LOS.columns.get_loc('detour_factor')] = req.D
                    df_OD_LOS.iloc[req.m_id, df_OD_LOS.columns.get_loc('summed_detour_factor')] += req.D
                    # add number of occurrences
                    df_OD_LOS.iloc[req.m_id, df_OD_LOS.columns.get_loc('number_of_occurrences')] += 1
                else: # if entries for this OD exists
                    df_OD_LOS.iloc[req.m_id, df_OD_LOS.columns.get_loc('summed_wait_time')] += (req.Tp - req.Cep)
                    df_OD_LOS.iloc[req.m_id, df_OD_LOS.columns.get_loc('summed_detour_factor')] += req.D
                    # add number of occurrences
                    df_OD_LOS.iloc[req.m_id, df_OD_LOS.columns.get_loc('number_of_occurrences')] += 1
                    #update wt detour factor
                    df_OD_LOS.iloc[req.m_id, df_OD_LOS.columns.get_loc('wait_time')] = df_OD_LOS.get_value(req.m_id, 'summed_wait_time')/df_OD_LOS.get_value(req.m_id, 'number_of_occurrences')
                    df_OD_LOS.iloc[req.m_id, df_OD_LOS.columns.get_loc('detour_factor')] = df_OD_LOS.get_value(req.m_id, 'summed_detour_factor')/df_OD_LOS.get_value(req.m_id, 'number_of_occurrences')
        elif req.DR and req.Cep >= T_WARM_UP and req.Cep <= T_WARM_UP+T_STUDY:
            count_distance_rejs += 1

    df_req = pd.DataFrame(od_request_list,columns=['m_id','origin_lng','origin_lat','destination_lng','destination_lat','detour', 'wait time',"req.Ts_seconds","re.Ds_m"])
    df_req = df_req.merge(df_diffprob, on=['m_id'], how='left')

    ridershipchange_car = df_req['differenceinprob_car'].sum()
    ridershipchange_walk = df_req['differenceinprob_walk'].sum()
    ridershipchange_bike = df_req['differenceinprob_bike'].sum()
    ridershipchange_taxi = df_req['differenceinprob_taxi'].sum()
    ridershipchange_bus = df_req['differenceinprob_bus'].sum()
    ridershipchange_rail = df_req['differenceinprob_rail'].sum()
    ridershipchange_intermodal = df_req['differenceinprob_intermodal'].sum()

    #match m_id to mode choice results id
    #record the delta prob beside each M_ID
    #expantion factor no because we're counting trips served not using LTDS to expand population simulated

    if not count_served == 0:
        in_veh_time /= count_served
        detour_factor /= count_served
        wait_time = (wait_time_ond + wait_time_adv)/count_served
    if not count_served_ond == 0:
        wait_time_ond /= count_served_ond
    if not count_served_adv == 0:
        wait_time_adv /= count_served_adv

    #for all value in dictionary with out a 0 number of entry then set as wait_time and detour_factor
    df_OD_LOS['wait_time'].replace(np.nan,wait_time,inplace=True)
    df_OD_LOS['detour_factor'].replace(np.nan,detour_factor,inplace=True)

    #create copy to ensure original is not affected
    df_OD_LOS_COPY = df_OD_LOS.copy(deep=True)

    # service rate
    service_rate = 0.0
    service_rate_ond = 0.0
    service_rate_adv = 0.0
    if not count_reqs == 0:
        service_rate = 100.0 * count_served / count_reqs
        wait_time_adj = (wait_time * service_rate + 2*MAX_WAIT * (100-service_rate))/100
    if not count_reqs_ond == 0:
        service_rate_ond = 100.0 * count_served_ond / count_reqs_ond
    if not count_reqs_adv == 0:
        service_rate_adv = 100.0 * count_served_adv / count_reqs_adv

    # vehicle performance
    veh_service_dist = 0.0
    veh_service_time = 0.0
    veh_pickup_dist = 0.0
    veh_pickup_time = 0.0
    veh_rebl_dist = 0.0
    veh_rebl_time = 0.0
    veh_load_by_dist = 0.0
    veh_load_by_time = 0.0
    veh_occupancy_events = 0.0
    veh_trips_per_occupancy = 0.0
    cost = 0.0

    for veh in model.vehs:
        veh_service_dist += veh.Ds
        veh_service_time += veh.Ts
        veh_pickup_dist += veh.Dp
        veh_pickup_time += veh.Tp
        veh_rebl_dist += veh.Dr
        veh_rebl_time += veh.Tr
        if not veh.Ds + veh.Dp + veh.Dr == 0:
            veh_load_by_dist += veh.Ld / (veh.Ds + veh.Dp + veh.Dr)
        veh_load_by_time += veh.Lt / T_STUDY
        veh_occupancy_events += veh.oes
        cost += COST_BASE + COST_MIN * T_STUDY/60 + COST_KM/1000 * (veh.Ds + veh.Dp + veh.Dr)
    veh_service_dist /= model.V
    veh_service_time /= model.V
    veh_service_time_percent = 100.0 * veh_service_time / T_STUDY
    veh_pickup_dist /= model.V
    veh_pickup_time /= model.V
    veh_pickup_time_percent = 100.0 * veh_pickup_time / T_STUDY
    veh_rebl_dist /= model.V
    veh_rebl_time /= model.V
    veh_rebl_time_percent = 100.0 * veh_rebl_time / T_STUDY
    veh_load_by_dist /= model.V
    veh_load_by_time /= model.V

    if not count_served == 0:
        veh_trips_per_occupancy = count_served / veh_occupancy_events

    overall_logsum = (logsum_w_AVPT - logsum_wout_AVPT) / 0.144 * 10


    print("*"*80)
    print("scenario: %s, step: %d" % (asc_name, step))
    print("simulation starts at %s, runtime time: %d s" % (datetime.datetime.now().strftime("%Y-%m-%d_%H:%M"), runtime))
    print("system settings:")
    print("  - period of study: %d s, with warm-up %d s, cool-down %d s" % (T_STUDY, T_WARM_UP, T_COOL_DOWN))
    print("  - fleet size: %d; capacity: %d" % (model.V, model.K))
    print("  - demand volume: %.1f trips/h" % (model.D))
    print("  - assignment method: %s, interval: %.1f s" % (MET_ASSIGN, INT_ASSIGN))
    print("  - rebalancing method: %s, interval: %.1f s" % (MET_REBL, INT_REBL))
    print("simulation results:")
    print("  - total simulated requests: {}".format(simulated_reqs))
    print("  - requests:")
    print("    + service rate: %.1f%% (%d/%d), wait time: %.1f s, adjusted: %.1f s" % (service_rate, count_served, count_reqs, wait_time, wait_time_adj))
    print("      - of which on-demand requests: %.1f%% (%d/%d), wait time: %.1f s" % (service_rate_ond, count_served_ond, count_reqs_ond, wait_time_ond))
    print("      - of which in-advance requests: %.1f%% (%d/%d), wait time: %.1f s" % (service_rate_adv, count_served_adv, count_reqs_adv, wait_time_adv))
    print("    + in-vehicle travel time: %.1f s" % (in_veh_time))
    print("    + detour factor: %.2f" % (detour_factor))
    print("    + requests rejected due to min distance threshold: %d" % count_distance_rejs)
    print("  - vehicles:")
    print("    + vehicle service distance travelled: %.1f m" % (veh_service_dist))
    print("    + vehicle service time travelled: %.1f s" % (veh_service_time))
    print("    + vehicle service time percentage: %.1f%%" % (veh_service_time_percent))
    print("    + vehicle pickup distance travelled: %.1f m" % (veh_pickup_dist))
    print("    + vehicle pickup time travelled: %.1f s" % (veh_pickup_time))
    print("    + vehicle pickup time percentage: %.1f%%" % (veh_pickup_time_percent))
    print("    + vehicle rebalancing distance travelled: %.1f m" % (veh_rebl_dist))
    print("    + vehicle rebalancing time travelled: %.1f s" % (veh_rebl_time))
    print("    + vehicle rebalancing time percentage: %.1f%%" % (veh_rebl_time_percent))
    print("    + vehicle average load: %.2f (distance weighted), %.2f (time weighted)" % (veh_load_by_dist, veh_load_by_time))
    print("  - cost-benefit analysis:")
    print("    + cost: %.2f, benefit: %.2f, profit: %.2f" % (cost, benefit, benefit-cost))
    print("*"*80)

    print("got to save results")
    # write and save the result analysis
    # filename1= "output/results_faremultiplier" + str(fare_multiplier) + "_fleetsize_" + str(fleet_size) + ".csv"
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        row = [asc_name, step, MET_ASSIGN, MET_REBL, T_STUDY, model.V, model.K, model.D,
         service_rate, count_served, count_reqs, service_rate_ond, count_served_ond, count_reqs_ond, service_rate_adv, count_served_adv, count_reqs_adv,
         wait_time, wait_time_adj, wait_time_ond, wait_time_adv, in_veh_time, detour_factor, veh_service_dist, veh_service_time, veh_service_time_percent,
         veh_pickup_dist, veh_pickup_time, veh_pickup_time_percent,
         veh_rebl_dist, veh_rebl_time, veh_rebl_time_percent, veh_load_by_dist, veh_load_by_time, veh_occupancy_events, veh_trips_per_occupancy,
         cost, benefit, logsum_w_AVPT, logsum_wout_AVPT, overall_logsum,
         ridershipchange_car,ridershipchange_walk,ridershipchange_bike,ridershipchange_taxi,ridershipchange_bus,ridershipchange_rail,ridershipchange_intermodal,
         simulated_reqs, None]
        writer.writerow(row)
        print("end of save the results- the service rate is: ", service_rate)

    # # write and save data of all requests
    # f = open('output/requests.csv', 'w')
    # writer = csv.writer(f)
    # writer.writerow(["id", "olng", "olat", "dlng", "dlat", "Ts", "OnD", "Tr", "Cep", "Tp", "Td", "WT", "VT", "D"])
    # for req in model.reqs:
    #     if req.Cep >= T_WARM_UP and req.Cep <= T_WARM_UP+T_STUDY:
    #         row = [req.id, req.olng, req.olat, req.dlng, req.dlat, req.Ts, req.OnD, req.Tr, req.Cep, req.Tp, req.Td,
    #          req.Tp-req.Cep if req.Tp >= 0 else -1, req.Td-req.Tp if req.Td >= 0 else -1, req.D]
    #         writer.writerow(row)
    # f.close()

    print(reqDF)
    if requests_filename is not None:
        reqDF.to_csv(requests_filename)

    return wait_time_adj, detour_factor, df_OD_LOS_COPY, reqDF

# animation
def anim(frames):
    def init():
        for i in range(len(vehs)):
            vehs[i].set_data([frames[0][i].lng], [frames[0][i].lat])
            r1x = []
            r1y = []
            r2x = []
            r2y = []
            r3x = []
            r3y = []
            count = 0
            for leg in frames[0][i].route:
                if leg.pod == 0:
                    for step in leg.steps:
                        geo = np.transpose( step.geo )
                        r3x.extend(geo[0])
                        r3y.extend(geo[1])
                    assert len(frames[0][i].route) == 1
                    continue
                count += 1
                if count == 1:
                    for step in leg.steps:
                        geo = np.transpose( step.geo )
                        r1x.extend(geo[0])
                        r1y.extend(geo[1])
                else:
                    for step in leg.steps:
                        geo = np.transpose( step.geo )
                        r2x.extend(geo[0])
                        r2y.extend(geo[1])
            routes1[i].set_data( r1x, r1y )
            routes2[i].set_data( r2x, r2y )
            routes3[i].set_data( r3x, r3y )
        return vehs, routes1, routes2, routes3

    def animate(n):
        for i in range(len(vehs)):
            vehs[i].set_data([frames[n][i].lng], [frames[n][i].lat])
            r1x = []
            r1y = []
            r2x = []
            r2y = []
            r3x = []
            r3y = []
            count = 0
            for leg in frames[n][i].route:
                if leg.pod == 0:
                    for step in leg.steps:
                        geo = np.transpose( step.geo )
                        r3x.extend(geo[0])
                        r3y.extend(geo[1])
                    assert len(frames[n][i].route) == 1
                    continue
                count += 1
                if count == 1:
                    for step in leg.steps:
                        geo = np.transpose( step.geo )
                        r1x.extend(geo[0])
                        r1y.extend(geo[1])
                else:
                    for step in leg.steps:
                        geo = np.transpose( step.geo )
                        r2x.extend(geo[0])
                        r2y.extend(geo[1])
            routes1[i].set_data( r1x, r1y )
            routes2[i].set_data( r2x, r2y )
            routes3[i].set_data( r3x, r3y )
        return vehs, routes1, routes2, routes3

    fig = plt.figure(figsize=(MAP_WIDTH, MAP_HEIGHT))
    plt.xlim((Olng, Dlng))
    plt.ylim((Olat, Dlat))
    img = mpimg.imread("map.png")
    plt.imshow(img, extent=[Olng, Dlng, Olat, Dlat], aspect=(Dlng-Olng)/(Dlat-Olat)*MAP_HEIGHT/MAP_WIDTH)
    fig.subplots_adjust(left=0.00, bottom=0.00, right=1.00, top=1.00)
    vehs = []
    routes1 = []
    routes2 = []
    routes3 = []
    random_color = lambda: random.randint(0, 255)
    for v in reversed(frames[0]):
        if v.id == 0:
            color = "#dc241f"
        elif v.id == 1:
            color = "#9b0058"
        elif v.id == 2:
            color = "#0019a8"
        elif v.id == 3:
            color = "#0098d8"
        elif v.id == 4:
            color = "#b26300"
        else:
            # generate random color for any other vehicles
            color = '#%02X%02X%02X' % (random_color(), random_color(), random_color())
        vehs.append( plt.plot([], [], color=color, marker='o', markersize=4, alpha=0.7)[0] )
        routes1.append( plt.plot([], [], linestyle='-', color=color, alpha=0.7)[0] )
        routes2.append( plt.plot([], [], linestyle='--', color=color, alpha=0.7)[0] )
        routes3.append( plt.plot([], [], linestyle=':', color=color, alpha=0.4)[0] )
    anime = animation.FuncAnimation(fig, animate, init_func=init, frames=len(frames), interval=100)
    return anime