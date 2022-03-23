import argparse
import os
import sys

import igraph as ig
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# Keep the pareto efficient set of points from an array of points given
# https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python/32793059#32793059
def keep_efficient(pts):
    # sort points by decreasing sum of coordinates
    pts = pts[pts.sum(1).argsort()[::1]]
    # initialize a boolean mask for undominated points
    # to avoid creating copies each iteration
    undominated = np.ones(pts.shape[0], dtype=bool)
    for i in range(pts.shape[0]):
        # process each point in turn
        n = pts.shape[0]
        undominated[i] = 1
        if i >= n:
            break
        # find all points not dominated by i
        # since points are sorted by coordinate sum
        # i cannot dominate any points in 1,...,i-1
        undominated[i+1:n] = ~(pts[i+1:] >= pts[i]).all(1)
        # keep points undominated so far
        pts = pts[undominated[:n]]
    return pts[pts[:, 0].argsort()[::1]]

def generateMetadata(graph_path, requests_path, vehicles_path, speeds_path, scenario, req_cols, veh_cols, all_routings_path):
    G = ig.Graph.Read_GML(graph_path)
    requests = pd.read_csv(requests_path)
    vehicles = pd.read_csv(vehicles_path)

    speeds = np.array(pd.read_csv(speeds_path, index_col=0))

    all_possible_routes = pd.read_csv(all_routings_path, index_col=0)

    T = np.array(G.es["length"] / speeds)
    G.es["ttmedian"] = np.median(T, axis=0)

    reqs = dict()
    vehs = dict()

    scenario_rids = list(scenario[req_cols])
    scenario_vids = list(scenario[veh_cols])

    # Create a dict of vehicle information for the simulated scenario
    for v in range(len(scenario_vids)):
        vid = scenario_vids[v]

        vnodename = "({}, {})".format(vehicles["vlng"][vid], vehicles["vlat"][vid])
        vnode = G.vs.find(vnodename)

        vehs[v] = {
            "vid": vehicles["vid"][vid],
            "lng": vehicles["vlng"][vid],
            "lat": vehicles["vlat"][vid],
            "vnode": vnode
        }

    # Create a dict of request information for the simulated scenario
    for r in range(len(scenario_rids)):
        rid = scenario_rids[r]

        onodename = "({}, {})".format(requests["olng"][rid], requests["olat"][rid])
        dnodename = "({}, {})".format(requests["dlng"][rid], requests["dlat"][rid])

        onode = G.vs.find(onodename)
        dnode = G.vs.find(dnodename)

        reqs[r] = {
            "rid": requests["rid"][rid],
            "olng": requests["olng"][rid],
            "olat": requests["olat"][rid],
            "dlng": requests["dlng"][rid],
            "dlat": requests["dlat"][rid],
            "Clp": requests["Clp"][rid],
            "Cld": requests["Cld"][rid],
            "onode": onode,
            "dnode": dnode,
        }

        min_vo_dist = np.inf

        for v in vehs.keys():
            veh = vehs[v]
            vnode = veh["vnode"]

            vo_dist = G.shortest_paths(source=vnode, target=onode, weights="length")[0][0]

            # print(vo_dist)

            if vo_dist < min_vo_dist:
                min_vo_dist = vo_dist

        reqs[r]["min_vo_dist"] = min_vo_dist
        reqs[r]["od_dist"] = G.shortest_paths(source=onode, target=dnode, weights="length")[0][0]

    avg_min_vo_dist = np.mean([reqs[r]["min_vo_dist"] for r in reqs.keys()])
    avg_od_dist = np.mean([reqs[r]["od_dist"] for r in reqs.keys()])

    route_data = list()

    for _, route in all_possible_routes.iterrows():
        route_id = route["ID"]
        route_dict = eval(route["route_dict"])

        average_delay = 0
        late_arrivals = 0
        total_distance = 0
        individual_delays = list()

        for v in route_dict.keys():
            veh = vehs[v]
            vnode = veh["vnode"]

            veh_route = route_dict[v]

            route_tt = np.zeros(len(T))
            route_dist = 0
            curr_loc = vnode

            for r, pod in veh_route:
                req = reqs[r]

                rnode = None
                Clr = 0

                if pod == 1:
                    rnode = req["onode"]
                    Clr = req["Clp"]
                elif pod == -1:
                    rnode = req["dnode"]
                    Clr = req["Cld"]

                route_path = G.get_shortest_paths(curr_loc, to=rnode, weights="ttmedian", output="epath")[0]

                for eid in route_path:
                    route_tt += T[:, eid]
                    route_dist += G.es[eid]["length"]

                # print(np.mean(route_tt))
                # print(Clr)
                # print("Delay = {:.2f}".format(np.mean(np.maximum(route_tt - Clr, np.zeros(len(T))))))
                # print("Late Arrival % = {:.1f}".format(100*np.mean(route_tt > Clr)))

                delay = np.maximum(route_tt - Clr, np.zeros(len(T)))

                for t in delay:
                    individual_delays.append(t)

                average_delay += np.mean(delay)
                late_arrivals += np.mean(route_tt > Clr)

                curr_loc = rnode

            total_distance += route_dist

        route_avg_delay = np.round(average_delay/(60*2*len(reqs)), 3)

        route_data.append({
            "route_id": route_id,
            "route": str(route_dict),
            "average_delay": route_avg_delay,
            "late_arrival_pct": np.round(100*late_arrivals/(2*len(reqs)), 3),
            "total_distance": total_distance,
            "delay_variance": np.round(np.mean(np.power(np.array(individual_delays)/60 - route_avg_delay, 2))/(2*len(reqs)), 3)
        })

    routing_df = pd.DataFrame(route_data)

    frontiers = {
        "all": ["average_delay", "late_arrival_pct", "total_distance", "delay_variance"],
        "no_var": ["average_delay", "late_arrival_pct", "total_distance"],
        "no_dist": ["average_delay", "late_arrival_pct", "delay_variance"],
        "no_rate": ["average_delay", "total_distance", "delay_variance"],
        "no_delay": ["late_arrival_pct", "total_distance", "delay_variance"]
    }

    scenario_metadata = {
        "frontiers": frontiers,
        "avg_min_vo_dist": avg_min_vo_dist,
        "avg_od_dist": avg_od_dist,
    }

    for frontier in frontiers.keys():
        columns = frontiers[frontier]
        all_routing_pts = np.array(routing_df[columns])
        pareto_pts = keep_efficient(all_routing_pts)
        efficient_routing_pts = pareto_pts[pareto_pts[:,0].argsort()[::1]]
        
        efficient_assignments = list()
        efficient_pts = list()

        for i, pt in enumerate(efficient_routing_pts):
            average_delay = pt[0]
            late_arrival_pct = pt[1]

            other_indices = list(set(range(len(efficient_routing_pts))) - {i,})
            # print(other_indices)
            # if len(other_indices) > 0:
            #     print(efficient_routing_pts[other_indices])
            #     print(efficient_routing_pts[other_indices] <= pt)
            #     print((efficient_routing_pts[other_indices] <= pt).all(1))
            #     print((efficient_routing_pts[other_indices] <= pt).all(1).any())

            if len(other_indices) == 0 or not (efficient_routing_pts[other_indices] <= pt).all(1).any():
                pt_routings = routing_df[np.isclose(routing_df["average_delay"], average_delay) & (np.isclose(routing_df["late_arrival_pct"], late_arrival_pct))]
                for _, row in pt_routings.iterrows():
                    efficient_assignments.append(row["route"])
                    efficient_pts.append((average_delay, late_arrival_pct))

        scenario_metadata["{}_efficient_assignments".format(frontier)] = efficient_assignments
        scenario_metadata["{}_efficient_pts".format(frontier)] = efficient_pts

    return scenario_metadata


def main(args):
    experiment_log = args[1]
    experiment_dir, _ = os.path.split(experiment_log)

    X = pd.read_csv(experiment_log)

    for _, row in X.iterrows():
        title = row["Experiment_Name"]
        G = ig.Graph.Read_GML(row["Graph_File"])
        requests = pd.read_csv(row["Input_Requests"])
        vehicles = pd.read_csv(row["Input_Vehicles"])
        scenarios = pd.read_csv(experiment_dir + "/" + title + "-scenarios.csv", index_col=0)
        routings = pd.read_csv(experiment_dir + "/" + title + "-routing.csv", index_col=0)
        metrics = pd.read_csv(experiment_dir + "/" + title + "-Processed-validation-metrics.csv", index_col=0)
        speeds = np.array(pd.read_csv(row["Input_Speeds"].replace("-speeds.csv", "-val-speeds.csv"), index_col=0))
        NV = row["Num_Vehs"]
        NR = row["Num_Reqs"]

        if NV == 3 and NR == 5:
            all_possible_routes = pd.read_csv("all_3v5r_routings.csv")
            enumerate_routes = True
        else:
            enumerate_routes = False

        T = np.array(G.es["length"] / speeds)
        G.es["ttmedian"] = np.median(T, axis=0)

        # fig, ax = plt.subplots()

        for i, scenario in scenarios.iterrows():
            print("Scenario {}".format(i))
            # print(scenario)

            reqs = dict()
            vehs = dict()

            # Create a dict of vehicle information for the simulated scenario
            for v in range(NV):
                vid = scenario["veh_{}".format(v)]

                vnodename = "({}, {})".format(vehicles["vlng"][vid], vehicles["vlat"][vid])
                vnode = G.vs.find(vnodename)

                vehs[v] = {
                    "vid": vehicles["vid"][vid],
                    "lng": vehicles["vlng"][vid],
                    "lat": vehicles["vlat"][vid],
                    "vnode": vnode
                }

            # Create a dict of request information for the simulated scenario
            for r in range(NR):
                rid = scenario["req_{}".format(r)]

                onodename = "({}, {})".format(requests["olng"][rid], requests["olat"][rid])
                dnodename = "({}, {})".format(requests["dlng"][rid], requests["dlat"][rid])

                onode = G.vs.find(onodename)
                dnode = G.vs.find(dnodename)

                reqs[r] = {
                    "rid": requests["rid"][rid],
                    "olng": requests["olng"][rid],
                    "olat": requests["olat"][rid],
                    "dlng": requests["dlng"][rid],
                    "dlat": requests["dlat"][rid],
                    "Clp": requests["Clp"][rid],
                    "Cld": requests["Cld"][rid],
                    "onode": onode,
                    "dnode": dnode,
                }

                min_vo_dist = np.inf

                for v in vehs.keys():
                    veh = vehs[v]
                    vnode = veh["vnode"]

                    vo_dist = G.shortest_paths(source=vnode, target=onode, weights="length")[0][0]

                    # print(vo_dist)

                    if vo_dist < min_vo_dist:
                        min_vo_dist = vo_dist

                reqs[r]["min_vo_dist"] = min_vo_dist

                reqs[r]["od_dist"] = G.shortest_paths(source=onode, target=dnode, weights="length")[0][0]

            avg_min_vo_dist = np.mean([reqs[r]["min_vo_dist"] for r in reqs.keys()])
            avg_od_dist = np.mean([reqs[r]["od_dist"] for r in reqs.keys()])

            print("Average distance between request origin and closest vehicle: {:.2f}".format(avg_min_vo_dist))
            print("Average distance between request origin and destination: {:.2f}".format(avg_od_dist))

            if enumerate_routes:
                route_data = list()

                for _, route in all_possible_routes.iterrows():
                    route_id = route["ID"]
                    route_dict = eval(route["route_dict"])

                    average_delay = 0
                    late_arrivals = 0

                    for v in route_dict.keys():
                        veh = vehs[v]
                        vnode = veh["vnode"]

                        veh_route = route_dict[v]

                        route_tt = np.zeros(len(T))
                        curr_loc = vnode

                        for r, pod in veh_route:
                            req = reqs[r]

                            rnode = None
                            Clr = 0

                            if pod == 1:
                                rnode = req["onode"]
                                Clr = req["Clp"]
                            elif pod == -1:
                                rnode = req["dnode"]
                                Clr = req["Cld"]

                            route_path = G.get_shortest_paths(curr_loc, to=rnode, weights="ttmedian", output="epath")[0]

                            for eid in route_path:
                                route_tt += T[:, eid]

                            # print(np.mean(route_tt))
                            # print(Clr)
                            # print("Delay = {:.2f}".format(np.mean(np.maximum(route_tt - Clr, np.zeros(len(T))))))
                            # print("Late Arrival % = {:.1f}".format(100*np.mean(route_tt > Clr)))

                            average_delay += np.mean(np.maximum(route_tt - Clr, np.zeros(len(T))))
                            late_arrivals += np.mean(route_tt > Clr)

                            curr_loc = rnode

                    route_data.append({
                        "scenario": scenario["scenario_id"],
                        "route_id": route_id,
                        "average_delay": average_delay/(60*2*NR),
                        "late_arrival_pct": 100*late_arrivals/(2*NR)
                    })

                    # print(route_dict)
                    # print("Average Delay: {:.2f} min. | Late Arrival Pct: {:.1f}%".format(average_delay/60, 100*late_arrivals/(2*NR)))

                routing_df = pd.DataFrame(route_data)

                all_routing_pts = np.around(np.array(routing_df[["average_delay", "late_arrival_pct"]]), 3)
                all_routing_pts.round()

                pareto_pts = keep_efficient(all_routing_pts)
                efficient_routing_pts = pareto_pts[pareto_pts[:,0].argsort()[::1]]

                method_pts = dict()
                for method, N in metrics.groupby(["Method_Name", "Num_Samples"]).groups:
                    data = metrics[(metrics["Scenario_ID"]==i) & (metrics["Method_Name"]==method) & (metrics["Num_Samples"]==N)]
                    pts = np.around(np.array(data[["Bounded_Delay", "Late_Arrival_Pct"]]), 3)

                    pts[:, 1] = 100 * (pts[:, 1] / (2*NR))
                    method_pts[method, N] = pts

                # method_pts = {
                #     (method, N): np.array(metrics[(metrics["Scenario_ID"]==scenario) & (metrics["Method_Name"]==method) & (metrics["Num_Samples"]==N)][["Bounded_Delay", "Late_Arrival_Pct"]])
                #     for (method, N) in metrics.groupby(["Method_Name", "Num_Samples"]).groups
                # }

                plt.plot(efficient_routing_pts[:, 0], efficient_routing_pts[:, 1], "--", label="Scenario {}".format(i), alpha=0.7)

                for method, N in method_pts.keys():
                    pts = method_pts[(method, N)]
                    if method == "Alonso-Mora":
                        method_label = "Benchmark"
                        s = 200
                    else:
                        method_label = method + "(N={})".format(N)
                        s = 50
                    plt.scatter(pts[:, 0], pts[:, 1], label=method_label, alpha=0.3, s=s)

                plt.scatter(all_routing_pts[:, 0], all_routing_pts[:, 1], c='k', s=20, alpha=0.5, label="All Feasible Solutions")

                plt.grid()
                plt.legend()
                plt.show()

                routing_df.to_csv("{}-routing-analysis.csv".format(title), header=(i==0), index=False)

        # plt.grid()
        # plt.legend()
        # plt.show()

if __name__ == "__main__":
    main(sys.argv)