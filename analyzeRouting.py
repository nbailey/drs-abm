import argparse
import csv

import igraph as ig
import numpy as np
import pandas as pd

import igraph as ig
import numpy as np
import pandas as pd

from lib.RoutingEngine import *
from optimizationTesting import parseReqsAndVehs, analyzeOptResults


METRIC_NAMES = ["CRVI", "Certain_Delayed_Trip_Count", "Raw_Delay", "Bounded_Delay", "Late_Arrival_Pct"]

def pathsFromRouting(routing, G, vehs):
	# Create dict of requests and the edge paths needed to reach their origin and destination
	reqPaths = dict()

	for vid in routing.keys():
		try:
			veh = vehs[vid]
		except IndexError:
			veh = None
			for v in vehs:
				if v.id == vid:
					veh = v
			assert veh is not None

		currPath = list()

		olng = veh.lng
		olat = veh.lat

		for rid, pod, tlng, tlat in routing[vid]:
			onodename = str((olng, olat))
			dnodename = str((tlng, tlat))
			legPath = G.get_shortest_paths(onodename, to=dnodename, weights='ttmedian', output='epath')[0]

			for eid in legPath:
				currPath.append(eid)

			reqPaths[(rid, pod)] = currPath.copy()

			olng = tlng
			olat = tlat

	return reqPaths


def arrivalTimesFromPaths(paths, edge_time_table, reqs):
	reqArrivalTimes = dict()
	POD_NAMES = {1: "pickup", -1: "dropoff"}

	for rid, pod in paths.keys():
		try:
			req = reqs[rid]
		except IndexError:
			req = None
			for r in reqs:
				if r.id == rid:
					req = r
			assert req is not None


		edgePath = paths[(rid, pod)]
		arrivalTimes = list()
		arrivalTimes.append(rid)
		if pod == 1:
			arrivalTimes.append(req.Clp)
		else:
			arrivalTimes.append(req.Cld)

		for timeTable in edge_time_table:
			t = 0

			for eid in edgePath:
				t += timeTable[eid]

			arrivalTimes.append(t)

		reqArrivalTimes["req_{}_{}".format(rid, POD_NAMES[pod])] = arrivalTimes

	colnames = ["rid", "Cl",] + list(range(len(edge_time_table)))

	arrivalTimeData = pd.DataFrame.from_dict(reqArrivalTimes)
	arrivalTimeData = arrivalTimeData.T

	arrivalTimeData.columns = colnames

	return arrivalTimeData


def calcRiskEquivalent(alpha, T):
    cvi = alpha * np.log((1/len(T))*np.sum([np.exp(T/alpha)]))
    return cvi


def calcCVI(T, l):
    alpha_ub = 2e6
    alpha_lb = 1e-2
    alpha = (alpha_ub + alpha_lb) / 2
    iteration = 0

    if np.all(T < l):
        return 0, True

    # If alpha >= 1e6, assume that constraint cannot be satisfied
    risk_equivalent = calcRiskEquivalent(alpha, T)
    if risk_equivalent > l:
        return np.float("inf"), False
    # Otherwise start bifurcative search
    else:
        while np.abs(risk_equivalent - l) >= 1e-5 and iteration <= 100:
            if risk_equivalent < l:
                alpha_ub = (alpha_ub + alpha) / 2
            else:
                alpha_lb = (alpha + alpha_lb) / 2

            alpha = (alpha_ub + alpha_lb) / 2
            risk_equivalent = calcRiskEquivalent(alpha, T)
            iteration += 1

    return alpha, True


def computeDelaysMetrics(routing, G, vehs, reqs, edge_time_table):
	req_paths = pathsFromRouting(routing, G, vehs)
	arrival_times = arrivalTimesFromPaths(req_paths, edge_time_table, reqs)

	scenarios = list(arrival_times.columns[2:])

	CVIs = list()
	CDTs = list()
	raw_delays = list()
	bounded_delays = list()
	late_arrivals = list()

	for req in list(arrival_times.index):	
		rid = arrival_times.loc[req, "rid"]
		threshold = arrival_times.loc[req, "Cl"]/60
		T = arrival_times.loc[req, scenarios]/60

		cvi = calcCVI(T, threshold)
		# Use the CVI returned if a value was found (returns (cvi_value, True)). Otherwise use 0, as the function
		# will return (inf, False). Each infeasible trip / rejection adds 1 inf to the total CRVI.
		if cvi[1]:
			CVIs.append(cvi[0])
			CDTs.append(0)
		else:
			CVIs.append(0)
			CDTs.append(1)
		raw_delays.append(np.mean(T - threshold))
		bounded_delays.append(np.mean(np.maximum(T - threshold, np.zeros(len(T)))))
		late_arrivals.append(np.sum(T > threshold) / len(T))

	cvi_sum = np.sum(CVIs)
	cdt_sum = np.sum(CDTs)
	raw_delay_mean = np.mean(raw_delays)
	bnd_delay_mean = np.mean(bounded_delays)
	exp_late_arrivals = np.sum(late_arrivals)

	metrics = (cvi_sum, cdt_sum, raw_delay_mean, bnd_delay_mean, exp_late_arrivals)

	return metrics, arrival_times
	# return np.sum(CVIs), np.mean(raw_delays), np.mean(bounded_delays), np.sum(late_arrivals)


def evaluateParetoFrontier(G, vehs, reqs, T, scenario_mapping, all_assignments_df):
	scenario_veh_map = scenario_mapping["vehs"]
	scenario_req_map = scenario_mapping["reqs"]
	inverse_req_map = {scenario_req_map[rid]: rid for rid in scenario_req_map.keys()}

	assignment_data = list()

	for _, assignment in all_assignments_df.iterrows():
		assignment_id = assignment["ID"]
		routes = eval(assignment["route_dict"])

		# print("    Assignment {}:".format(assignment_id))
		# for vid in routes.keys():
		# 	print("    - {} | {}".format(vid, " -> ".join([str(x) for x in routes[vid]])))

		unmapped_routes = dict()

		for vid in scenario_veh_map.keys():
			scenario_vid = scenario_veh_map[vid]

			unmapped_routes[vid] = list()

			if scenario_vid in routes.keys():
				route = routes[scenario_vid]

				for scenario_rid, pod in route:
					rid = inverse_req_map[scenario_rid]
					req = reqs[rid]

					if pod == 1:
						tlng = req.olng
						tlat = req.olat
					elif pod == -1:
						tlng = req.dlng
						tlat = req.dlat

					unmapped_routes[vid].append((rid, pod, tlng, tlat))

		# print("    Converted assignment:")
		# for vid in unmapped_routes.keys():
		# 	print("    - {} | {}".format(vid, " -> ".join([str(x) for x in unmapped_routes[vid]])))

		metrics, arrival_times = computeDelaysMetrics(unmapped_routes, G, vehs, reqs, T)
		# print("    Assignment metrics:")
		# for i, metric in enumerate(METRIC_NAMES):
		# 	print(    "    - {}: {:.2f}".format(metric, metrics[i]))

		# Bounded delays is the 4th metric (index 3) and late arrivals are the 5th metric (index 4)
		assignment_data.append({
				"id": assignment_id,
				"routes": str(routes),
				"avg_delay": metrics[3],
				"late_arrivals": metrics[4],
			})

	assignment_df = pd.DataFrame(assignment_data)

	all_assignment_pts = np.array(assignment_df[["avg_delay", "late_arrivals"]])
	frontier_pts = keep_efficient(all_assignment_pts)
	frontier_routings = list()

	for pt in frontier_pts:
		print(pt)
		pt_df = assignment_df[np.isclose(assignment_df["avg_delay"], pt[0]) & np.isclose(assignment_df["late_arrivals"], pt[1])]
		print(pt_df)
		for _, row in pt_df.iterrows():
			frontier_routings.append(row["routes"])

	return frontier_routings, frontier_pts


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
        if i >= n:
            break
        # find all points not dominated by i
        # since points are sorted by coordinate sum
        # i cannot dominate any points in 1,...,i-1
        undominated[i+1:n] = (pts[i+1:] < pts[i]).any(1)
        # keep points undominated so far
        pts = pts[undominated[:n]]
    return pts


def main(title, graph, speeds, requests, vehicles, scenario_file, routing, scenario_metadata_dict=None):

	G = ig.Graph.Read_GML(graph)
	router = RoutingEngine(graph=G, cst_speed=6)

	R = pd.read_csv(requests)
	V = pd.read_csv(vehicles)

	routing_df = pd.read_csv(routing)

	S_val = np.array(pd.read_csv(speeds, index_col=0))
	T_val = G.es["length"] / S_val

	for idx, edge_speeds in enumerate(T_val.T):
		edge = G.es[idx]
		edge["ttmedian"] = np.median(edge_speeds)

	reqs, vehs = parseReqsAndVehs(router, R, V)

	scenarios = pd.read_csv(scenario_file, index_col=0)
	
	num_scenarios = len(scenarios)
	scenario_reverse_map = dict()
	scenario_frontiers = dict()

	req_cols = [col for col in scenarios if col.startswith('req')]
	veh_cols = [col for col in scenarios if col.startswith('veh')]
	
	for scenario_id, row in scenarios.iterrows():
		scenario_rids = list(row[req_cols])
		scenario_vids = list(row[veh_cols])

		scenario_reverse_map[scenario_id] = {"reqs": dict(), "vehs": dict()}

		for i in range(len(scenario_rids)):
			scenario_reverse_map[scenario_id]["reqs"][scenario_rids[i]] = i
		for j in range(len(scenario_vids)):
			scenario_reverse_map[scenario_id]["vehs"][scenario_vids[j]] = j

		# if evaluate_all_routes and all_assignments_path is not None:
		# 	print("Evaluating the Pareto frontier for scenario {} in {}...".format(scenario_id, title))
		# 	all_assignments_df = pd.read_csv(all_assignments_path)

		# 	print("  Scenario reverse map:")
		# 	print(scenario_reverse_map[scenario_id])

		# 	t_start = time.time()
		# 	frontier_assignments, frontier_pts = evaluateParetoFrontier(G, vehs, reqs, T_val, scenario_reverse_map[scenario_id], all_assignments_df)
		# 	t_frontier = time.time() - t_start

		# 	print("  Frontier calculated in {:.1f} seconds".format(t_frontier))

		# 	print("  The assignments that form the Pareto frontier are:")
		# 	for i, assignment in enumerate(frontier_assignments):
		# 		# print("  {}.".format(i))
		# 		print("  {}. {}".format(i, assignment))
		# 		# routing = eval(assignment)
		# 		# for vid in routing.keys():
		# 		# 	print("    - {} | {}".format(vid, " -> ".join([str(x) for x in routing[vid]])))

		# 	print(frontier_assignments)

		# 	scenario_frontiers[scenario_id] = frontier_assignments


	metrics_outpath = "output/{}-validation-metrics.csv".format(title)
	metric_headers = ["Experiment", "Scenario_ID", "Method_Name", "Num_Samples", "Iter"]
	for metric in METRIC_NAMES:
		metric_headers.append(metric)

	if len(scenario_metadata_dict) > 0:
		metric_headers.append("On_Pareto_Frontier")
		metric_headers.append("Avg_Min_VO_Dist")
		metric_headers.append("Avg_OD_Dist")

	metric_headers.append("Soln_Time")
	with open(metrics_outpath, "w", newline="") as outcsv:
		writer = csv.writer(outcsv)
		writer.writerow(metric_headers)

	arrival_times_outpath = "output/{}-validation-arrival-times.csv".format(title)
	arrival_times_headers = ["Experiment", "Scenario_ID", "Method_Name", "Num_Samples", "Iter", "LocName", "ReqID", "Arrival_Time", "Delay", "Bounded_Delay", "Late_Arrival", "On_Frontier"]
	with open(arrival_times_outpath, "w", newline="") as outcsv:
		writer = csv.writer(outcsv)
		writer.writerow(arrival_times_headers)

	for _, row in routing_df.iterrows():
		print("\nScenario {} - {}, N={} (Iter {})".format(row["Scenario_ID"], row["Method_Name"], row["Num_Samples"], row["Iter"]))

		scenario_id = row["Scenario_ID"]

		reverse_map = scenario_reverse_map[scenario_id]

		# print("Optimal Route:")

		routes = dict()
		for vid in range(len(reverse_map["vehs"])):
			veh_route = eval(row["Veh_{}_Reqs".format(vid)])
			routes[vid] = veh_route
			# print("  {} | {}".format(vid, " -> ".join([str((rid, pod)) for rid, pod, tlng, tlat in veh_route])))

		metadata = dict()

		# if scenario_id in scenario_frontiers.keys():
		# 	frontier_assignments = scenario_frontiers[scenario_id]
		# 	if routes in frontier_assignments:
		# 		metadata["On_Frontier"] = True
		# 	else:
		# 		metadata["On_Frontier"] = False

		if len(scenario_metadata_dict) > 0:
			scenario_metadata = scenario_metadata_dict[scenario_id]

			frontier_assignments = scenario_metadata["efficient_assignments"]

			# print("Pareto Frontier Routes:")
			# for i, assignment in enumerate(frontier_assignments):
			# 	frontier_route = eval(assignment)
			# 	print("{}.".format(i+1))
			# 	for vid in frontier_route.keys():
			# 		print("  {} | {}".format(vid, " -> ".join([str(x) for x in frontier_route[vid]])))

			avg_min_vo_dist = scenario_metadata["avg_min_vo_dist"]
			avg_od_dist = scenario_metadata["avg_od_dist"]

			# print("\n".join(frontier_assignments))
			simplified_route = str({vid: [(rid, pod) for rid, pod, tlng, tlat in routes[vid]] for vid in routes.keys()})
			# print(simplified_route)
			# print(simplified_route in frontier_assignments)

			metadata = {
				"On_Frontier": simplified_route in frontier_assignments,
				"Avg_Min_VO_Dist:": avg_min_vo_dist,
				"Avg_OD_Dist": avg_od_dist,
			}

			print(metadata)

		results = analyzeOptResults(routes, metadata, G, vehs, reqs, T_val, reverse_map, title, scenario_id, row["Method_Name"], row["Num_Samples"], row["Iter"],
									routing_outpath=None, metrics_outpath=metrics_outpath, arrival_times_outpath=arrival_times_outpath)


if __name__ == "__main__":
	import sys

	parser = argparse.ArgumentParser(
		description="Analyze the routings generated from a previous optimization experiment using a validation set.")
	parser.add_argument("-T", "--title", help="Name of experimental runs", type=str, required=True)
	parser.add_argument("-G", "--graph", help="Filepath to road graph network used", type=str, required=True)
	parser.add_argument("-S", "--speeds", help="Filepath to link speed draws (validation set)", type=str, required=True)
	parser.add_argument("-R", "--requests", help="Filepath to pre-generated request info", type=str, required=True)
	parser.add_argument("-V", "--vehicles", help="Filepath to pre-generated vehicle info", type=str, required=True)
	parser.add_argument("-C", "--scenarios", help="Filepath to scenario info from experiments", type=str, required=True)
	parser.add_argument("-O", "--routing", help="Filepath to routing output from experiments", type=str, required=True)
	
	args = parser.parse_args(sys.argv[1:])

	main(args.title, args.graph, args.speeds, args.requests, args.vehicles, args.scenarios, args.routing)