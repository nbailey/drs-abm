import argparse
import copy
import csv
import igraph as ig
import numpy as np
import pandas as pd
import time

from lib.Agents import *
from lib.Optimization import *
from lib.RoutingEngine import *

from scripts.analyzeRouting import METRIC_NAMES, computeDelaysMetrics


def parseReqsAndVehs(router, reqDF, vehDF, K=4):
	reqs = list()
	vehs = list()

	for _, row in vehDF.iterrows():
		vid = row[0]
		vlng = row[1]
		vlat = row[2]
		
		veh = Veh(int(vid), None, ini_loc=(vlng, vlat), K=K)
		vehs.append(veh)

	for _, row in reqDF.iterrows():
		rid = row[0]
		olng = row[1]
		olat = row[2]
		dlng = row[3]
		dlat = row[4]
		Cep = row[5]
		Clp = row[6]
		Ced = row[7]
		Cld = row[8]
		constraint_dict = {"Cep": Cep, "Clp": Clp, "Ced": Ced, "Cld": Cld}

		req = Req(router, int(rid), 0, olng, olat, dlng, dlat, constraint_dict)
		reqs.append(req)

	return reqs, vehs


def analyzeOptResults(routes, G, vehs, reqs, T, scenario_mapping, solnTime, title, scenario_id, method, iteration, routing_outpath="", metrics_outpath="", arrival_times_outpath=None):
	if routing_outpath is not None and len(routing_outpath) == 0:
		routing_outpath = "output/{}-routing.csv".format(title)
	if metrics_outpath is not None and len(metrics_outpath) == 0:
		metrics_outpath = "output/{}-metrics.csv".format(title)
	if arrival_times_outpath is not None and len(arrival_times_outpath) == 0:
		arrival_times_outpath = "output/{}-arrival-times.csv".format(title)

	out_routing = dict()
	unmapped_routes = dict()

	routing_row = [title, scenario_id, method, iteration]
	metrics_row = [title, scenario_id, method, iteration]

	scenario_veh_map = scenario_mapping["vehs"]
	scenario_req_map = scenario_mapping["reqs"]
	inverse_req_map = {scenario_req_map[rid]: rid for rid in scenario_req_map.keys()}

	if len(routes) > 0:
		for vid in scenario_veh_map.keys():
			scenario_vid = scenario_veh_map[vid]

			unmapped_routes[vid] = list()
			out_routing[scenario_vid] = list()

			if scenario_vid in routes.keys():
				route = routes[scenario_vid]

				for rid, pod, lng, lat in route:
					out_routing[scenario_vid].append((rid, pod, lng, lat))

					unmapped_routes[vid].append((inverse_req_map[rid], pod, lng, lat))

		for veh_idx in range(len(scenario_mapping["vehs"].keys())):
			routing_row.append(out_routing[veh_idx])

		metrics, arrival_times = computeDelaysMetrics(unmapped_routes, G, vehs, reqs, T)

		print(unmapped_routes)
	else:
		metrics = [None]*len(METRIC_NAMES)
		arrival_times = None

	if routing_outpath is not None:
		with open(routing_outpath, "a+", newline="") as outcsv:
			writer = csv.writer(outcsv)
			writer.writerow(routing_row)

	if metrics_outpath is not None:
		for metric in metrics:
			metrics_row.append(metric)

		metrics_row.append(solnTime)

		with open(metrics_outpath, "a+", newline="") as outcsv:
			writer = csv.writer(outcsv)
			writer.writerow(metrics_row)

	if arrival_times_outpath is not None:
		arrival_times_data = list()
		scenarios = list(arrival_times.columns[2:])
		for req in list(arrival_times.index):
			rid = arrival_times.loc[req, "rid"]
			threshold = arrival_times.loc[req, "Cl"]/60
			for scenario in scenarios:
				T = arrival_times.loc[req, scenario]/60
				arrival_times_row = [title, scenario_id, method, iteration, req, rid, T, T-threshold, np.max([T-threshold, 0]), T > threshold]
				arrival_times_data.append(arrival_times_row)

		arrival_times_output = pd.DataFrame(arrival_times_data)
		arrival_times_output.to_csv(arrival_times_outpath, mode="a", header=False, index=False)

	return out_routing, metrics


def main():
	parser = argparse.ArgumentParser(
		description="Run optimization experiments")
	parser.add_argument("-T", "--title", help="Name of experimental runs", type=str, required=True)
	parser.add_argument("-G", "--graph", help="Filepath to road graph network used", type=str, required=True)
	parser.add_argument("-ST", "--speeds", help="Filepath to link speed draws (testing set)", type=str, required=True)
	parser.add_argument("-SV", "--validation", help="Filepath to link speed draws (validation set)", type=str, required=False)
	parser.add_argument("-R", "--requests", help="Filepath to pre-generated request info", type=str, required=True)
	parser.add_argument("-V", "--vehicles", help="Filepath to pre-generated vehicle info", type=str, required=True)
	parser.add_argument("-C", "--scenarios", help="Scenarios to evaluate: filepath or integer", type=str, required=True)
	parser.add_argument("-NR", "--reqsize", help="Number of requests to include in each scenario", type=int, required=True)
	parser.add_argument("-NV", "--vehsize", help="Number of vehicles to include in each scenario", type=int, required=True)
	parser.add_argument("-NS", "--samples", help="Number of speed samples to draw for stocahstic optimizations", nargs="+", type=int, required=True)
	parser.add_argument("-NE", "--evaluations", help="Number of evaluations to run for each scenario", type=int, required=True)
	parser.add_argument("--seed", help="Random seed used", type=int)

	args = parser.parse_args()

	G = ig.Graph.Read_GML(args.graph)
	router = RoutingEngine(graph=G, cst_speed=6)

	R = pd.read_csv(args.requests)
	V = pd.read_csv(args.vehicles)

	S_test = np.array(pd.read_csv(args.speeds, index_col=0))
	T_test = G.es["length"] / S_test

	for idx, edge_speeds in enumerate(T_test.T):
		edge = G.es[idx]
		edge["ttmedian"] = np.median(edge_speeds)

	if args.validation is not None:
		S_val = np.array(pd.read_csv(args.speeds, index_col=0))
		T_val = G.es["length"] / S_val
	else:
		T_val = T_test

	if args.seed is not None:
		rs = np.random.RandomState(args.seed)
	else:
		rs = np.random.RandomState()

	reqs, vehs = parseReqsAndVehs(router, R, V, K=2)

	routing_outpath = "output/{}-routing.csv".format(args.title)
	routing_headers = ["Experiment", "Scenario_ID", "Method", "Iter",]
	for i in range(args.vehsize):
		routing_headers.append("Veh_{}_Reqs".format(i))
	with open(routing_outpath, "w", newline="") as outcsv:
		writer = csv.writer(outcsv)
		writer.writerow(routing_headers)

	metrics_outpath = "output/{}-metrics.csv".format(args.title)
	metric_headers = ["Experiment", "Scenario_ID", "Method", "Iter"]
	for metric in METRIC_NAMES:
		metric_headers.append(metric)
	metric_headers.append("Soln_Time")
	with open(metrics_outpath, "w", newline="") as outcsv:
		writer = csv.writer(outcsv)
		writer.writerow(metric_headers)

	arrival_times_outpath = "output/{}-arrival-times.csv".format(args.title)
	arrival_times_headers = ["Experiment", "Scenario_ID", "Method", "Iter", "LocName", "ReqID", "Arrival_Time", "Delay", "Bounded_Delay", "Late_Arrival"]
	with open(arrival_times_outpath, "w", newline="") as outcsv:
		writer = csv.writer(outcsv)
		writer.writerow(arrival_times_headers)

	print("Preparing scenario information...")

	scenarios = list()
	scenario_vehreqs = dict()
	scenario_reverse_map = dict()

	if args.scenarios.isdecimal():
		num_scenarios = int(args.scenarios)
		for scenario_id in range(num_scenarios):
			print("  Generating Scenario {}/{}".format(scenario_id, num_scenarios))
			scenario_rids = rs.choice(range(len(R)), args.reqsize, replace=False)
			scenario_reqs = [copy.deepcopy(reqs[rid]) for rid in scenario_rids]

			scenario_vids = rs.choice(range(len(V)), args.vehsize, replace=False)
			scenario_vehs = [copy.deepcopy(vehs[vid]) for vid in scenario_vids]

			scenario = dict()
			scenario_reverse_map[scenario_id] = {"reqs": dict(), "vehs": dict()}
			scenario["experiment"] = args.title
			scenario["scenario_id"] = scenario_id
			for i in range(args.reqsize):
				scenario["req_{}".format(i)] = scenario_rids[i]
				scenario_reverse_map[scenario_id]["reqs"][scenario_rids[i]] = i
			for j in range(args.vehsize):
				scenario["veh_{}".format(j)] = scenario_vids[j]
				scenario_reverse_map[scenario_id]["vehs"][scenario_vids[j]] = j

			for i, req in enumerate(scenario_reqs):
				req.id = i
			for j, veh in enumerate(scenario_vehs):
				veh.id = j

			scenario_vehreqs[scenario_id] = (scenario_vehs, scenario_reqs)
			scenarios.append(scenario)
	else:
		scenarios = pd.read_csv(args.scenarios)
		num_scenarios = len(scenarios)
		req_cols = [col for col in scenarios if col.startswith('req')]
		veh_cols = [col for col in scenarios if col.startswith('veh')]
		for scenario_id, row in scenarios.iterrows():
			print("  Loading Scenario {}/{}".format(scenario_id, num_scenarios))
			scenario_rids = list(row[req_cols])
			scenario_reqs = [copy.deepcopy(reqs[rid]) for rid in scenario_rids]

			scenario_vids = list(row[veh_cols])
			scenario_vehs = [copy.deepcopy(vehs[vid]) for vid in scenario_vids]

			scenario = dict()
			scenario_reverse_map[scenario_id] = {"reqs": dict(), "vehs": dict()}
			scenario["experiment"] = args.title
			scenario["scenario_id"] = scenario_id
			for i in range(args.reqsize):
				scenario["req_{}".format(i)] = scenario_rids[i]
				scenario_reverse_map[scenario_id]["reqs"][scenario_rids[i]] = i
			for j in range(args.vehsize):
				scenario["veh_{}".format(j)] = scenario_vids[j]
				scenario_reverse_map[scenario_id]["vehs"][scenario_vids[j]] = j

			for i, req in enumerate(scenario_reqs):
				req.id = i
			for j, veh in enumerate(scenario_vehs):
				veh.id = j

			scenario_vehreqs[scenario_id] = (scenario_vehs, scenario_reqs)

	scenario_DF = pd.DataFrame(scenarios)
	scenario_DF.to_csv("output/{}-scenarios.csv".format(args.title))

	print("Scenario information output to output/{}-scenarios.csv\n".format(args.title))

	print(scenario_reverse_map)

	alonsomora = AlonsoMora(params={
		"method": "tabu",
		"tabuMaxTime": 2,
	})

	flowmatching = MinDelayFlowMatching(params=dict())

	for scenario_id in range(num_scenarios):
		print("Initializing scenario {}/{}".format(scenario_id, num_scenarios))
		scenario_vehs, scenario_reqs = scenario_vehreqs[scenario_id]

		# print("Benchmark method 1: constrained Alonso-Mora")
		# t_start = time.time()
		# routes, rej = alonsomora.optimizeAssignment(scenario_vehs, scenario_reqs, G)
		# t_opt = time.time() - t_start
		# print("Optimization finished in {:.2f} seconds".format(t_opt))

		# # metrics = computeDelayMetrics(routes, G, vehs, reqs, T)
		# results = analyzeOptResults(routes, G, vehs, reqs, T_val, scenario_reverse_map[scenario_id], t_opt, args.title, scenario_id, "Constrained Alonso-Mora", 0)

		unconstrainedReqs = copy.deepcopy(scenario_reqs)
		for req in unconstrainedReqs:
			req.Clp = 1e6
			req.Cld = 1e6

		print("Benchmark method 2: unconstrained Alonso-Mora")
		t_start = time.time()
		routes, rej = alonsomora.optimizeAssignment(scenario_vehs, unconstrainedReqs, G)
		t_opt = time.time() - t_start
		print("Optimization finished in {:.2f} seconds".format(t_opt))

		# metrics = computeDelayMetrics(routes, G, vehs, reqs, T)
		results = analyzeOptResults(routes, G, vehs, reqs, T_val, scenario_reverse_map[scenario_id], t_opt, args.title, scenario_id, "Unconstrained Alonso-Mora", 0)

		for eval_iter in range(args.evaluations):
			for num_samples in args.samples:
				# First use weights [1,0,0] i.e. only consider the average delay
				method = "Stoch[1,0,0] (N={})".format(num_samples)
				print("Stochastic formulation [1,0,0] ({} samples drawn) - iteration {}".format(num_samples, eval_iter))
				t_start = time.time()
				routes, rej = flowmatching.optimizeAssignment(scenario_vehs, scenario_reqs, G, T_test, num_samples, weights=[1,0,0], seed=rs.randint(100000000),
															  title="{}-scen{}-stoch[1,0,0]-N{}-iter{}".format(args.title, scenario_id, num_samples, eval_iter))
				t_opt = time.time() - t_start
				print("Optimization finished in {:.2f} seconds".format(t_opt))

				results = analyzeOptResults(routes, G, vehs, reqs, T_val, scenario_reverse_map[scenario_id], t_opt, args.title, scenario_id, method, eval_iter)

				# Next use weights [1,0,16] i.e. 16 m of travel distance = 1 s of delay
				method = "Stoch[1,0,16] (N={})".format(num_samples)
				print("Stochastic formulation [1,0,16] ({} samples drawn) - iteration {}".format(num_samples, eval_iter))
				t_start = time.time()
				routes, rej = flowmatching.optimizeAssignment(scenario_vehs, scenario_reqs, G, T_test, num_samples, weights=[1,0,16], seed=rs.randint(100000000),
															  title="{}-scen{}-stoch[1,0,16]-N{}-iter{}".format(args.title, scenario_id, num_samples, eval_iter))
				t_opt = time.time() - t_start
				print("Optimization finished in {:.2f} seconds".format(t_opt))

				results = analyzeOptResults(routes, G, vehs, reqs, T_val, scenario_reverse_map[scenario_id], t_opt, args.title, scenario_id, method, eval_iter)


if __name__ == "__main__":
	main()