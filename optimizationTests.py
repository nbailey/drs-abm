import copy
import csv
import igraph as ig
import numpy as np
import pandas as pd
import time

from lib.Agents import *
from lib.Optimization import *
from lib.RoutingEngine import *

# import scripts.optDraftTwo as od
from scripts.analyzeRouting import computeDelaysMetrics
import scripts.setupGraph as sg

OUT_TABLE = "output/optimization-testing-gr40-10v15r-unconstrainedAM-pt3.csv"

METRIC_HEADERS = ["CRVI", "Certain_Delayed_Trip_Count", "Raw_Delay", "Bounded_Delay", "Late_Arrival_Pct", "Routing"]

SEEDS = [76430, 404532, 33056, 400529, 752796,
		 576792, 858374, 909047, 987227, 905252, 
		 115304, 735597, 456409, 880912, 626737,
		 956382, 625166, 406721, 364452, 12627,
		 838455, 584522, 690561, 492257, 669087,
		 34207, 27258, 26107, 60378,]

rs = np.random.RandomState(seed=SEEDS[-2])

print("Reading speeds table...")

# speeds_iid_all = np.array(pd.read_csv("input/speeds/iid-all.csv", index_col=0))
# speeds_iid_thru = np.array(pd.read_csv("input/speeds/iid-thru.csv", index_col=0))
# speeds_hood_all = np.array(pd.read_csv("input/speeds/hood-all.csv", index_col=0))
# speeds_hood_thru = np.array(pd.read_csv("input/speeds/hood-thru.csv", index_col=0))
# speeds_corr_thru = np.array(pd.read_csv("input/speeds/corr-thru.csv", index_col=0))
speeds_hood_ring = np.array(pd.read_csv("input/speeds/hood-monoc-G40.csv",index_col=0))

GR = ig.Graph.Read_GML("input/maps/synthetic-graph-40-trimmed.gml")

# times_iid_all = GR.es["length"] / speeds_iid_all
# times_iid_thru = GR.es["length"] / speeds_iid_thru
# times_hood_all = GR.es["length"] / speeds_hood_all
# times_hood_thru = GR.es["length"] / speeds_hood_thru
# times_corr_thru = GR.es["length"] / speeds_corr_thru
times_hood_ring = GR.es["length"] / speeds_hood_ring

TIMES = {
	# "iid-all": times_iid_all,
	# "iid-thru": speeds_iid_thru,
	# "hood-all": times_hood_all,
	# "hood-thru": speeds_hood_thru,
	# "corr-thru": times_corr_thru,
	"hood-ring": times_hood_ring,
}

alonsomora = AlonsoMora(params={
	"method": "tabu",
	"tabuMaxTime": 2,
})

flowmatching = MinDelayFlowMatching(params=dict())


NV = 10
NR = 15

# hood_alloc = None
hood_alloc = {1: 1, 2: 0, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 0, 13: 0, 14: 1, 15: 0, 16: 1}

assigned_vehs = rs.choice(range(NV), int(np.floor(NV/10)), replace=False)
assigned_reqs = rs.choice(range(NR), int(np.floor(NR/3)), replace=False)

# A = {assigned_vehs[0]: [assigned_reqs[0], assigned_reqs[1]], assigned_vehs[1]: [assigned_reqs[2],]}#, assigned_vehs[2]: [assigned_reqs[3], assigned_reqs[4]]}
A = None

# # _, GF_small, paths_small, vehs_small, reqs_small = sg.setupGraph(5, 10, None)
# _, GF1, paths1, vehs1, reqs1 = sg.setupGraph(NV, NR, A, seed=rs.randint(0, 10**9))
# # _, GF2, paths2, vehs2, reqs2 = sg.setupGraph(NV, NR, A)
# # _, GF3, paths3, vehs3, reqs3 = sg.setupGraph(NV, NR, A)

# GF1.write("input/maps/optimization-testing-flow-graph-alonsomora-test.gml", "gml")

# setups = [
# 	# ("small", GF_small, paths_small, vehs_small, reqs_small),
# 	(0, GF1, paths1, vehs1, reqs1),
# 	# (1, GF2, paths2, vehs2, reqs2),
# 	# (2, GF3, paths3, vehs3, reqs3)
# ]

header_row = ["Method", "Speeds", "Iter", "Num_Scenarios", "Num_Vehs", "Num_Reqs", "Runtime", "Num_Rej"]
for metric_name in METRIC_HEADERS:
	header_row.append(metric_name)

# for v in range(NV):
# 	header_row.append("Veh_{}_Route".format(v))
# header_row.append("Rej_Trips")
# header_row.append("Num_Rej")

with open(OUT_TABLE, "w", newline="") as outcsv:
	writer = csv.writer(outcsv)
	writer.writerow(header_row)

for times_dbn in TIMES.keys():
	times_table = TIMES[times_dbn]

	for idx, edge_speeds in enumerate(times_table.T):
		edge = GR.es[idx]
		edge["ttmedian"] = np.median(edge_speeds)

	_, GF, pathrows, vehs, reqs = sg.setupGraph(NV, NR, A, seed=rs.randint(0, 10**9), GR=GR, neighborhood_alloc=hood_alloc)
	GF.write("input/maps/opt-test-G40-{}.gml".format(times_dbn), "gml")

	unconstrainedReqs = copy.deepcopy(reqs)
	for req in unconstrainedReqs:
		req.Clp = 1e6
		req.Cld = 1e6

	t_start = time.time()
	routes, rej = alonsomora.optimizeAssignment(vehs, unconstrainedReqs, GR)
	t_opt = time.time() - t_start

	print(routes)
	print(rej)

	results_row = ["Alonso-Mora", times_dbn, 0, 0, len(vehs), len(reqs), t_opt, len(rej)]

	# for veh in vehs:
	# 	if len(routes) > 0 and veh.id in routes.keys():
	# 		route = routes[veh.id]
	# 		vehRoute = list()
	# 		olng = int(veh.lng)
	# 		olat = int(veh.lat)
	# 		for leg in route:
	# 			src = GF.vs.find(lng=olng, lat=olat)

	# 			dlng = int(leg[2])
	# 			dlat = int(leg[3])
	# 			tgt = GF.vs.find(lng=dlng, lat=dlat)

	# 			vehRoute.append((src["nid"], tgt["nid"]))

	# 			olng = dlng
	# 			olat = dlat
	# 		results_row.append(str(vehRoute))
	# 	else:
	# 		src = GF.vs.find(name="veh_{}_start".format(veh.id))
	# 		tgt = GF.vs.find(vtype="sink")
	# 		results_row.append(str([(src["nid"], tgt["nid"]),]))

	# results_row.append(str(rej))
	# results_row.append(len(rej))

	metrics = computeDelaysMetrics(routes, GR, vehs, reqs, times_table)

	for metric in metrics:
		results_row.append(metric)

	results_row.append(str(routes))

	with open(OUT_TABLE, "a+", newline="") as outcsv:
		writer = csv.writer(outcsv)
		writer.writerow(results_row)

	for iteration in range(3):
		for numScenarios in (3, 5, 10):#, 20, 50):
		# for numScenarios in (3,):
			method = "Stoch_{}".format(numScenarios)
			print("Optimizing using stochastic formulation ({} scenarios) for {} speeds - iter {}".format(numScenarios, times_dbn, iteration))
			t_start = time.time()

			# m, flow, rej, scen = od.optimize_assignment(GF, numScenarios, pathrows, times_table, seed=SEEDS[iteration])
			# m, flow, rej, scen = od.optimize_assignment(GF, numScenarios, pathrows, times_table, seed=SEEDS[iteration], TIME_LIMIT=200*numScenarios)
			routes, rej = flowmatching.optimizeAssignment(vehs, reqs, GR, times_table, numScenarios, weights=[0,0,1], seed=SEEDS[iteration])
			t_opt = time.time() - t_start

			print(routes)
			print(rej)

			results_row = [method, times_dbn, iteration, numScenarios, len(vehs), len(reqs), t_opt, len(rej)]

			metrics = computeDelaysMetrics(routes, GR, vehs, reqs, times_table)

			for metric in metrics:
				results_row.append(metric)

			results_row.append(str(routes))

			# for veh in vehs:
			# 	vid = veh.id
			# 	veh_flow = [f[0] for f in flow.keys() if f[1] == vid and np.isclose(flow[f], 1.0)]
			# 	veh_nodes = [(GF.es[e].source, GF.es[e].target) for e in veh_flow]
			# 	results_row.append(str(veh_nodes))

			# rej_trips = [r for r in rej if np.isclose(r, 1.0)]
			# results_row.append(str(rej_trips))
			# results_row.append(len(rej_trips))

			with open(OUT_TABLE, "a+", newline="") as outcsv:
				writer = csv.writer(outcsv)
				writer.writerow(results_row)