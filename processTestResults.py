import os
import sys
import time

import pandas as pd

from analyzeRouting import main as analyzeRouting
from generateMetadata import generateMetadata

experiment_log = sys.argv[1]
experiment_dir, _ = os.path.split(experiment_log)

X = pd.read_csv(experiment_log)

if len(sys.argv) >= 3:
	all_routings = sys.argv[2]

	metadata_dict = dict()
	# if os.path.exists(experiment_log.replace(".csv", "-pareto-frontiers.csv")):
	# 	print("Metadata file found! Importing...")
	# 	metadata_file = pd.read_csv(experiment_log.replace(".csv", "-pareto-frontiers.csv"), index_col=0)
	# 	for _, row in metadata_file.iterrows():
	# 		g = row["Graph_File"]
	# 		r = row["Input_Requests"]
	# 		v = row["Input_Vehicles"]
	# 		s = row["Scenarios"]
	# 		scenario_identifier = row["Scenario_Identifier"]
	# 		if (g, r, v, s) not in metadata_dict.keys():
	# 			metadata_dict[g, r, v, s] = dict()

	# 		if scenario_identifier not in metadata_dict[g, r, v, s].keys():
	# 			metadata_dict[g, r, v, s][scenario_identifier] = {
	# 		        "avg_min_vo_dist": list(),
	# 		        "avg_od_dist": list(),
	# 		        "efficient_assignments": list(),
	# 		        "efficient_pts": list()}

	# 		metadata_dict[g, r, v, s][scenario_identifier]["efficient_assignments"].append(row["Assignment"])

	# 	print("Successfully imported metadata from file.")

else:
	all_routings = None

for _, row in X.iterrows():
	title = row["Experiment_Name"] + "-Processed"
	graph = row["Graph_File"]
	requests = row["Input_Requests"]
	vehicles = row["Input_Vehicles"]
	scenarios = experiment_dir + "/" + row["Experiment_Name"] + "-scenarios.csv"
	routings = experiment_dir + "/" + row["Experiment_Name"] + "-routing.csv"
	speeds = row["Input_Speeds"].replace("-speeds.csv", "-val-speeds.csv")

	if all_routings is None:
		scenario_metadata_dict = dict()
	else:
		print("Evaluating Pareto frontier for {}...".format(title))

		pre_evaluated_scenarios = dict()
		scenario_metadata_dict = dict()

		if (graph, requests, vehicles, speeds) in metadata_dict.keys():
			pre_evaluated_scenarios = metadata_dict[(graph, requests, vehicles, speeds)]
		else:
			metadata_dict[(graph, requests, vehicles, speeds)] = dict()

		scenario_df = pd.read_csv(scenarios, index_col=0)

		req_cols = [col for col in scenario_df if col.startswith('req')]
		veh_cols = [col for col in scenario_df if col.startswith('veh')]

		for scenario_id, scenario in scenario_df.iterrows():
			# scenario_id = scenario["scenario_id"]
			print("Scenario {}".format(scenario["scenario_id"]))

			scenario_identifier = "Reqs: " + ", ".join([str(x) for x in list(scenario[req_cols])]) + " | Vehs: " + ", ".join([str(x) for x in list(scenario[veh_cols])])

			if scenario_identifier in pre_evaluated_scenarios.keys():
				print("  Metadata for scenario {} already evaluated!".format(scenario["scenario_id"]))
				scenario_metadata_dict[scenario_id] = pre_evaluated_scenarios[scenario_identifier]

			else:
				print("  Calculating new metadata for scenario {}".format(scenario["scenario_id"]))

				t = time.time()
				scenario_metadata = generateMetadata(graph, requests, vehicles, speeds, scenario, req_cols, veh_cols, all_routings)

				scenario_metadata_dict[scenario_id] = scenario_metadata
				metadata_dict[(graph, requests, vehicles, speeds)][scenario_identifier] = scenario_metadata

				meta_time = time.time() - t
				print("  Metadata for scenario {} calculated in {:.1f} seconds".format(scenario["scenario_id"], meta_time))

				# for key in scenario_metadata:
				# 	print("    Scenario {} {} = {}".format(scenario_id, key, scenario_metadata[key]))

	analyzeRouting(title, graph, speeds, requests, vehicles, scenarios, routings, scenario_metadata_dict)

frontiers = list()
for g, r, v, s in metadata_dict.keys():
	for scenario_identifier in metadata_dict[g, r, v, s].keys():
		for assignment in metadata_dict[g, r, v, s][scenario_identifier]["efficient_assignments"]:
			frontier_info = {"Graph_File": g, "Input_Requests": r, "Input_Vehicles": v, "Scenarios": s, "Scenario_Identifier": scenario_identifier, "Assignment": assignment}
			frontiers.append(frontier_info)

frontier_df = pd.DataFrame(frontiers)
frontier_df.to_csv(experiment_log.replace(".csv", "-pareto-frontiers.csv"))
