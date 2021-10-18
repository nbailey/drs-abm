import argparse
import importlib
import logging
import os
import shutil
import sys
import time

from lib.Utils import *
from lib.RoutingEngine import *
from lib.Agents import *
from lib.Demand import *
from lib.Constants import *
from lib.ModeChoice import *
from local import *


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Agent-based model for Automated Mobility-on-Demand service simulation")
	parser.add_argument("fleet_spec_file", metavar="FLEET_SPEC_FILEPATH", type=str,
						help="The path to the file containing the fleet specification to use")
	parser.add_argument("-G", "--graph", metavar="GRAPH_FILEPATH", type=str,
						help="The filepath of the graph to use for routing")
	parser.add_argument("-D", "--demand", metavar="DEMAND_FILEPATH", type=str,
						help="The filepath of the demand specification to use for generating trips")

	args = parser.parse_args()

	spec_path_dir, spec_path_filename = os.path.split(args.fleet_spec_file)
	spec_module_name = ""
	if len(spec_path_dir) == 0:
		spec_module_name = spec_path_filename[:-3] # Strip the trailing .py
	else:
		relpath = os.path.relpath(spec_path_dir) # Path to the specegy file from this directory
		print(relpath)
		spec_module_name = relpath.replace("\\", ".") + "." + spec_path_filename[:-3]
		print(spec_module_name)

	try:
		spec_module = importlib.import_module(spec_module_name)
		FLEET_SPECS = spec_module.FLEET_SPECS
		FLEET_DATA_PATHS = spec_module.FLEET_DATA_PATHS
	except ModuleNotFoundError:
		raise FileNotFoundError("No file found containing fleet specification at {}".format(args.fleet_spec_file))
	except AttributeError:
		raise ValueError("Simulation expected two dictionaries named FLEET_SPECS and FLEET_DATA_PATHS in {} but one or more was not found".format(args.fleet_spec_file))

	demand_filepath = None
	if args.demand is not None:
		demand_filepath = args.demand
	else:
		try:
			demand_filepath = settings.DEMAND_FILE
		except AttributeError:
			raise ValueError("Demand file must be specified with either -D/--demand command line argument or in settings.py")

	graph_loc = None
	if args.graph is not None:
		graph_loc = args.graph
	else:
		try:
			graph_loc = settings.GRAPH_LOC
		except AttributeError:
			raise ValueError("Graph location must be specified with either -G/--graph command line argument or in settings.py")

	# Initialize fleet output files
	for fleet in FLEET_SPECS:
		fleet_filename = "./output/results-{}-f{}.csv".format(EXP_NAME, fleet["name"])
		with open(fleet_filename, 'a') as results_file:
			writer = csv.writer(results_file)
			row = ["ASC", "step", "ASSIGN", "WINDOW", "REBL", "T_STUDY", "fleet_size", "capacity", "service_rate",
				   "count_served", "count_reqs", "service_rate_ond", "count_served_ond", "count_reqs_ond",
				   "service_rate_adv", "count_served_adv", "count_reqs_adv", "wait_time", "wait_time_adj",
				   "wait_time_ond", "wait_time_adv", "in_veh_time", "detour_factor", "veh_service_dist",
				   "veh_service_time", "veh_service_time_percent", "veh_pickup_dist", "veh_pickup_time",
				   "veh_pickup_time_percent", "veh_rebl_dist", "veh_rebl_time", "veh_rebl_time_percent",
				   "veh_load_by_dist", "veh_load_by_time", "veh_occupancy_events", "veh_trips_per_occupancy",]
			writer.writerow(row)

	# if road network is enabled, initialize the routing server
	# otherwise, use Euclidean distance
	router = RoutingEngine(graph_loc, CST_SPEED, seed = UNCERTAINTY_SEEDS[0])

	initial_kernel_data = dict()
	for fleet in FLEET_SPECS:
		initial_fleet_data = pd.read_csv(FLEET_DATA_PATHS[fleet["name"]])
		initial_kernel_data[fleet["name"]] = [initial_fleet_data,]

	run_full_simulation(router, SAMPLE_SIZE, demand_filepath, FLEET_SPECS, initial_kernel_data, UNCERTAINTY_SEEDS[1:])