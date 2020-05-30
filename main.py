import argparse
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
from local import graph_loc


if __name__ == "__main__":
	logging.basicConfig(filename='C:/Users/n3tur/Documents/GitHub/drs-abm/wtf-log',level=logging.DEBUG)
	# try:
	parser = argparse.ArgumentParser(
		description="Agent-based model for Automated Mobility-on-Demand service simulation")
	parser.add_argument('-f', '--fleet',
						help='Fleet sizes to simulate, formatted as comma-separated list (i.e. "-f 250,275,300")')
	parser.add_argument('-i', '--iteration', type=int,
						help='Iteration number (if running many in parallel)')

	args = parser.parse_args()
	if args.fleet:
		fleet_size_array = [int(x) for x in args.fleet.split(',')]
	else:
		fleet_size_array = FLEET_SIZE

	results_filename = './output/results{}{}.csv'.format(
		'-fleet'+str(args.fleet) if args.fleet else '',
		'-iter'+str(args.iteration) if args.iteration else '')


	print('Writing results to {}'.format(results_filename))

	with open(results_filename, 'a') as results_file:
		writer = csv.writer(results_file)
		row = ["ASC", "step", "ASSIGN", "REBL", "T_STUDY", "fleet_size", "capacity", "volume", "service_rate",
			   "count_served", "count_reqs", "service_rate_ond", "count_served_ond", "count_reqs_ond",
			   "service_rate_adv", "count_served_adv", "count_reqs_adv", "wait_time", "wait_time_adj",
			   "wait_time_ond", "wait_time_adv", "in_veh_time", "detour_factor", "veh_service_dist",
			   "veh_service_time", "veh_service_time_percent", "veh_pickup_dist", "veh_pickup_time",
			   "veh_pickup_time_percent", "veh_rebl_dist", "veh_rebl_time", "veh_rebl_time_percent",
			   "veh_load_by_dist", "veh_load_by_time", "cost", "benefit", "logsum_w_AVPT", "logsum_wout_AVPT",
			   "overall_logsum", "ridershipchange_car", "ridershipchange_walk", "ridershipchange_bike",
			   "ridershipchange_taxi", "ridershipchange_bus", "ridershipchange_rail", "ridershipchange_intermodal",
			   None]
		writer.writerow(row)

	# if road network is enabled, initialize the routing server
	# otherwise, use Euclidean distance
	router = RoutingEngine(graph_loc, CST_SPEED, seed = UNCERTAINTY_SEEDS[0])

	for fleet_iter, fleet_size in enumerate(fleet_size_array):
		veh_capacity = VEH_CAPACITY
		wait_time_adj = INI_WAIT
		detour_factor = INI_DETOUR
		demand_matrix = INI_MAT
		asc_avpt = ASC_AVPT
		counter = 0
		fare = []
		for value in FARE:
				fare.append(value)
		# delete later
		print("fare:", fare)

		#initalize decision variables as 0 after each loop completion
		df_OD_LOS = pd.DataFrame(pd.np.empty((1057, 6)) * pd.np.nan,columns=['m_id', 'summed_wait_time', 'summed_detour_factor', 'number_of_occurances','wait_time', 'detour_factor'])
		df_OD_LOS['m_id'] = df_OD_LOS.index
		df_OD_LOS['number_of_occurances'] = 0
		df_OD_LOS['summed_wait_time'] = 0
		df_OD_LOS['summed_detour_factor'] = 0
		df_OD_LOS['wait_time'] = int(INI_WAIT)
		df_OD_LOS['detour_factor'] = int(INI_DETOUR)

		#iteration
		for step in range(SAMPLE_SIZE):
			reqfname = './output/requests{}{}{}.csv'.format(
				'-fleet'+str(args.fleet) if args.fleet else '',
				'-iter'+str(args.iteration) if args.iteration else '',
				'-step'+str(step))
			# run simulation
			# model, step, runtime, logsum_w_AVPT, logsum_wout_AVPT, df_diffprob = run_simulation(
				# osrm, step, demand_matrix, fleet_size, veh_capacity, asc_avpt, fare, df_OD_LOS)
			model, step, runtime, logsum_w_AVPT, logsum_wout_AVPT, df_diffprob = run_simulation(
				router, step, demand_matrix, fleet_size, veh_capacity, asc_avpt, fare, df_OD_LOS, AMOD_SEEDS[step])
			# output the simulation results and save data
			wait_time_adj, detour_factor, df = print_results(
				model, step, runtime, fare, logsum_w_AVPT, logsum_wout_AVPT,fleet_size, 1,
				df_diffprob, results_filename, requests_filename=reqfname)

			df_OD_LOS = df.copy(deep=True)

		del df_OD_LOS
	# except Exception as e:
	#     logging.info(e)
	#     print("!")