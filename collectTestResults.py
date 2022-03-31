import csv
import os
import sys

import numpy as np
import pandas as pd

experiment_log = sys.argv[1]
benchmark_log = "benchmark-experiment-log.csv"

benchmark_output = "output/final-benchmark-combined-metrics.csv"
benchmark_arrival_output = "output/final-benchmark-combined-arrival-summary.csv"
experiment_output = "output/" + experiment_log[:-7] + "combined-metrics.csv" # replace "log.csv" with "combined-metrics.csv"
arrival_output = "output/" + experiment_log[:-7] + "combined-arrival-summary.csv"

experiments = pd.read_csv(experiment_log, index_col=False)
benchmarks = pd.read_csv(benchmark_log, index_col=False)

identifiers = list(experiments.columns)[2:]

experiment_metrics = dict()
experiment_metric_names = dict()
print("Reading in experiment data...")
for title in set(experiments["Experiment_Title"]):
	print(title)
	data_filepath = "output/" + title + "-Processed-validation-metrics.csv"

	# colnames = pd.read_csv(data_filepath, nrows=1)
	# data = pd.read_csv(data_filepath, header=None, skiprows=1, usecols=list(range(len(colnames.columns)-1)), names=list(colnames.columns)[:-1])
	# experiment_metrics[title] = data

	# colnames = pd.read_csv(data_filepath, nrows=1)
	# experiment_metrics[title] = pd.read_csv(data_filepath, header=None, skiprows=1, usecols=list(range(len(colnames.columns))), names=colnames.columns)
	experiment_metrics[title] = pd.read_csv(data_filepath)
	experiment_metric_names[title] = list(experiment_metrics[title].columns)[5:]

if os.path.exists(benchmark_output):
	prev_output = pd.read_csv(benchmark_output, index_col=False)
	benchmark_id = np.max(prev_output["Benchmark ID"]) + 1
	print("Previous benchmark data found! Starting benchmark ID: {}".format(benchmark_id))
else:
	benchmark_id = 0
	print("No previous benchmark data found! Starting benchmark ID: 0")

benchmark_dict = dict()

print("Processing benchmark system-level performance data...")
for _, benchmark_row in benchmarks.iterrows():
	if benchmark_row["Experiment_Title"] in set(experiments["Experiment_Title"]):
		title = benchmark_row["Experiment_Title"]
		metrics = experiment_metrics[title]
		METRIC_NAMES = experiment_metric_names[title]

		benchmark_metrics = metrics[metrics["Method_Name"] == benchmark_row["Method_Name"]]

		for _, metrics_row in benchmark_metrics.iterrows():
			output_row = [benchmark_id, title, metrics_row["Scenario_ID"]] + [metrics_row[metric] for metric in METRIC_NAMES]

			with open(benchmark_output, "a+", newline="") as outcsv:
				writer = csv.writer(outcsv)
				writer.writerow(output_row)

			benchmark_dict[(title, metrics_row["Scenario_ID"])] = benchmark_id
			benchmark_id += 1

print("Processing experiment system-level performance data...")
experiments_dict = dict()
for _, experiment_row in experiments.iterrows():
	# print(experiment_row)
	title = experiment_row["Experiment_Title"]
	method = experiment_row["Method_Name"]

	experiments_dict[(title, method)] = experiment_row

	print(" - {}: {}".format(title, method))

	metrics = experiment_metrics[title]
	METRIC_NAMES = experiment_metric_names[title]

	matching_metrics = metrics[metrics["Method_Name"] == method]

	for _, metrics_row in matching_metrics.iterrows():
		benchmark_id = benchmark_dict[(title, metrics_row["Scenario_ID"])]

		output_row = [experiment_row[identifier] for identifier in identifiers] + [benchmark_id, metrics_row["Num_Samples"]] + [metrics_row[metric] for metric in METRIC_NAMES]

		with open(experiment_output, "a+", newline="") as outcsv:
			writer = csv.writer(outcsv)
			writer.writerow(output_row)

# Work on arrival times
print("Processing experiment request-level performance data...")
for title in set(experiments["Experiment_Title"]):
	print(" - {}".format(title))

	arrival_times = pd.read_csv("output/" + title + "-Processed-validation-arrival-times.csv", index_col=False)
	print("   Arrival time data processed!")

	benchmark_info = benchmarks[benchmarks["Experiment_Title"] == title]

	for method in set(benchmark_info["Method_Name"]):
		benchmark_arrival_times = arrival_times[arrival_times["Method_Name"] == method]

		for scenario, data in benchmark_arrival_times.groupby("Scenario_ID"):
			print("   * Benchmark: Scenario {}".format(scenario))

			benchmark_id = benchmark_dict[(title, scenario)]

			pickups = data[data["LocName"].str.contains("pickup")]
			dropoffs = data[data["LocName"].str.contains("dropoff")]

			groups = {"pickups": pickups, "dropoffs": dropoffs, "all": data}
			group_info = dict()

			for group in groups.keys():
				group_data = groups[group]
				mean_lateness = np.mean(group_data["Delay"])
				std_lateness = np.std(group_data["Delay"])
				mean_delay = np.mean(group_data["Bounded_Delay"])
				std_delay = np.std(group_data["Bounded_Delay"])
				late_arrival_pct = np.mean(group_data["Late_Arrival"])

				group_info[group] = [mean_lateness, std_lateness, mean_delay, std_delay, late_arrival_pct]

				for pctile in range(5,100,5):
					group_info[group].append(np.percentile(group_data["Delay"], pctile))

			output_row = [benchmark_id,] + [x for x in group_info["all"]] + [y for y in group_info["pickups"]] + [z for z in group_info["dropoffs"]]

			with open(benchmark_arrival_output, "a+", newline="") as outcsv:
				writer = csv.writer(outcsv)
				writer.writerow(output_row)

	experiment_info = experiments[experiments["Experiment_Title"] == title]

	for method in set(experiment_info["Method_Name"]):
		method_arrival_times = arrival_times[arrival_times["Method_Name"] == method]

		for grouping, data in method_arrival_times.groupby(["Scenario_ID", "Num_Samples"]):
			scenario = grouping[0]
			N = grouping[1]

			print("   * {}: Scenario {}, N={}".format(method, scenario, N))

			benchmark_id = benchmark_dict[(title, scenario)]

			pickups = data[data["LocName"].str.contains("pickup")]
			dropoffs = data[data["LocName"].str.contains("dropoff")]

			groups = {"pickups": pickups, "dropoffs": dropoffs, "all": data}
			group_info = dict()

			for group in groups.keys():
				group_data = groups[group]
				mean_lateness = np.mean(group_data["Delay"])
				std_lateness = np.std(group_data["Delay"])
				mean_delay = np.mean(group_data["Bounded_Delay"])
				std_delay = np.std(group_data["Bounded_Delay"])
				late_arrival_pct = np.mean(group_data["Late_Arrival"])

				group_info[group] = [mean_lateness, std_lateness, mean_delay, std_delay, late_arrival_pct]

				for pctile in range(5,100,5):
					group_info[group].append(np.percentile(group_data["Delay"], pctile))

			experiment_row = experiments_dict[(title, method)]
			output_row = [experiment_row[identifier] for identifier in identifiers] + [benchmark_id, N] + [x for x in group_info["all"]] + [y for y in group_info["pickups"]] + [z for z in group_info["dropoffs"]]

			with open(arrival_output, "a+", newline="") as outcsv:
				writer = csv.writer(outcsv)
				writer.writerow(output_row)

	del arrival_times