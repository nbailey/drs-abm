import csv
import os
import sys

import numpy as np
import pandas as pd

experiment_log = sys.argv[1]
benchmark_log = "benchmark-experiment-log.csv"

benchmark_output = "output/benchmark-combined-metrics.csv"
experiment_output = "output/" + experiment_log[:-7] + "combined-metrics.csv" # replace "log.csv" with "combined-metrics.csv"
arrival_output = "output/" + experiment_log[:-7] + "combined-arrival-summary.csv"

experiments = pd.read_csv(experiment_log, index_col=False)
benchmarks = pd.read_csv(benchmark_log, index_col=False)

identifiers = list(experiments.columns)[2:]

experiment_metrics = dict()
experiment_metric_names = dict()
for title in set(experiments["Experiment_Title"]):
	experiment_metrics[title] = pd.read_csv("output/" + title + "-Processed-validation-metrics.csv", index_col=False)
	# experiment_metrics[title] = pd.read_csv("output/" + title + "-metrics.csv")
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
for _, experiment_row in experiments.iterrows():
	# print(experiment_row)
	title = experiment_row["Experiment_Title"]	
	metrics = experiment_metrics[title]
	METRIC_NAMES = experiment_metric_names[title]

	matching_metrics = metrics[metrics["Method_Name"] == experiment_row["Method_Name"]]
	# print(matching_metrics)
	for _, metrics_row in matching_metrics.iterrows():
		benchmark_id = benchmark_dict[(title, metrics_row["Scenario_ID"])]

		output_row = [experiment_row[identifier] for identifier in identifiers] + [benchmark_id, metrics_row["Num_Samples"]] + [metrics_row[metric] for metric in METRIC_NAMES]

		with open(experiment_output, "a+", newline="") as outcsv:
			writer = csv.writer(outcsv)
			writer.writerow(output_row)

# Work on arrival times
print("Processing experiment request-level performance data...")
for title in set(experiments["Experiment_Title"]):
	arrival_times = pd.read_csv("output/" + title + "-Processed-arrival-times.csv", index_col=False)

	experiment_info = experiments[experiments["Title"] == title]

	for method in set(experiment_info["Method_Name"]):
		method_arrival_times = arrival_times[arrival_times["Method"] == method]

		for submethod, data in method_arrival_times.groupby(["Scenario_ID", "Num_Samples"]):
			scenario = submethod[0]
			N = submethod[1]

			benchmark_id = benchmark_dict[(title, scenario)]

			pickups = data[data["LocName"].str.contains("pickup")]
			dropoffs = data[data["LocNAme"].str.contains("dropoff")]

			groups = {"pickups": pickups, "dropoffs": dropoffs, "all": data}
			group_info = dict()

			for group in groups.keys():
				group_data = groups[group]
				mean_lateness = np.mean(group["Delay"])
				std_lateness = np.std(group["Delay"])
				mean_delay = np.mean(group["Bounded_Delay"])
				std_delay = np.mean(group["Bounded_Delay"])
				late_arrival_pct = np.mean(group["Late_Arrival"])

				group_info[group] = {"Avg Lateness": mean_lateness,
									 "Std Lateness": std_lateness,
									 "Avg Delay": mean_delay,
									 "Std Delay": std_delay,
									 "Late Arrival Pct": late_arrival_pct}

			output_row = [experiment_row[identifier] for identifier in identifiers] + [benchmark_id, N] + [x for x in group_info["all"]] + [y for y in group_info["pickups"]] + [z for z in group_info["dropoffs"]]

		with open(arrival_output, "a+", newline="") as outcsv:
			writer = csv.writer(outcsv)
			writer.writerow(output_row)