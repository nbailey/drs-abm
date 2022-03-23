import os
import sys

import pandas as pd

from analyzeRouting import main as analyzeRouting

experiment_log = sys.argv[1]
experiment_dir, _ = os.path.split(experiment_log)

X = pd.read_csv(experiment_log)

for _, row in X.iterrows():
	title = row["Experiment_Name"] + "-Processed"
	graph = row["Graph_File"]
	requests = row["Input_Requests"]
	vehicles = row["Input_Vehicles"]
	scenarios = experiment_dir + "/" + row["Experiment_Name"] + "-scenarios.csv"
	routings = experiment_dir + "/" + row["Experiment_Name"] + "-routing.csv"
	speeds = row["Input_Speeds"].replace("-speeds.csv", "-val-speeds.csv")

	analyzeRouting(title, graph, speeds, requests, vehicles, scenarios, routings)