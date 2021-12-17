import argparse
import os

import pandas as pd


def main():
	parser = argparse.ArgumentParser(
		description="Parse command line optimization tests from file and execute sequentially")

	parser.add_argument("-F", "--file", help="Filepath to input commands file.", type=str, required=True)

	args = parser.parse_args()

	commands = pd.read_csv(args.file)

	for idx, row in commands.iterrows():
		if os.path.isfile("output/" + row["Title"] + "-metrics.csv"):
			print("Command {} (\"{}\") has already output results".format(idx, row["Title"]))
		else:
			os.system(row["Command"])


if __name__ == "__main__":
	main()