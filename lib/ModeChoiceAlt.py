import numpy as np
import pandas as pd

from lib.Constants import *
from lib.ModeChoiceVars import *


def calcShrUtility(attrs):
	cost = attrs["cost"]
	ewt = attrs["ewt"]
	ett = attrs["ett"]
	ttv = attrs["ttv"]

	V = B_SHR_COST * cost + B_SHR_EWT * ewt + B_SHR_ETT * ett + B_SHR_TTV * ttv + ASC_SHR + ASC_SHR_RND * np.random.normal()

	return V

def calcExcUtility(attrs):
	cost = attrs["cost"]
	ewt = attrs["ewt"]
	ett = attrs["ett"]
	ttv = attrs["ttv"]

	V = B_EXC_COST * cost + B_EXC_EWT * ewt + B_EXC_ETT * ett + B_EXC_TTV * ttv + ASC_EXC + ASC_EXC_RND * np.random.normal()

	return V

def calcPTUtility(ptAttrs):
	return 1

def calcCarUtility(carAttrs):
	return 1


UTILITY_FUNCTIONS = {
	"exc": calcExcUtility,
	"shr": calcShrUtility,	
	"pt": calcPTUtility,
	"car": calcCarUtility,
}

class DemandEntry():
	# Origin location
	# Destination location
	# Mode availability
	# Individual sociodemographics
	def __init__(self, olat, olng, dlat, dlng, ptAttrs=None, carAttrs=None):
		self.olng = olng
		self.olat = olat
		self.dlng = dlng
		self.dlat = dlat
		self.ptAttrs = self.ptAttrs
		self.carAttrs = self.carAttrs

	# Give the probability for this traveler selecting each mode available to them
	# Currently supports exclusive and shared ridehailing ("exc" and "shr"), plus
	# public transit ("pt") and private car ("car").
	def calc_mode_probs(self, excAttrs, shrAttrs):
		num_alternatives = 2

		allModeAttrs = {
			"exc": excAttrs,
			"shr": shrAttrs,
		}

		if self.ptAttrs is not None and self.ptAttrs["av"]:
			allModeAttrs["pt"] = self.ptAttrs
			num_alternatives += 1

		if self.carAttrs is not None and self.carAttrs["av"]:
			allModeAttrs["car"] = self.carAttrs
			num_alternatives += 1

		V = dict()
		for mode in allModeAttrs.keys():
			V[mode] = UTILITY_FUNCTIONS[mode](allModeAttrs[mode])

		exp_sum = np.sum([np.exp(v_mode) for v_mode in V.values()])
		mode_probs = dict()

		for mode in ("exc", "shr", "pt", "car"):
			if mode in allModeAttrs.keys():
				mode_probs[mode] = np.exp(V[mode]) / exp_sum
			else:
				mode_probs[mode] = 0

		return mode_probs


class DemandUpdater():
	# 
	def __init__(self, router):
		self.router = router
		self.entries = list()

	def addEntriesFromCsv(self, filepath):
		# Add a tuple to the entries list for each row in the input csv
		# containing a DemandEntry object representing that row and a
		# rate parameter for trips represented by that entry

	def generateDemand(self, T):
		# Randomly generates a trip O/D for a traveler entering the system to use
		# as the input for offer generation for each ridehailing system in operation

	def updateEntries(self, performances):
		# Updates any parameters affected by the level of service of a ridehailing system in the previous
		# timestep, such as trust in on-time performance
