import numpy as np
import pandas as pd

from lib.Constants import *
from lib.ModeChoiceVars import *


def calcShrUtility(attrs, rs):
	cost = attrs["cost"]
	ewt = attrs["ewt"]
	ett = attrs["ett"]
	ttv = attrs["ttv"]

	V = B_SHR_COST * cost + B_SHR_EWT * ewt + B_SHR_ETT * ett + B_SHR_TTV * ttv + ASC_SHR + ASC_SHR_RND * rs.normal()

	return V

def calcExcUtility(attrs, rs):
	cost = attrs["cost"]
	ewt = attrs["ewt"]
	ett = attrs["ett"]
	ttv = attrs["ttv"]

	V = B_EXC_COST * cost + B_EXC_EWT * ewt + (B_EXC_ETT + B_EXC_ETT_S*rs.normal()) * ett + B_EXC_TTV * ttv

	return V

def calcPTUtility(ptAttrs, rs):
	return -np.inf

def calcCarUtility(carAttrs, rs):
	return -np.inf


UTILITY_FUNCTIONS = {
	"exc": calcExcUtility,
	"shr": calcShrUtility,
	"pt": calcPTUtility,
	"car": calcCarUtility,
}

class Trip():
	# Origin location
	# Destination location
	# Mode availability
	# Individual sociodemographics
	def __init__(self, olat, olng, dlat, dlng, ptAttrs=None, carAttrs=None):
		self.olng = olng
		self.olat = olat
		self.dlng = dlng
		self.dlat = dlat
		self.modeAttrs = {"pt": ptAttrs, "car": carAttrs}
		self.T = None

	# Give the probability for this traveler selecting each mode available to them
	# Currently supports exclusive and shared ridehailing ("exc" and "shr"), plus
	# public transit ("pt") and private car ("car").
	def calc_mode_probs(self, newModeAttrs, rs):
		num_alternatives = len(newModeAttrs)

		allModeAttrs = newModeAttrs

		for mode in self.modeAttrs.keys():
			attrs = self.modeAttrs[mode]
			if attrs is not None and attrs["av"]:
				allModeAttrs[mode] = attrs
				num_alternatives += 1

		V = dict()
		for mode in allModeAttrs.keys():
			V[mode] = UTILITY_FUNCTIONS[mode](allModeAttrs[mode], rs)

		exp_sum = np.sum([np.exp(v_mode) for v_mode in V.values()])
		mode_probs = dict()

		for mode in ("exc", "shr", "pt", "car"):
			if mode in allModeAttrs.keys():
				mode_probs[mode] = np.exp(V[mode]) / exp_sum
			else:
				mode_probs[mode] = 0

		return mode_probs

	def setTime(self, T):
		self.T = T

	def createInstance(self, T):
		instance = Trip(self.olat, self.olng, self.dlat, self.dlng, self.modeAttrs["pt"], self.modeAttrs["car"])
		instance.setTime(T)

		return instance


class DemandUpdater():
	# 
	def __init__(self, seed=None):
		if seed is None:
			seed = np.random.randint(0,1000000)
			print(' - Seed generated for Demand Updater: {}'.format(seed),
				file=open('output/seeds.txt', 'a'))
		self.rs = np.random.RandomState(seed)
		self.trips = list()
		self.freqs = list()
		self.counts = list()

	def addTripsFromCsv(self, filepath):
		# Add a tuple to the trips list for each row in the input csv
		# containing a Trip object representing that row and a
		# rate parameter for the arrival rate of that type of trip

		M = pd.read_csv(filepath)

		total_trips = np.sum(M["count"])

		cum_trips = 0

		for _, row in M.iterrows():
			oloc = (np.round(row["olat"], 8), np.round(row["olng"], 8))
			dloc = (np.round(row["dlat"], 8), np.round(row["dlng"], 8))

			ptAttrs = None
			carAttrs = None

			if row["pt_av"] == 1:
				ptAttrs = None
				# Public transit stuff here...
				pass

			if row["car_av"] == 1:
				carAttrs = None
				# Car stuff here...
				pass

			trip = Trip(oloc[0], oloc[1], dloc[0], dloc[1], ptAttrs, carAttrs)

			freq = float(row["count"] / total_trips)
			self.trips.append(trip)
			self.freqs.append(freq)
			self.counts.append(row["count"])


	def generateArrival(self, T):
		# Randomly generates a trip O/D for a traveler entering the system to use
		# as the input for offer generation for each ridehailing system in operation
		r = self.rs.rand()

		cum_freq = np.cumsum(self.freqs)
		trip_idx = np.argmax(cum_freq > r)

		trip = self.trips[trip_idx]

		return trip.createInstance(T)

	def getDemandVolume(self):
		V = np.sum(self.counts)
		return V

	def updateTrips(self, performances):
		# Updates any parameters affected by the level of service of a ridehailing system in the previous
		# timestep, such as trust in on-time performance

		return None