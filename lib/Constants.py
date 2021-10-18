"""
constants are found here
"""

from lib.Demand import *

# List of tuples of two random seeds for use in the Model
AMOD_SEEDS = [(750647, 605886),
			  (989792, 939889),
			  (97990, 451740),
			  (174830, 844466),
			  (889073, 609353),
			  (393230, 61254),
			  (507793, 791264),
			  (995524, 156179),
			  (503345, 989240),
			  (82059, 600134),
			  (628566, 635310),
			  (550825, 342710),
			  (478102, 718818),
			  (737520, 954525),
			  (197916, 748653),
			  (426615, 270593),
			  (31274, 936874),
			  (4077, 648826),
			  (426811, 717490),
			  (603262, 536182),
			  (266333, 859300),
			  (17388, 605667),
			  (413539, 779558),
			  (408962, 94663),
			  (229152, 396390),
			  (588632, 193445),
			  (246887, 574366),]

UNCERTAINTY_SEEDS = [76430,
	[404532, 33056, 400529, 752796],
	[576792, 858374, 909047, 987227],
	[905252, 115304, 735597, 456409],
	[880912, 626737, 956382, 625166],
	[406721, 364452, 12627, 838455],
	[584522, 690561, 492257, 669087,],
	[34207, 27258, 26107, 60378],
	[606888, 565062, 576961, 803493],
	[421167, 297518, 129191, 961417],
	[3196, 890554, 799327, 447215],
	[957737, 739430, 562751, 991267],
	[241537, 363019, 116581, 750647]]

# Use the above list or generate and record random seeds
USE_SEEDS = True

# How many simulations at each fleet size to do
SAMPLE_SIZE = 10

EXP_NAME = "Test"

# Base Fare from Leo
# price_base = 0.831 # dollars/ per trip
# price_unit_time = 0.111 # dollars/min
# price_unit_distance = 0.547 # dollars/km
# sharing_discount = 0.75 # 25% discount
# transit_connect_discount = 1.33 # dollars
# min_cost_avpt = 1.73 # dollars

# #Fare
# price_base = 1.662 # dollars/ per trip
# price_unit_time = 0.222 # dollars/min
# price_unit_distance = 1.094 # dollars/km
# sharing_discount = 1 # 25% discount
# transit_connect_discount = 0 # dollars
# min_cost_avpt = 5.67 # dollars
# FARE = [price_base, price_unit_time, price_unit_distance, sharing_discount, transit_connect_discount, min_cost_avpt]

# fleet size and vehicle capacity
FLEET_SIZE = [300]
VEH_CAPACITY = 4

# Distance threshold for minimum service distance (setting to 0 means no distance threshold - trips can be of any length)
DISTANCE_THRESHOLD = 0 # meters
# TODO: Add capability for trips to specific destinations (i.e. rail station) still be served regardless of distance threshold
TIME_THRESHOLD = 0 # seconds

# ASC and the nickname of the run
ASC_AVPT = -3.0
ASC_NAME = "AVPT" + str(ASC_AVPT)

# cost-benefit analysis
COST_BASE = 0.0
COST_MIN = 0.061
COST_KM = 0.289
PRICE_BASE = 0.831
PRICE_MIN = 0.111
PRICE_KM = 0.527
PRICE_DISC = 0.75

# Whether the link travel times use a triangular distribution (tt_opt, tt_avg, tt_pes) or not
LINK_UNCERTAINTY = True
UNCERTAINTY_MULTIPLIER = 1.5

# Whether the vehicles know the link travel time ahead of insertion or afterwards
PRE_DRAW_TTS = False

# Whether any requests are generated in advance
IN_ADV_REQS = False

# initial wait time and detour factor when starting the interaction
INI_WAIT = 400
INI_DETOUR = 1.25

# number of iteration steps
ITER_STEPS = 1

# warm-up time, study time and cool-down time of the simulation (in seconds)
T_WARM_UP = 60*30 # 60*30
T_STUDY = 60*60 # 60*60
T_COOL_DOWN = 60*60 # 60*30
T_TOTAL = (T_WARM_UP + T_STUDY + T_COOL_DOWN)

# methods for vehicle-request assignment and rebalancing
# ins = insertion heuristics, alonsomora = RTV-graph optimal assignment
# sar = simple anticipatory rebalancing, orp = optimal rebalancing problem, dqn = deep Q network
MET_ASSIGN = "alonsomora"
MET_REOPT = "no"
MET_REBL = "no"

# Optimization parameters for assignment
OPT_PARAMS = {
	"method": "tabu",
	"tabuMaxTime": 3,
}

# intervals for vehicle-request assignment and rebalancing
INT_ASSIGN = 30
INT_REBL = 150

# if true, activate the animation
IS_ANIMATION = False

# Enables printing of state of simulation every update
PRINT_PROGRESS = True

# maximum detour factor and maximum wait time window
MAX_DETOUR = 1.5
MAX_WAIT = 60*10

# constant vehicle speed when road network is disabled (in meters/second)
CST_SPEED = 9 # Based on empirical results from routing engine
# CST_SPEED = 6

# Conversion factors between miles per hour and meters per second
MPH_2_MPS = 0.44704
MPS_2_MPH = 2.23694

# probability that a request is sent in advance (otherwise, on demand)
PROB_ADV = 0.0
# time before which system gets notified of the in-advance requests
T_ADV_REQ = 60*30

# coefficients for wait time and in-vehicle travel time in the utility function
# coefficient for extra wait time applies when forced to wait longer after already
# being assigned to a vehicle
COEF_INVEH = 1.0
COEF_WAIT = 1.5
COEF_EXTRA_WAIT = 3.0

# map width and height
MAP_WIDTH = 5.52
MAP_HEIGHT = 6.63

# coordinates
# (Olng, Olat) lower left corner
Olng = -0.02
Olat = 51.29
# (Dlng, Dlat) upper right corner
Dlng = 0.18
Dlat = 51.44
# number of cells in the gridded map
Nlng = 10
Nlat = 10
# number of moving cells centered around the vehicle
Mlng = 5
Mlat = 5
# length of edges of a cell
Elng = 0.02
Elat = 0.015