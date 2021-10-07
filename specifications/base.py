FLEET_SPECS = [
	{"name": "exc",
	 "fleet_size": 200,
	 "veh_capacity": 1,
	 "assign_method": "alonsomora",
	 "window_method": "percentile-90",
	 "rebl_method": "no",
	 "fare": [1.662, 0.222, 1.094, 5.67] # Base price ($), price/time ($/min), price/distance ($/km), minimum fare ($)
	},
	{"name": "shr",
	 "fleet_size": 200,
	 "veh_capacity": 4,
	 "assign_method": "alonsomora",
	 "window_method": "percentile-90",
	 "rebl_method": "no",
	 "fare": [0.75*x for x in [1.662, 0.222, 1.094, 5.67]] # 75% of exclusive fare
	},
]

FLEET_DATA_PATHS = {
	"exc": "output/processed/all-requests-iter15111.csv",
	"shr": "output/processed/all-requests-iter15111.csv",
}