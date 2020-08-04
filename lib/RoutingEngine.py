"""
Open Source Routing Machine (OSRM)
"""

import csv
import os
import requests
import json
import time
import math
import numpy as np
from subprocess import Popen, PIPE

import igraph as ig

from lib.Constants import *
# from lib.LinkTravelTimes import link_travel_times_prec4 as LINK_TRAFFIC_DICT


class RoutingEngine(object):
    """
    RoutingEngine is the class for the routing server
    Attributes:
        graph_loc: the location of the GML file graph representation of the network
        G: the igraph Graph object loaded from graph_loc
        cst_speed: A constant speed to use in some situations
        seed: A random seed used to draw link travel times from a random distribution
        rs: A numpy random state seeded with a random value recorded in output, or a passed in seed
    """
    def __init__(self, graph_loc, cst_speed, seed=None, initial_hour=None, speed_table=None):
        self.G = ig.Graph.Read_GML(graph_loc)
        self.cst_speed = cst_speed
        if seed is None:
            seed = np.random.randint(0,1000000)
            print('ROUTING ENGINE SEED: {}'.format(seed), file=open('output/seeds.txt', 'a+'))
        self.rs = np.random.RandomState(seed)

        if initial_hour is not None:
            self.hour = initial_hour
            self.speeds = speed_table[speed_table["hour"]==self.hour].sample()
        else:
            self.hour = None
            self.speeds = None


    def update_hour(new_hour):
        self.hour = new_hour
        self.speeds = SPEED_TABLE[SPEED_TABLE["hour"]==self.hour].sample()


    def draw_link_travel_time(self, link_index, use_uncertainty=True):
        graph_link = self.G.es[link_index]

        if use_uncertainty:

            if self.speeds is not None:
                mu = float(self.speeds["edge_{}_speed_mean".format(int(link_index))])
                sigma = float(self.speeds["edge_{}_speed_stddev".format(int(link_index))])

                # Set lower limit on road speed to 5mph to avoid outlier results and errors
                speed_draw = np.max([self.rs.normal(mu, sigma), 5])

                return graph_link["length"] / (speed_draw * MPH_2_MPS)

            else:
                delta = graph_link["slnshift"]
                mu = graph_link["slnmean"]
                sigma = graph_link["slnsd"]

                tt_draw = delta + np.exp(self.rs.normal(mu, sigma))

                return tt_draw
        else:
            return graph_link['ttmean']

    # get the best route from origin to destination
    # Can pass in pre_drawn_tt, a tuple where the 0th element is the total route travel time if pre-drawn
    # and the 1st element is a dict of the pre-drawn link travel times with keys as the link index
    def get_routing(self, olng, olat, dlng, dlat, pre_drawn_tt=None, use_uncertainty=False):
        onodename = str((olng, olat))
        dnodename = str((dlng, dlat))
        path = self.G.get_shortest_paths(onodename, to=dnodename, weights='ttmedian', output='epath')[0]

        # Convert path from list of edge ids in Graph to json-like route
        route = dict()

        route['distance'] = sum([self.G.es[idx]['length'] for idx in path])
        route['steps'] = list()
        route['duration'] = 0

        if pre_drawn_tt is not None:
            # print(pre_drawn_tt)
            found_path = pre_drawn_tt[2]
            # print("This path was already found!\n{}".format(found_path))
            # print("This is the new path found during routing:\n{}".format(path))

            if found_path != path:
                print("Path found in routing doesn't match pre-drawn path!!")
                print("ROUTE ALREADY FOUND:\n{}\n".format(found_path))
                print("ROUTE FOUND NOW:\n{}\n".format(path))
                print("Leg starts at: {}".format(onodename))
                print("Leg ends at: {}".format(dnodename))
                print("Diagnostics:")
                return None

            route['duration'] = pre_drawn_tt[0]
            routing_path = pre_drawn_tt[1]
        else:
            routing_path = path

        # Edge sources and targets are not consistent --> keep track iteratively
        # Start with the origin node as the first edge source
        cur_node = onodename

        for edge_index in routing_path:
            edge = self.G.es[edge_index]
            step_dict = dict()

            step_dict['distance'] = edge['length']
            step_dict['mean_duration'] = self.draw_link_travel_time(edge.index, use_uncertainty=False)

            if pre_drawn_tt is not None:
                step_dict['duration'] = pre_drawn_tt[1][edge_index]
            else:
                drawn_link_tt = self.draw_link_travel_time(edge.index, use_uncertainty)
                step_dict['duration'] = drawn_link_tt
                route['duration'] += drawn_link_tt

            if not use_uncertainty:
                assert step_dict['duration'] == step_dict['mean_duration']

            step_dict['weight'] = step_dict['duration']

            # The current node we are at is the source of this step
            srcnode = self.G.vs.find(name=cur_node)

            # The edge's source and target are not always defined in the order
            # that we need for our shortest path traversal in undirected graphs.
            # If the source of this edge matches the current node we are at, then
            # use the target of this edge as the target of this step.
            if srcnode.index == edge.source:
                tgtnode = self.G.vs[edge.target]
            # If the source of this edge doesn't match the current node, that means
            # that the current node is the target of this edge. Thus we use the
            # source of this edge as the target of this step.
            else:
                assert srcnode.index == edge.target
                tgtnode = self.G.vs[edge.source]

            # Convert the source and target for this step into lat/lon coordinates
            srcloc = (srcnode['lng'], srcnode['lat'])
            tgtloc = (tgtnode['lng'], tgtnode['lat'])

            # Add this step to the list of steps on the route
            intersections = [{'location': srcloc}, {'location': tgtloc}]
            step_dict['intersections'] = intersections
            route['steps'].append(step_dict)

            # Set the current node to the target of this step, so that it is
            # the source of the next step.
            cur_node = tgtnode['name']

        # Assign route weight after finishing steps in the case that tt is not pre-drawn and so
        # it is computed while the steps dict is being populated
        route['weight'] = route['duration']

        # If the path is of 0 length then we add a dummy step for staying in place at the origin
        if len(path) == 0:
            srcnode = self.G.vs.find(onodename)
            tgtnode = self.G.vs.find(dnodename)
            assert srcnode == tgtnode

            srcloc = (srcnode['lng'], srcnode['lat'])
            tgtloc = (tgtnode['lng'], tgtnode['lat'])

            step_dict = dict()
            step_dict['distance'] = 0.0
            step_dict['duration'] = 0.0
            step_dict['mean_duration'] = 0.0
            step_dict['weight'] = 0.0

            intersections = [{'location': srcloc}, {'location': tgtloc}]
            step_dict['intersections'] = intersections
            route['steps'].append(step_dict)

        return route


    # get the distance of the best route from origin to destination
    def get_distance(self, olng, olat, dlng, dlat):
        onodename = str((olng, olat))
        dnodename = str((dlng, dlat))

        path = self.G.get_shortest_paths(onodename, to=dnodename, weights='ttmean', output='epath')[0]

        total_distance = 0
        for edge_index in path:
            total_distance += self.G.es[edge_index]['length']

        return total_distance

    # get the expected duration of the best route (by modal TT) from origin to destination
    # this function returns only an expected TT - use get_routing to calculate a realized TT
    def get_duration(self, olng, olat, dlng, dlat):
        onodename = str((olng, olat))
        dnodename = str((dlng, dlat))
        path = self.G.get_shortest_paths(onodename, to=dnodename, weights='ttmean', output='epath')[0]

        duration = 0
        for edge_index in path:
            duration += self.G.es[edge_index]['ttmean']

        return duration

    # get both distance and duration
    def get_distance_duration(self, olng, olat, dlng, dlat, euclidean=False):
        if euclidean:
            gc_dist = (6371000*2*math.pi/360 * np.sqrt( (math.cos((olat+dlat)*math.pi/360)*(olng-dlng))**2 + (olat-dlat)**2))
            return gc_dist, gc_dist / self.cst_speed
        else:
            return self.get_distance(olng, olat, dlng, dlat), self.get_duration(olng, olat, dlng, dlat)