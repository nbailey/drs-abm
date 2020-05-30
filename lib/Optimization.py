import gurobipy as gp
import igraph as ig
import numpy as np
import time

from gurobipy import GRB
from lib.Constants import *
from lib.optimUtils import *


class AssignmentMethod():
    """
    A class for optimization methods that assign requests to vehicles in a Model (see Agents.py)

    Contains the following methods:
    - optimizeAssignment(vehs, reqs, G): Takes a set of vehicles in the Model, requests that need
        to be assigned, and the road network, and returns an optimal assignment, formatted as a
        dict of vehicles with their assigned requests and optimal route to serve said requests.
    """

    def __init__(self, params):
        self.parameters = params

    def optimizeAssignment(self, vehs, reqs, G):
        pass

class VehReqMatchingAssignment(AssignmentMethod):
    """
    A class for assignment methods using a 4-step process to generate a bipartite graph to
    optimally match vehicles with sets of requests. Based on Alonso-Mora et al. (2018)

    Contains the following methods:
    - generatePairwiseGraph(vehs, reqs, G): Takes a set of vehicles, requests, and the road
        network, and returns a graph that connects individual requests with vehicles (determining
        whether that vehicle can serve that request) and requests with other requests (determining
        whether those requests can be pooled).
    - generateRTVGraph(reqVehGraph, rvGraph, G): Takes the graph returned by the previous function
        and returns a combined graph where passengers connect to pools of passengers that they can
        be grouped into, and vehicles connect to pools of passengers that they can serve
    - assignFromRTVGraph(rtvGraph): Takes a RTV graph returned by the previous step and returns
        an optimal matching.
    """

    def optimizeAssignment(self, vehs, reqs, G, T=0):
        print("  Generating R-V graph for T={}".format(T))
        t = time.time()
        rvGraph = self.generatePairwiseGraph(vehs, reqs, G, T)
        rvTime = time.time() - t
        print("  R-V Graph successfully generated  in {:.2f} seconds! |V|={}, |E|={}".format(rvTime, len(rvGraph.vs), len(rvGraph.es)))
        rvGraph.write("output/graphs/rv-assignment-{}.gml".format(T), "gml")

        print("  Generating RTV graph for T={}".format(T))
        t = time.time()
        rtvGraph = self.generateRTVGraph(rvGraph, G, T)
        rtvTime = time.time() - t
        print("  RTV Graph successfully generated in {:.2f} seconds! |V|={}, |E|={}".format(rtvTime, len(rtvGraph.vs), len(rtvGraph.es)))
        rtvGraph.write("output/graphs/rtv-assignment-{}.gml".format(T), "gml")

        t = time.time()
        assignment = self.assignFromRTVGraph(rtvGraph)
        assTime = time.time() - t
        print("  Assignment completed in {:.2f} seconds!".format(assTime))
        return assignment

    def generatePairwiseGraph(self, vehs, reqs, G):
        pass

    def generateRTVGraph(self, rvGraph, G):
        pass

    def assignFromRTVGraph(rtvGraph):
        pass


class MOTAP(VehReqMatchingAssignment):
    """
    An assignment method that (M)aximizes (O)n-(T)ime (A)rrival (P)robability
    """

    def generatePairwiseGraph(self, vehs, reqs, G):
        rvGraph = ig.Graph()
        return rvGraph

    def generateRTVGraph(self, reqVehGraph, rvGraph, G):
        rtvGraph = ig.Graph()
        return rtvGraph


class AlonsoMora(VehReqMatchingAssignment):
    """
    The assignment method described in Alonso-Mora et al. (2018)
    """

    def __init__(self, params):
        if "method" in params.keys():
            self.method = params["method"]
        else:
            self.method = "tabu"

        if "tabuMaxTime" in params.keys():
            self.tabuMaxTime = params["tabuMaxTime"]
        else:
            self.tabuMaxTime = 3

    def generatePairwiseGraph(self, vehs, reqs, G, T=0):
        rvGraph = ig.Graph()
        for req in reqs:
            if not req.assigned:
                # Create a vertex representing this request with
                # all relevant information needed for matching
                # so that the reqs don't have to keep getting referenced
                dropoff_time_window = req.get_dropoff_time_window()
                rvGraph.add_vertex("req_{}".format(req.id),
                                    vtype="request",
                                    rid=req.id,
                                    origin=(req.olng, req.olat),
                                    destination=(req.dlng, req.dlat),
                                    Cep=req.Cep-T, Clp=req.Clp,
                                    Ced=dropoff_time_window[0],
                                    Cld=dropoff_time_window[1])

        for veh in vehs:
            pax_rids, wait_rids = veh.get_passenger_requests()
            pax_reqs = [reqs[rid] for rid in pax_rids]
            wait_reqs = [reqs[rid] for rid in wait_rids]
            veh_loc, t = veh.get_next_node()
            # veh_route = [(leg.rid, leg.pod) for leg in veh.route]
            rvGraph.add_vertex("veh_{}".format(veh.id),
                                vtype="vehicle",
                                vid=veh.id,
                                capacity=veh.K,
                                location=(veh_loc[0], veh_loc[1]),
                                t=t,
                                # route=veh_route,
                                pax_reqs=pax_reqs,
                                wait_reqs=wait_reqs)

        feasCount = 0
        infCount = 0

        # Create edges in the R-R graph if a vehicle starting
        # at one request's origin can pickup and dropoff both
        # requests while obeying all constraints
        for r1 in rvGraph.vs.select(vtype="request"):
            for r2 in rvGraph.vs.select(vtype="request"):
                # Can't pair requests with themselves, and ignore
                # requests we've already found can be paired. Additionally,
                # don't bother with any request where crow's flight distance
                # between request origins exceeds wait time constraints.
                reqsDist = greatCircleDistance(r1["origin"][1], r1["origin"][0], r2["origin"][1], r2["origin"][0])
                reqsMinTime = reqsDist/CST_SPEED
                if r1 == r2 or r2 in r1.neighbors() or T+reqsMinTime > r1["Clp"] or T+reqsMinTime > r2["Clp"]:
                    continue
                # Determine the optimal cost and route for the set of locations
                # (O1, O2, D1, D2) using Tabu search
                if self.method == "tabu":
                    cost, route = tabuOptimalRouting((r1, r2), G, T=T, maxTime=self.tabuMaxTime)
                elif self.method == "enumerate":
                    cost, route = findOptimalRouting((r1, r2), G, T=T)

                # If a feasible path was found then we add an edge between these
                # two requests in the R-R graph and store the path attributes
                if route is not None:
                    rvGraph.add_edge(r1, r2, cost=cost, route=route, etype="req_req")
                    feasCount += 1
                else:
                    infCount += 1

        print("    Feasible pairings found amongst requests: {}".format(feasCount))
        print("    Infeasible pairings evaluated: {}".format(infCount))

        feasCount = 0
        infCount = 0
        savedCount = 0
        skippedCount = 0

        # print("Creating edges between requests and vehicles", end="")
        for req in rvGraph.vs.select(vtype="request"):
            requests = set()
            requests.add(req)

            for veh in rvGraph.vs.select(vtype="vehicle"):
                # Don't bother with any vehicle where the shortest path from the vehicle's
                # current location to the request's origin exceeds wait time constraints
                fastestArrivalTime = T + shortestPathTime(G, veh["location"][1], veh["location"][0], req["origin"][1], req["origin"][0])
                if fastestArrivalTime > req["Clp"]:
                    skippedCount += 1
                    continue

                # Create a fictitious vehicle with same location & capacity
                # as real vehicle but without any previously assigned requests
                # and evaluate whether this vehicle could feasibly serve the
                # request. This serves as a quick feasibility check, since
                # adding requests won't make the path feasible.
                if len(veh["pax_reqs"]) + len(veh["wait_reqs"]) > 0:
                    dummy_veh = {"t": veh["t"],
                                 "location": veh["location"],
                                 "capacity": veh["capacity"],
                                 "pax_reqs": [],
                                 "wait_reqs": []}
                    if self.method == "tabu":
                        _, dummy_route = tabuOptimalRouting(requests, G, dummy_veh, T=T, maxTime=self.tabuMaxTime)
                    elif self.method == "enumerate":
                        _, dummy_route = findOptimalRouting(requests, G, dummy_veh, T=T)

                if len(veh["pax_reqs"]) + len(veh["wait_reqs"]) == 0 or dummy_route is not None:
                    if self.method == "tabu":
                        cost, route = tabuOptimalRouting(requests, G, veh, T=T, maxTime=self.tabuMaxTime)
                    elif self.method == "enumerate":
                        cost, route = findOptimalRouting(requests, G, veh, T=T)
                    # print("    Vehicle {} path computed ({} pax, {} assigned reqs)".format(
                    #     veh["vid"], len(veh["pax_reqs"]), len(veh["wait_reqs"])))

                    if route is not None:
                        rvGraph.add_edge(veh, req, cost=cost, route=route, etype="req_veh")
                        feasCount += 1
                    else:
                        infCount += 1
                else:
                    savedCount += 1

        print("    Feasible pairings found between requests and vehicles: {}".format(feasCount))
        print("    Infeasible pairings evaluated: {}".format(infCount))
        print("    Infeasible pairings skipped via dummy evaluation: {}".format(savedCount))
        print("    Pairings skipped because vehicle was too far away: {}".format(skippedCount))

        return rvGraph

    def generateRTVGraph(self, rvGraph, G, T=0):
        # Copy the vehicle and request information from rvGraph
        # but keep no edge information
        rtvGraph = rvGraph.copy()
        rtvGraph.delete_edges(rtvGraph.es)
        assert len(rtvGraph.es) == 0

        vehs = rvGraph.vs.select(vtype="vehicle")

        feasCount = 0
        infCount = 0
        savedCount = 0

        for veh in vehs:
            # Find the set of requests that this vehicle connects to in the RV graph. Since
            # edges are only drawn between vehs and reqs, each veh's neighbors are all reqs
            veh_reqs = veh.neighbors()

            trips = dict()
            # trips = {req: rvGraph.es.find(_source=veh, _target=req)["cost"] for req in veh_reqs}

            # Add trips of size one
            trips[1] = dict()
            for req in veh_reqs:
                # Which of veh and req is source and which is target seems inconsistent
                try:
                    rvEdge = rvGraph.es.find(_source=req.index, _target=veh.index)
                except ValueError:
                    print("SRC = VEH, TGT = REQ")
                    rvEdge = rvGraph.es.find(_source=veh.index, _target=req.index)

                reqs = rvGraph.vs.select(name=req["name"])

                trips[1][req["name"]] = {"reqs": reqs,
                                         "cost": rvEdge["cost"],
                                         "route": rvEdge["route"]}

            # Add trips of size two
            trips[2] = dict()
            for r1_name in trips[1].keys():
                for r2_name in trips[1].keys():
                    if r1_name == r2_name:
                        continue

                    if not (r2_name, r1_name) in trips[2].keys():
                        reqs = rvGraph.vs.select(name_in=(r1_name, r2_name))
                        if len(reqs.subgraph().es) == 1:
                            if self.method == "tabu":
                                cost, route = tabuOptimalRouting(reqs, G, veh, T=T, maxTime=self.tabuMaxTime)
                            elif self.method == "enumerate":
                                cost, route = findOptimalRouting(reqs, G, veh, T=T)

                            if route is not None:
                                trips[2][(r1_name, r2_name)] = {"reqs": reqs,
                                                                "cost": cost,
                                                                "route": route}
                                feasCount += 1
                            else:
                                infCount += 1

                        else:
                            savedCount += 1

            # Add trips of size three and more
            for k in range(3, veh["capacity"]+1):
                trips[k] = dict()
                # Create a list to keep track of what sets of trips have been tested
                tested_k_trips = list()
                for t1 in trips[k-1].keys():
                    for t2 in trips[k-1].keys():
                        # Create a set with the combined requests of two trips of size k-1
                        trip_reqs = set()
                        for req_name in t1:
                            trip_reqs.add(req_name)
                        for req_name in t2:
                            trip_reqs.add(req_name)

                        # Test if the combined set represents a trip of size k we haven't yet tested
                        if len(trip_reqs) == k and trip_reqs not in tested_k_trips:
                            # Add the request set to the list of tested trips
                            tested_k_trips.append(trip_reqs)

                            # Check whether each subset of k-1 requests is a feasible trip
                            trip_subsets = list()
                            for req_name in trip_reqs:
                                # Create a set of all request IDs save one in this trip
                                trips_without_req = trip_reqs.copy()
                                trips_without_req.remove(req_name)

                                # Check the keys of trips[k-1] to see if all request IDs in
                                # the set without 1 request IDs are featured in any single key
                                trip_keys = [trip for trip in trips[k-1].keys() if all(
                                    [req in trip for req in trips_without_req])]
                                # Record whether such a trip of size k-1 exists or not
                                trip_subsets.append(len(trip_keys) == 1)

                            # If any subset trip of size k-1 is infeasible, then the full
                            # trip is guaranteed to be infeasible as well
                            if not all(trip_subsets):
                                savedCount += 1
                                break

                            # If all subsets are feasible, then try to find an optimal
                            # routing for the combined trip of size k
                            reqs = rvGraph.vs.select(name_in=trip_reqs)
                            if len(reqs.subgraph().es) == len(reqs)*(len(reqs)-1)/2.0:
                                if self.method == "tabu":
                                    cost, route = tabuOptimalRouting(reqs, G, veh, T=T, maxTime=self.tabuMaxTime)
                                elif self.method == "enumerate":
                                    cost, route = findOptimalRouting(reqs, G, veh, T=T)

                                if route is not None:
                                    trips[k][tuple(trip_reqs)] = {
                                        "reqs": reqs,
                                        "cost": cost,
                                        "route": route}
                                    feasCount += 1
                                else:
                                    infCount += 1
                            else:
                                savedCount += 1

            for k in trips:
                for t in trips[k]:
                    trip_info = trips[k][t]
                    rtvGraph, trip = addTripToRTVGraph(rtvGraph, trip_info, veh)

        print("    Feasible trip assignments evaluated: {}".format(feasCount))
        print("    Infeasible trip assignments evaluated: {}".format(infCount))
        print("    Infeasible trip evaluations skipped via incomplete subgraphs / previous trip length checks: {}".format(savedCount))
        return rtvGraph

    def assignFromRTVGraph(self, rtvGraph):
        m = gp.Model("rtv-assignment")
        # m.setParam(GRB.Param.OutputFlag, 0)
        m.setParam(GRB.Param.LogToConsole, 0)
        m.setParam(GRB.Param.LogFile, "output/gurobi/log.txt")

        e_ij = gp.tuplelist([(edge.source, edge.target) for edge in rtvGraph.es.select(etype="veh_trip")])
        x_k = gp.tuplelist([(v.index) for v in rtvGraph.vs.select(vtype="request")])

        req_trips = gp.tupledict({
            req.index: [rtvGraph.es[edge].target for edge in rtvGraph.incident(req, ig.ALL)]
            for req in rtvGraph.vs.select(vtype="request")})

        cost_ij = gp.tupledict({(edge.source, edge.target): edge["cost"] for edge in rtvGraph.es.select(etype="veh_trip")})
        cost_ko = 1e6

        trips = gp.tuplelist([(v.index) for v in rtvGraph.vs.select(vtype="trip")])
        vehs = gp.tuplelist([(v.index) for v in rtvGraph.vs.select(vtype="vehicle")])
        reqs = gp.tuplelist([(v.index) for v in rtvGraph.vs.select(vtype="request")])

        veh_trip_flow = m.addVars(e_ij, vtype=GRB.BINARY, name="e")
        req_ignored = m.addVars(x_k, vtype=GRB.BINARY, name="x")

        m.setObjective(veh_trip_flow.prod(cost_ij) + cost_ko * req_ignored.sum(), GRB.MINIMIZE)
        m.addConstrs((veh_trip_flow.sum(i, "*") <= 1 for i in vehs), "one_veh_per_trip")
        for k in reqs:
            m.addConstr(sum([veh_trip_flow.sum("*", j) for j in req_trips[k]]) + req_ignored[k] == 1,
                "one_veh_max_per_req[{}]".format(k))

        m.optimize()

        if m.status == GRB.OPTIMAL:
            veh_trips = m.getAttr('x', veh_trip_flow)
            veh_routes = dict()
            for edge in rtvGraph.es.select(etype="veh_trip"):
                veh = rtvGraph.vs[edge.source]
                vehID = veh.index
                tripID = edge.target
                if veh_trips[vehID, tripID] > 0:
                    veh_routes[veh["vid"]] = edge["route"]

            ignored_reqs = m.getAttr('x', req_ignored)
            rejected_reqs = set()
            for req in reqs:
                if ignored_reqs[req] > 0:
                    rejected_reqs.add(rtvGraph.vs[req]["rid"])
        else:
            return None, None

        return veh_routes, rejected_reqs