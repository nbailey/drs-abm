import csv
import time

import gurobipy as gp
import igraph as ig
import numpy as np

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

    def optimizeAssignment(self, vehs, reqs, G, T=0):
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
        # rvGraph.write("output/graphs/rv-assignment-{}.gml".format(T), "gml")

        print("  Generating RTV graph for T={}".format(T))
        t = time.time()
        rtvGraph = self.generateRTVGraph(rvGraph, G, T)
        rtvTime = time.time() - t
        print("  RTV Graph successfully generated in {:.2f} seconds! |V|={}, |E|={}".format(rtvTime, len(rtvGraph.vs), len(rtvGraph.es)))
        # rtvGraph.write("output/graphs/rtv-assignment-{}.gml".format(T), "gml")

        if len(rtvGraph.vs) > 0 and len(rtvGraph.es) > 0:
            t = time.time()
            assignment = self.assignFromRTVGraph(rtvGraph)
            assTime = time.time() - t
            print("  Assignment completed in {:.2f} seconds!".format(assTime))
            return assignment
        else:
            return list(), list()

    def generatePairwiseGraph(self, vehs, reqs, G):
        pass

    def generateRTVGraph(self, rvGraph, G):
        pass

    def assignFromRTVGraph(rtvGraph):
        pass


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
            if req.assigned is not None and not req.assigned:
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
                                    Ced=req.Ced, Cld=req.Cld)

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


class FlowNetworkMatchingAssignment(AssignmentMethod):
    def optimizeAssignment(self, vehs, reqs, G_road, edge_time_array, N_scenarios, weights=[1,1,1], seed=None, T=0, title="flow-network-T0"):
        print("  Generating flow network for T={}".format(T))
        t = time.time()
        G_flow, arc_paths = self.generateFlowNetwork(vehs, reqs, G_road)
        flowTime = time.time() - t
        print("  Flow network successfully generated  in {:.2f} seconds! |V|={}, |E|={}".format(flowTime, len(G_flow.vs), len(G_flow.es)))
        G_flow.write("output/graphs/{}-flow-graph-{}.gml".format(title, T), "gml")
        print("    Written to output/graphs/{}-rv-assignment-{}.gml".format(title, T))

        initSoln = self.findInitialFeasibleSolution(G_flow, G_road, T)

        # G_pruned, updatedInitSoln = self.pruneFlowNetwork(G_flow, arc_paths, edge_time_array, initSoln)
        G_pruned = self.pruneFlowNetwork(G_flow, arc_paths, edge_time_array, initSoln)
        # for edge in G_flow.es:
        #     edge["pruned"] = False
        print("  {} edges pruned from flow network in pre-processing".format(len(G_pruned.es.select(pruned=True))))

        t = time.time()
        assignment = self.bendersDecomposition(G_pruned, N_scenarios, arc_paths, edge_time_array, initSoln, weights, seed, title)
        # assignment = self.bendersDecomposition(G_flow, N_scenarios, arc_paths, edge_time_array, initSoln, weights, seed, title)
        assTime = time.time() - t

        print("  Assignment completed in {:.2f} seconds!".format(assTime))

        return assignment

    def generateFlowNetwork(self, vehs, reqs, G_road, T=0):
        pass

    def pruneFlowNetwork(self, G_flow, arc_paths, edge_time_array):
        pass

    def findInitialFeasibleSolution(self, G_flow):
        pass

    def assignFromFlowNetwork(self, G_flow, N_scenarios, arc_paths, edge_time_array, initSoln, seed=None, T=0):
        pass


class MinDelayFlowMatching(FlowNetworkMatchingAssignment):
    def generateFlowNetwork(self, vehs, reqs, G_road, T=0):
        G_flow = ig.Graph(directed=True)

        pax_rid_map = dict()
        wait_rid_map = dict()

        # Add a node to the flow network for each vehicle's starting location
        for veh in vehs:
            pax_rids, wait_rids = veh.get_passenger_requests()

            pax_reqs = [reqs[rid] for rid in pax_rids]
            for pax_rid in pax_rids:
                pax_rid_map[pax_rid] = veh.id

            wait_reqs = [reqs[rid] for rid in wait_rids]
            for wait_rid in wait_rids:
                # If multiple assignment is added, will need to change this to list or something
                wait_rid_map[wait_rid] = veh.id

            veh_loc, t = veh.get_next_node()
            G_flow.add_vertex("veh_{}_start".format(veh.id),
                              vtype="vehicle",
                              dummy=False,
                              vid=veh.id,
                              nid=veh.id,
                              capacity=veh.K,
                              location=(veh_loc[0], veh_loc[1]),
                              lng=veh_loc[0], lat=veh_loc[1],
                              t=T+t,
                              pax_reqs=pax_reqs,
                              wait_reqs=wait_reqs,
                              Ce=-1, Cl=-1,
                              pod=0)

        # Add two nodes for each request to the flow network: one for its origin and
        # one for its destination
        for req in reqs:
            if req.id in pax_rid_map.keys():
                # If the request has already been picked up, we use a dummy node 
                vid = pax_rid_map[req.id]
                veh = G_flow.vs.find("veh_{}_start".format(vid))
                G_flow.add_vertex("req_{}_pickup_dummy".format(req.id),
                                  vtype="origin",
                                  dummy=True,
                                  rid=req.id,
                                  nid=req.id+len(vehs),
                                  location=veh["location"],
                                  lng=veh["lng"], lat=veh["lat"],
                                  veh_assigned=pax_rid_map[req.id],
                                  Ce=-1, Cl=-1,
                                  pod=1)
            elif req.id in wait_rid_map.keys():
                G_flow.add_vertex("req_{}_pickup".format(req.id),
                                  vtype="origin",
                                  dummy=False,
                                  rid=req.id,
                                  nid=req.id+len(vehs),
                                  location=(req.olng, req.olat),
                                  lng=req.olng, lat=req.olat,
                                  veh_assigned=wait_rid_map[req.id],
                                  Ce=req.Cep-T, Cl=req.Clp-T,
                                  pod=1)
            else:
                G_flow.add_vertex("req_{}_pickup".format(req.id),
                                  vtype="origin",
                                  dummy=False,
                                  rid=req.id,
                                  nid=req.id+len(vehs),
                                  location=(req.olng, req.olat),
                                  lng=req.olng, lat=req.olat,
                                  veh_assigned=None,
                                  Ce=req.Cep-T, Cl=req.Clp-T,
                                  pod=1)

            G_flow.add_vertex("req_{}_dropoff".format(req.id),
                              vtype="destination",
                              dummy=False,
                              rid=req.id,
                              nid=req.id+len(reqs)+len(vehs),
                              location=(req.dlng, req.dlat),
                              lng=req.dlng, lat=req.dlat,
                              Ce=req.Ced-T, Cl=req.Cld-T,
                              pod=-1)

        G_flow.add_vertex("veh_sink_dummy",
                          vtype="sink",
                          dummy=True,
                          nid=2*len(reqs)+len(vehs),
                          Ce=-1, Cl=-1,
                          pod=0)

        # Create a list of edge information containing the source and target for each edge,
        # its length, and the road network path used to construct it.
        edgeList = list()

        # Create a list of lists to later be converted into a numpy array, where each row
        # indicates which road network edges are used in the shortest path for each flow
        # network arc
        pathRows = list()

        # Create a dictionary containing the final origin dummy associated with each vehicle,
        # from which edges will be drawn to connect to the rest of the network
        final_pre_dummies = dict()

        new_req_origins = G_flow.vs.select(vtype="origin", dummy=False)
        req_destinations = G_flow.vs.select(vtype="destination")

        # Add the info for all edges originating from each vehicle vertex to the edgeList
        for vehNode in G_flow.vs.select(vtype="vehicle"):
            # Vehicles with no passengers already riding inside them have an edge directly
            # to the vehicle sink dummy node
            if len(vehNode["pax_reqs"]) == 0:
                edgeInfo = (vehNode["name"], "veh_sink_dummy", [], 0, 0)
                edgeList.append(edgeInfo)

                vehStartNodeName = vehNode["name"]
            # Vehicles with passengers inside have serial edges connecting them to each
            # of their assigned passengers. The final passenger in this sequence has its
            # origin dummy node become a proxy for the vehicle starting location.
            else:
                for i, req in enumerate(vehNode["pax_reqs"]):
                    if i == 0:
                        v1 = vehNode["name"]
                    else:
                        v1 = "req_{}_pickup_dummy".format(vehNode["pax_reqs"][i-1].id)

                    v2 = "req_{}_pickup_dummy".format(req.id)
                    edgeInfo = (v1, v2, [], 0, 0)
                    edgeList.append(edgeInfo)

                vehStartNodeName = v2

            vehStartNode = G_flow.vs.find(vehStartNodeName)

            # If the vehicle has any passengers already, its starting location has an edge
            # connecting to each passenger's dropoff location
            for pax_req in vehNode["pax_reqs"]:
                paxNodeName = "req_{}_dropoff".format(pax_req.id)
                paxReqNode = G_flow.vs.find(paxNodeName)
                edgeList.append(getEdgeInfoFromRoadGraph(vehStartNode, paxReqNode, G_road))

            # All vehicles have an edge connecting their starting location to each request's
            # origin that is not already a passenger of a vehicle (the new_req_origins)
            for oReqNode in new_req_origins:
                edgeList.append(getEdgeInfoFromRoadGraph(vehStartNode, oReqNode, G_road))

        # Each origin node for requests that aren't already passengers of a vehicle has
        # edges connecting it to each other non-passenger origin and each request destination
        for oReqNode in new_req_origins:
            for oReqNode2 in new_req_origins:
                if oReqNode.index == oReqNode2.index:
                    continue
                edgeList.append(getEdgeInfoFromRoadGraph(oReqNode, oReqNode2, G_road))

            for dReqNode in req_destinations:
                edgeList.append(getEdgeInfoFromRoadGraph(oReqNode, dReqNode, G_road))

                # Additionally, each destination has an edge back to each origin location
                # of requests besides its own (since you can't visit a dropoff location before
                # the corresponding pickup location)
                if dReqNode["rid"] != oReqNode["rid"]:
                    edgeList.append(getEdgeInfoFromRoadGraph(dReqNode, oReqNode, G_road))

        # Each destination node has an edge to each other destination node, and to the
        # vehicle sink dummy node
        for dReqNode in req_destinations:
            for dReqNode2 in req_destinations:
                if dReqNode.index == dReqNode2.index:
                    continue
                edgeList.append(getEdgeInfoFromRoadGraph(dReqNode, dReqNode2, G_road))

            edgeInfo = (dReqNode, "veh_sink_dummy", [], 0, 0)
            edgeList.append(edgeInfo)

        # Using the information from each edge collected above, add those edges to the flow
        # network and add a row containing information on the road network path to the pathRows matrix
        for edgeInfo in edgeList:
            oNodeName = edgeInfo[0]
            dNodeName = edgeInfo[1]
            path = edgeInfo[2]
            path_length = edgeInfo[3]
            path_ttmedian = edgeInfo[4]

            row = np.zeros(len(G_road.es))
            row[path] = 1
            pathRows.append(row)

            G_flow.add_edges([(oNodeName, dNodeName),], {"length": path_length, "ttmedian": path_ttmedian})

        return G_flow, np.matrix(pathRows)

    def pruneFlowNetwork(self, G_flow, arc_paths, edge_time_array, initSoln=None):
        arc_times = arc_paths * edge_time_array.T
        min_arc_times = [np.min(times) for times in arc_times]
        flow_edge_times = edge_time_array * arc_paths.T

        if initSoln is not None:
            initFlows = initSoln[0]
            initOrder = initSoln[1]
            initFlowSet = set([flow[0] for flow in initFlows])
        else:
            initFlowSet = set()

        G_flow.es["min_time"] = min_arc_times
        G_flow.vs["min_arr_time"] = [0] * len(G_flow.vs)

        V_O = G_flow.vs.select(vtype="origin")
        V_D = G_flow.vs.select(vtype="destination")
        vehs = G_flow.vs.select(vtype="vehicle")
        dummy_O = G_flow.vs.select(vtype="origin", dummy=True)
        vehs_and_dummies = set(vehs) | set(dummy_O)

        # in_edges = {v.index: [e.index for e in G_flow.es.select(_target=v.index)] for v in G_flow.vs}
        # veh_in_edges = {v.index: [e.index for e in G_flow.es.select(_source_in=set([veh.index for veh in vehs_and_dummies]), _target=v.index)] for v in G_flow.vs}

        # for v in V_O:
        #     e_in_veh = veh_in_edges[v.index]
        #     v["min_arr_time"] = np.min(G_flow.es[e_in_veh]["min_time"])

        # for v in V_D:
        #     d_name = v["name"].split("_")
        #     o_name = d_name.copy()
        #     o_name[2] = "pickup"
        #     v_o = G_flow.vs.find(name="_".join(o_name))
        #     o_d_edge = G_flow.es.select(_source=v_o.index, _target=v.index)
        #     v["min_arr_time"] = v_o["min_arr_time"] + o_d_edge["min_time"]

        removal_set = set()

        # for v in V_D:
        #     d_name = v["name"].split("_")
        #     o_name = d_name.copy()
        #     o_name[2] = "pickup"
        #     v_o = G_flow.vs.find(name="_".join(o_name))

        #     e_in = in_edges[v.index]
        #     for eid in e_in:
        #         edge = G_flow.es[eid]
        #         src = G_flow.vs[edge.source]
        #         if src.index == v_o.index:
        #             continue
        #         o_src_edge = G_flow.es.find(_source=v_o.index, _target=src.index)

        #         if v_o["min_arr_time"] + o_src_edge["min_time"] + edge["min_time"] > v["Cl"]:
        #             removal_set.add(eid)

        # for v in V_O:
        #     e_in = in_edges[v.index]
        #     for eid in e_in:
        #         edge = G_flow.es[eid]
        #         src = G_flow.vs[edge.source]

        #         if src["min_arr_time"] + edge["min_time"] > v["Cl"]:
        #             removal_set.add(eid)

        G_pruned = G_flow.copy()
        G_pruned.es["pruned"] = [False,] * len(G_pruned.es)
        for eid in removal_set - initFlowSet:
            G_pruned.es[eid]["pruned"] = True

        return G_pruned

    def findInitialFeasibleSolution(self, G_flow, G_road, T=0):
        vehNodes = G_flow.vs.select(vtype="vehicle")
        reqONodes = G_flow.vs.select(vtype="origin")
        reqDNodes = G_flow.vs.select(vtype="destination")
        allReqNodes = set(reqONodes) | set(reqDNodes)
        sinkNode = G_flow.vs.find(vtype="sink")
        sinkId = sinkNode["nid"]

        veh_flows = {veh["vid"]: list() for veh in vehNodes}
        veh_routes = {veh["vid"]: list() for veh in vehNodes}
        veh_req_nodes = {veh["vid"]: list() for veh in vehNodes}

        # Create an initial path for each vehicle from its starting node through
        # each assigned request's origins and destinations (if applicable), and
        # terminating at the vehicle sink dummy node.
        for vehNode in vehNodes:
            vid = vehNode["vid"]

            if len(vehNode["pax_reqs"]) == 0:
                edge = G_flow.es.find(_source=vehNode.index, _target=sinkNode.index)
                veh_flows[vid].append(edge.index)

            else:
                for i, req in enumerate(vehNode["pax_reqs"]):
                    if i == 0:
                        v1 = G_flow.vs.find(name=vehNode["name"])
                    else:
                        v1 = G_flow.vs.find(name="req_{}_pickup_dummy".format(vehNode["pax_reqs"][i-1].id))

                    v2 = G_flow.vs.find("req_{}_pickup_dummy".format(req.id))

                    edge = G_flow.es.find(_source=v1.index, _target=v2.index)
                    veh_flows[vid].append(edge.index)
                    veh_req_nodes[vid].append(v2)

                unassigned_reqs = set(vehNode["pax_reqs"])
                current_loc = G_flow.vs[G_flow.es[veh_flows[vid][-1]].target]
                while len(unassigned_reqs) > 0:
                    dReq = 1e10
                    bestReq = None
                    bestReqNode = None
                    bestEdge = None
                    for req in unassigned_reqs:
                        reqDname = "req_{}_dropoff".format(req.id)
                        reqDNode = G_flow.vs.find(name=reqDname)
                        edge = G_flow.es.find(_source=current_loc.index, _target=reqDNode.index)
                        dEdge = edge["length"]
                        if dEdge < dReq:
                            dReq = dEdge
                            bestReq = req
                            bestReqNode = reqDNode
                            bestEdge = edge

                    assert bestEdge is not None
                    veh_flows[vid].append(edge.index)
                    veh_req_nodes[vid].append(bestReqNode)
                    current_loc = G_flow.vs[bestEdge.target]
                    unassigned_reqs.remove(bestReq)

                edge = G_flow.es.find(_source=current_loc.index, _target=sinkNode.index)
                veh_flows[vid].append(edge.index)

        # For each request, compute an optimal routing for each vehicle if this request's origin and
        # destination node were added to the vehicle's route using a Tabu search. Assign the request to
        # the vehicle with the lowest-cost route.
        for reqONode in reqONodes:
            if reqONode["dummy"]:
                continue

            req = reqONode["rid"]
            reqDNode = G_flow.vs.find(nid=reqONode["nid"] + len(reqONodes))

            bestCost = 1e10
            bestVeh = None
            bestRoute = None

            for veh in vehNodes:
                vid = veh["vid"]
                loc_nodes = veh_req_nodes[vid] + [reqONode, reqDNode]

                # if len(veh_req_nodes[vid]) >= 4:
                #     continue

                startLoc, stopLocs = createLocationsFromFlowNodes(veh, loc_nodes)

                cost, route = tabuSearchPath(stopLocs, G_road, T, startLoc, veh["capacity"], len(veh["pax_reqs"]), maxTabuSize=(3*(len(stopLocs)+2)))

                if cost < bestCost:
                    bestCost = cost
                    bestVid = vid
                    bestRoute = route

            veh_req_nodes[bestVid] += [reqONode, reqDNode]
            veh_routes[bestVid] = bestRoute

        veh_flows = getFlowsFromRoutes(veh_routes, G_flow)

        # print(veh_routes)
        # print({vid: [(G_flow.vs[G_flow.es[edge].source]["name"], G_flow.vs[G_flow.es[edge].target]["name"]) for edge in veh_flows[vid]] for vid in veh_flows.keys()})
        # print({vid: [v["name"] for v in veh_req_nodes[vid]] for vid in veh_req_nodes.keys()})

        flow_keys = list()
        req_veh_order = dict()
        for veh in veh_flows.keys():
            veh_flow = veh_flows[veh]
            order = len(veh_flow)
            req_veh_order[(veh, veh)] = order

            for edge in veh_flow:
                flow_keys.append((edge, veh))
                tgtNode = G_flow.vs[G_flow.es[edge].target]
                # if tgtNode["vtype"] in ("origin", "destination"):
                order -= 1
                req_veh_order[(tgtNode["nid"], veh)] = order

        return flow_keys, req_veh_order

    def createUpperLevelProblem(self, G_flow, weights, N_scenarios, scenario_edge_times, initSoln=None, heuristic_cutoff_flows=None):
        m = gp.Model("routing-upper-level-problem")
        m.setParam(GRB.Param.LogToConsole, 0)

        # Create sets of nodes by vertex type for easy reference when generating constraints
        vehNodes = G_flow.vs.select(vtype="vehicle")
        reqONodes = G_flow.vs.select(vtype="origin")
        reqDNodes = G_flow.vs.select(vtype="destination")
        allReqNodes = set(reqONodes) | set(reqDNodes)
        nonVehNodes = G_flow.vs.select(vtype_ne="vehicle")
        nonSinkNodes = G_flow.vs.select(vtype_ne="sink")
        sinkNode = G_flow.vs.find(vtype="sink")
        sinkId = sinkNode["nid"]

        # Create sets of node IDs by vertex type
        K = [veh["nid"] for veh in vehNodes] # Vehicles
        V_O = [reqO["nid"] for reqO in reqONodes] # Origins
        V_D = [reqD["nid"] for reqD in reqDNodes] # Destinations
        V = [req["nid"] for req in allReqNodes] # All request locations (Os + Ds)

        # The number of requests in the system. For an origin node index i, i+N_R is the corresponding destination node
        N_R = len(V_O)

        # For easy iteration, create a set of edge indices coupled with the corresponding node id of the source i and target j of edge e=(i,j)
        EIJ = [(edge.index, G_flow.vs[edge.source]["nid"], G_flow.vs[edge.target]["nid"]) for edge in G_flow.es if not edge["pruned"]]
        
        # Create a dict for easy lookup of all edges entering (in_edges) or exiting (out_edges) from a given node ID
        out_edges = gp.tupledict({vertex["nid"]: [edge.index for edge in G_flow.es.select(_source=vertex.index, pruned=False)] for vertex in G_flow.vs})
        in_edges = gp.tupledict({vertex["nid"]: [edge.index for edge in G_flow.es.select(_target=vertex.index, pruned=False)] for vertex in G_flow.vs})

        # Create dicts for the capacity of each vehicle and the amount of space that each request takes up when picked up
        veh_capacities = gp.tupledict({veh["nid"]: veh["capacity"] for veh in vehNodes})
        node_occupancies = gp.tupledict({vertex["nid"]: vertex["pod"] for vertex in G_flow.vs})

        # The maximum capacity of any vehicle in the network
        max_capacity = np.max(list(veh_capacities.values()))

        # Create a dict for the latest acceptable arrival time for each request location, after which it is considered delayed and counts towards objective function
        latest_dropoffs = gp.tupledict({vertex["nid"]: vertex["Cl"] for vertex in G_flow.vs.select(vtype_in=("origin", "destination"))})

        # Create a dict for each request-vehicle combination that indicates whether that vehicle has already been assigned to that request in a prior iteration
        former_assignments = gp.tupledict({(req["nid"], veh["nid"]): req["rid"] in set([wait_req.id for wait_req in veh["wait_reqs"]]) for req in reqONodes for veh in vehNodes})

        # A dictionary for whether it is possible for a vehicle k to flow on an edge e (for instance, veh 0 can never travel from veh 1's origin to any other node)
        possible_flow = gp.tupledict({(edge.index, veh["nid"]): (G_flow.vs[edge.source]["vtype"] != "vehicle" or G_flow.vs[edge.source]["nid"] == veh["nid"]) for edge in G_flow.es for veh in vehNodes})
        for e, k in heuristic_cutoff_flows:
            possible_flow[e, k] = False
 
        # Lists of variable indices
        x_ek = gp.tuplelist([(edge.index, veh["nid"]) for edge in G_flow.es for veh in vehNodes if possible_flow[edge.index, veh["nid"]] and not edge["pruned"]]) # Also used for phi_ek
        # p_ik = gp.tuplelist([(node["nid"], veh["nid"]) for node in G_flow.vs for veh in vehNodes]) # Descending order
        # p_i = gp.tuplelist([node["nid"] for node in G_flow.vs]) # Descending order
        n_ik = gp.tuplelist([(node["nid"], veh["nid"]) for node in nonSinkNodes for veh in vehNodes]) # Also used for Delta_ik

        edge_dist = gp.tupledict({edge.index: np.round(edge["length"], 3) for edge in G_flow.es if not edge["pruned"]})

        veh_edge_flow = m.addVars(x_ek, vtype=GRB.BINARY, name="x") # 1 if vehicle k traverses edge e, 0 otherwise
        # delay_multiplier = m.addVars(x_ek, vtype=GRB.CONTINUOUS, name="phi", lb=0) # How many times edge e's travel time counts towards delay heuristics
        # req_order = m.addVars(p_i, vtype=GRB.CONTINUOUS, name="P", lb=0) # Descending order of location i's order in its assigned vehicle's route
        # req_veh_order = m.addVars(p_ik, vtype=GRB.CONTINUOUS, name="P", lb=0) # Descending order of location i's order in vehicle k's route
        # veh_occupancy = m.addVars(n_ik, vtype=GRB.CONTINUOUS, name="n", lb=0) # How many passengers are in vehicle k after it services location i
        req_veh_newass = m.addVars(n_ik, vtype=GRB.BINARY, name="Delta") # 1 if request i is newly assigned to vehicle k in this assignment, 0 otherwise
        theta = m.addVars(gp.tuplelist([i for i in range(N_scenarios)]), vtype=GRB.CONTINUOUS, name="theta", lb=0) # Delay in scenario s \in {1,...,N_scenarios}
        veh_delay = m.addVars(gp.tuplelist([(veh["nid"], s) for veh in vehNodes for s in range(N_scenarios)]), vtype=GRB.CONTINUOUS, name="z", lb=0)

        model_var_dict = {
            "x": veh_edge_flow,
            # "phi": delay_multiplier,
            # "P": req_order,
            # "P": req_veh_order,
            # "n": veh_occupancy,
            "Delta": req_veh_newass,
            "theta": theta,
            "z": veh_delay,
        }

        # Add flow constraints:
        # 1) Total flow out of vehicle start node by that vehicle = 1
        m.addConstrs((gp.quicksum([veh_edge_flow[e, k] for e in out_edges[k]]) == 1 for k in K), "veh_start_flow")
        # 2) Total flow into sink node by each vehicle = 1
        m.addConstrs((gp.quicksum([veh_edge_flow[e, k] for e in in_edges[sinkId] if possible_flow[e, k]]) == 1 for k in K), "veh_sink_flow")
        # 3) Total inflow and outflow for each node by any vehicle must be equal
        m.addConstrs((gp.quicksum([veh_edge_flow[e, k] for e in in_edges[i] if possible_flow[e, k]]) - sum([veh_edge_flow[e, k] for e in out_edges[i] if possible_flow[e, k]]) == 0 for i in V for k in K),
            "flow_conservation")
        # 4) Flow into a pickup node must be accompanied by flow out of the corresponding destination node
        m.addConstrs((sum([veh_edge_flow[e, k] for e in in_edges[i] if possible_flow[e, k]]) - sum([veh_edge_flow[e, k] for e in out_edges[i+N_R] if possible_flow[e, k]]) == 0 for i in V_O for k in K),
            "pickup_dropoff_flow")

        # # Add order constraints (recall that order variables are descending along route):
        # # 1) The order of a downstream node in a vehicle's route must be 1 less than the order of the upstream node if the vehicle flows along the edge
        # m.addConstrs((2*N_R*(veh_edge_flow[e, k] - 1) + req_veh_order[getTargetNodeId(G_flow, e), k] - req_veh_order[i, k] <= -1 for e, i, j in EIJ for k in K if possible_flow[e, k]),
        #     "req_veh_flow_order_ub")
        # m.addConstrs((2*N_R*(1 - veh_edge_flow[e, k]) + req_veh_order[getTargetNodeId(G_flow, e), k] - req_veh_order[i, k] >= -1 for e, i, j in EIJ for k in K if possible_flow[e, k]),
        #     "req_veh_flow_order_lb")
        # # 2) The order variable for a dropoff location in vehicle's route must be less than the order of the pickup location if the vehicle services that pickup location
        # m.addConstrs((req_veh_order[i, k] - req_veh_order[i+N_R,k] >= gp.quicksum([veh_edge_flow[e, k] for e in in_edges[i] if possible_flow[e, k]]) for i in V_O for k in K),
        #     "dropoff_after_pickup")
        # # 3) The order of the sink node is always 0 for each vehicle
        # m.addConstrs((req_veh_order[sinkId, k] == 0 for k in K), "veh_end_loc_order")
        # # 4) If a vehicle does not flow into a location, then that location has order 0 in the vehicle's route
        # m.addConstrs((req_veh_order[i, k] <= 2*N_R*gp.quicksum([veh_edge_flow[e, k] for e in in_edges[i] if possible_flow[e, k]]) for i in V for k in K), "no_order_without_node_visit")

        # # Add order constraints (recall that order variables are descending along route):
        # # 1) The order of a downstream node in a vehicle's route must be 1 less than the order of the upstream node if the vehicle flows along the edge
        # m.addConstrs((2*N_R*(veh_edge_flow.sum(e, "*") - 1) + req_order[getTargetNodeId(G_flow, e)] - req_order[i] <= -1 for e, i, j in EIJ),
        #     "req_flow_order_ub")
        # m.addConstrs((2*N_R*(1 - veh_edge_flow.sum(e, "*")) + req_order[getTargetNodeId(G_flow, e)] - req_order[i] >= -1 for e, i, j in EIJ),
        #     "req_flow_order_lb")
        # # 2) The order variable for a dropoff location in vehicle's route must be less than the order of the pickup location (other constraints ensure it is same vehicle)
        # m.addConstrs((req_order[i] - req_order[i+N_R] >= 1 for i in V_O), "dropoff after pickup")
        # # 3) The order of the sink node is always 0 
        # m.addConstr(req_order[sinkId] == 0, "end_loc_order")

        # # Add capacity constraints:
        # # 1) The occupancy of a vehicle at a downstream node is equal to the occupancy at the upstream node plus the passenger change in occupancy at the downstream node if the vehicle flows along that edge
        # # m.addConstrs((veh_capacities[k]*(veh_edge_flow[e, k] - 1) + veh_occupancy[getTargetNodeId(G_flow, e), k] - veh_occupancy[i, k] <= node_occupancies[getTargetNodeId(G_flow, e)] for k in K for i in V for e in out_edges[i] if possible_flow[e, k] and getTargetNodeId(G_flow, e) != sinkId),
        # #     "pod_occupancy_ub")
        # m.addConstrs((veh_capacities[k]*(1 - veh_edge_flow[e, k]) + veh_occupancy[getTargetNodeId(G_flow, e), k] - veh_occupancy[i, k] >= node_occupancies[getTargetNodeId(G_flow, e)] for k in K for i in V for e in out_edges[i] if possible_flow[e, k] and getTargetNodeId(G_flow, e) != sinkId),
        #     "pod_occupancy_lb")
        # # 2) The vehicle occupancy must always be less than or equal to the vehicle capacity at each node
        # m.addConstrs((veh_occupancy[i, k] <= veh_capacities[k] for i in V for k in K), "veh_capacity_constraint")

        # Exactly one vehicle must be assigned to each request
        m.addConstrs((gp.quicksum([veh_edge_flow.sum(e, "*") for e in in_edges[i]]) == 1 for i in V_O), "exactly_one_assignment")

        # Any request assigned to a vehicle that wasn't a former assignment counts as a new assignment
        m.addConstrs((req_veh_newass[i, k] + former_assignments[i, k] >= gp.quicksum([veh_edge_flow[e, k] for e in in_edges[i] if possible_flow[e, k]]) for i in V_O for k in K),
            "added_veh_assignments")
        # # A vehicle cannot flow directly from its origin to the sink node AND have new assignments
        # m.addConstrs((gp.quicksum([veh_edge_flow[e, k] for e in out_edges[k] if getTargetNodeId(G_flow, e) == sinkId]) + (1/(2*N_R))*req_veh_newass.sum("*", k) <= 1 for k in K),
        #     "prevent_cycles")

        # # The delay multiplier for an edge/vehicle pair is equal to the order of the downstream location in the vehicle's route as long as the vehicle traverses edge e, 0 otherwise
        # # m.addConstrs((delay_multiplier[e, k] >= req_veh_order[j, k] - 2*N_R*(1-veh_edge_flow[e, k]) for e, _, j in EIJ for k in K if possible_flow[e, k]),
        # #     "delay_multiplier_from_flow_and_order")
        # m.addConstrs((delay_multiplier[e, k] >= req_order[j] - 2*N_R*(1 - veh_edge_flow[e, k]) for e, _, j in EIJ for k in K if possible_flow[e, k]),
        #     "delay_multiplier_from_flow_and_order")
        # # Create a heuristic lower bound for the delay by taking the delay multipliers times the edges' travel times for all traversed edges and subtracting the time constraints for each node
        # m.addConstrs((theta[s] >= gp.quicksum([delay_multiplier.sum(e.index, "*")*scenario_edge_times[s][e.index] for e in G_flow.es]) - latest_dropoffs.sum() for s in range(N_scenarios)),
        #     "delay_lb_from_flow")

        # m.addConstrs((theta[s] >= gp.quicksum([veh_edge_flow.sum(e.index, "*")*scenario_edge_times[s][e.index] for e in G_flow.es if not e["pruned"]]) - \
        #               gp.quicksum([veh_edge_flow.sum(e, "*")*latest_dropoffs[i] for e, i, j in EIJ if i in V_D and j == sinkId]) for s in range(N_scenarios)),
        #              "delay_heuristic_lb")

        m.addConstrs((veh_delay[k, s] >= gp.quicksum([veh_edge_flow[e, k]*scenario_edge_times[s][e] for e, _, _ in EIJ if possible_flow[e, k]]) - \
                      gp.quicksum([veh_edge_flow[e, k]*latest_dropoffs[i] for e, i, j in EIJ if possible_flow[e, k] and i in V and j == sinkId]) for k in K for s in range(N_scenarios)),
                     "scenario_veh_delay_heuristic")
        m.addConstrs((theta[s] >= veh_delay.sum("*", s) for s in range(N_scenarios)), "delay_heuristic_lb")


        # Optional heuristic to constrain routes to 2 request locations long at most to prevent searching inefficient routings
        m.addConstrs((req_veh_newass.sum("*", k) <= 2 for k in K),
            "heuristic_limit_two_new_assignments")

        distWeight = weights[0] # in meters
        newAssWeight = weights[1] # in num. of requests
        delayWeight = weights[2] # in seconds

        # Objective is a weighted combination of the total distance traveled by vehicles, the number of new assignments, and the average incurred delay across all scenarios
        m.setObjective(distWeight*gp.quicksum([veh_edge_flow.sum(e.index, "*")*edge_dist[e.index] for e in G_flow.es if not e["pruned"]]) + \
                       newAssWeight*req_veh_newass.sum() + \
                       delayWeight*theta.sum()/N_scenarios,
                       GRB.MINIMIZE)

        # Initialize a solution for the solver to use as a start
        if initSoln is not None:
            initFlows = initSoln[0]
            initOrder = initSoln[1]

            for e, k in veh_edge_flow.keys():
                if (e, k) in initFlows:
                    veh_edge_flow[e, k].start = 1
                    # m.addConstr(veh_edge_flow[flow] == 1, "fix_initial_flow_soln[{}, {}]".format(flow[0], flow[1]))

                    # delay_multiplier[e, k].start = initOrder[getTargetNodeId(G_flow, e), k]
                else:
                    veh_edge_flow[e, k].start = 0
                    # delay_multiplier[e, k].start = 0

            # for i, k in req_veh_order.keys():
            #     if (i, k) in initOrder.keys():
            #         req_veh_order[i, k].start = initOrder[i, k]
            #         # m.addConstr(req_veh_order[order] == initOrder[order], "fix_initial_order_soln[{}, {}]".format(order[0], order[1]))
            #     else:
            #         req_veh_order[i, k].start = 0

            for s in range(N_scenarios):
                theta[s].start = np.sum([np.rint(np.asscalar(scenario_edge_times[s][e])) for e, _ in initFlows])
            #     print("Theta[{}] start value: {:.2f}".format(s, np.sum([np.rint(np.asscalar(scenario_edge_times[s][e])) for e, _ in initFlows])))
            #     # m.addConstr(theta[s] == np.sum([np.rint(np.asscalar(scenario_edge_times[s][e])) for e, _ in initFlows]), "fix_theta[{}]".format(s))

        m.update()

        return m, model_var_dict


    def createSubproblemWithoutObjective(self, G_flow, subproblem_idx, heuristic_cutoff_flows=list()):
        m = gp.Model("delay-subproblem-given-routing-{}".format(subproblem_idx))
        m.setParam(GRB.Param.LogToConsole, 0)

        vehNodes = G_flow.vs.select(vtype="vehicle")
        reqONodes = G_flow.vs.select(vtype="origin")
        reqDNodes = G_flow.vs.select(vtype="destination")
        allReqNodes = set(reqONodes) | set(reqDNodes)
        sinkNode = G_flow.vs.find(vtype="sink")

        K = [veh["nid"] for veh in vehNodes]
        V = [req["nid"] for req in allReqNodes]
        EIJ = [(edge.index, G_flow.vs[edge.source]["nid"], G_flow.vs[edge.target]["nid"]) for edge in G_flow.es if not edge["pruned"]]

        out_edges = gp.tupledict({vertex["nid"]: [edge.index for edge in G_flow.es.select(_source=vertex.index, pruned=False)] for vertex in G_flow.vs})
        in_edges = gp.tupledict({vertex["nid"]: [edge.index for edge in G_flow.es.select(_target=vertex.index, pruned=False)] for vertex in G_flow.vs})

        pi_ek = gp.tuplelist([(edge.index, veh["nid"]) for edge in G_flow.es for veh in vehNodes if edge.target is not sinkNode.index])
        lam_i = gp.tuplelist([vertex["nid"] for vertex in allReqNodes])

        edge_time_duals = m.addVars(pi_ek, vtype=GRB.CONTINUOUS, name="pi{}".format(subproblem_idx), lb=0)
        node_delay_duals = m.addVars(lam_i, vtype=GRB.CONTINUOUS, name="lambda{}".format(subproblem_idx), lb=0, ub=1)
        possible_flow = gp.tupledict({(edge.index, veh["nid"]): (G_flow.vs[edge.source]["vtype"] != "vehicle" or G_flow.vs[edge.source]["nid"] == veh["nid"]) for edge in G_flow.es for veh in vehNodes if not edge["pruned"]})
        for e, k in heuristic_cutoff_flows:
            possible_flow[e, k] = False

        m.addConstrs((gp.quicksum([edge_time_duals[e, k] for e in out_edges[i] if possible_flow[e, k] and G_flow.es[e].target is not sinkNode.index]) - gp.quicksum([edge_time_duals[e, k] for e in in_edges[i] if possible_flow[e, k]]) + node_delay_duals[i] >= 0 for i in V for k in K),
            "edge_travel_time_dual_constraint_{}".format(subproblem_idx))

        m.update()

        return m, edge_time_duals, node_delay_duals


    def updateAndSolveSubproblem(self, G_flow, s, pi, lam, scenario_times, veh_flow, upper_vars):
        vehNodes = G_flow.vs.select(vtype="vehicle")
        sinkNode = G_flow.vs.find(vtype="sink")
        K = [veh["nid"] for veh in vehNodes]
        V = [req["nid"] for req in G_flow.vs.select(vtype_in=("origin", "destination"))]
        EIJ = [(edge.index, G_flow.vs[edge.source]["nid"], G_flow.vs[edge.target]["nid"]) for edge in G_flow.es if not edge["pruned"]]
        in_edges = gp.tupledict({vertex["nid"]: [edge.index for edge in G_flow.es.select(_target=vertex.index, pruned=False)] for vertex in G_flow.vs})
        latest_dropoffs = gp.tupledict({vertex["nid"]: vertex["Cl"] for vertex in G_flow.vs.select(vtype_in=("origin", "destination"))})

        num_constrs = len(s.getConstrs())

        s.setObjective(gp.quicksum([pi[e, k]*(np.asscalar(scenario_times[e]) - (1e6 * (1 - np.rint(veh_flow[e, k])))) for e, _, _  in EIJ for k in K if (e, k) in veh_flow.keys() and G_flow.es[e].target is not sinkNode.index]) - \
            gp.quicksum([lam[i]*latest_dropoffs[i] for i in V]), GRB.MAXIMIZE)

        pi_restrictors = s.addConstrs((pi[e, k] <= len(V)*np.rint(veh_flow[e, k]) for e, _, _ in EIJ for k in K if (e, k) in veh_flow.keys() and G_flow.es[e].target is not sinkNode.index), "pi_flow_limit")

        s.update()
        # s.write("output/gurobi/subproblem-model.lp")

        s.optimize()

        assert s.status == GRB.OPTIMAL

        pi_opt = s.getAttr("x", pi)
        lambda_opt = s.getAttr("x", lam)

        constrained_eids = {(eid, vid): pi_opt[eid, vid] for eid, vid in pi_opt.keys() if pi_opt[eid, vid] > 0.5}
        constrained_lambdas = {nid: lambda_opt[nid] for nid in lambda_opt.keys() if lambda_opt[nid] > 0.5}

        constraint_generators = list()

        # Identify which edges in constrained_eids have vehicle node at source, and select all other edges with:
        #   - Source = a different vehicle node
        #   - Target = the same node as this edge
        #   - Scenario travel time >= this edge's scenario travel time
        vids = set([v.index for v in vehNodes])
        veh_source_edges = G_flow.es.select(_source_in=vids)

        veh_node_outflows = {v["nid"]: e.index for v in vehNodes for e in veh_source_edges if (e.index, v["nid"]) in veh_flow.keys() and np.isclose(veh_flow[(e.index, v["nid"])], 1.0)}

        for eid, vid in constrained_eids.keys():
            original_edge = G_flow.es[eid]
            original_veh = G_flow.vs.find(nid=vid)
            original_out_edges = {e.target: e.index for e in G_flow.es.select(_source=original_veh.index)}

            if original_edge in veh_source_edges:
                same_target_edges = veh_source_edges.select(_target=original_edge.target, _source_ne=original_edge.source)
                for new_edge in same_target_edges:
                    new_veh = G_flow.vs[new_edge.source]
                    swap_out_edge = G_flow.es[veh_node_outflows[new_veh["nid"]]]
                    swap_in_edge = G_flow.es[original_out_edges[swap_out_edge.target]]

                    if np.asscalar(scenario_times[new_edge.index]) > np.asscalar(scenario_times[original_edge.index]) and np.asscalar(scenario_times[swap_in_edge.index]) > np.asscalar(scenario_times[swap_out_edge.index]):
                        swap_constraint = constrained_eids.copy()

                        new_nid = new_veh["nid"]
                        old_nid = original_veh["nid"]
                        new_flow_keys = list()

                        swap_constraint[(swap_out_edge.index, new_nid)] = 0
                        swap_constraint[(new_edge.index, new_nid)] = pi_opt[(original_edge.index, old_nid)]
                        new_flow_keys.append((new_edge.index, new_nid))

                        swap_constraint[(original_edge.index, old_nid)] = 0
                        swap_constraint[(swap_in_edge.index, old_nid)] = pi_opt[(swap_out_edge.index, old_nid)]
                        new_flow_keys.append((swap_in_edge.index, old_nid))

                        for e, k in constrained_eids.keys():
                            if e in (original_edge.index, new_edge.index, swap_out_edge.index, swap_in_edge.index):
                                continue

                            if k == old_nid:
                                swap_constraint[(e, new_nid)] = pi_opt[(e, k)]
                                swap_constraint[(e, old_nid)] = 0
                                new_flow_keys.append((e, new_nid))
                            elif k == new_nid:
                                swap_constraint[(e, old_nid)] = pi_opt[(e, k)]
                                swap_constraint[(e, new_nid)] = 0
                                new_flow_keys.append((e, old_nid))

                        if all([(e, k) in veh_flow.keys() for e, k in new_flow_keys]):
                            constraint_generators.append((swap_constraint, constrained_lambdas))

            if G_flow.vs[original_edge.source]["nid"] in V and G_flow.vs[original_edge.target]["nid"] in V:
                original_prev_edges = [e for e in G_flow.es.select(_target=original_edge.source) if (e.index, vid) in veh_flow.keys() and np.isclose(veh_flow[e.index, vid], 1.0)]
                assert len(original_prev_edges) == 1
                original_prev_edge = original_prev_edges[0]

                try:
                    swap_prev_edge = G_flow.es.find(_source=original_prev_edge.source, _target=original_edge.target)
                    swap_edge = G_flow.es.find(_source=original_edge.target, _target=original_edge.source)

                    if (swap_prev_edge.index, vid) in veh_flow.keys() and (swap_edge.index, vid) in veh_flow.keys() and np.asscalar(scenario_times[swap_prev_edge.index]) >= np.asscalar(scenario_times[original_prev_edge.index]) + np.asscalar(scenario_times[original_edge.index]):
                        swap_constraint = constrained_eids.copy()

                        swap_constraint[(original_prev_edge.index, vid)] = 0
                        swap_constraint[(original_edge.index, vid)] = 0
                        swap_constraint[(swap_prev_edge.index, vid)] = pi_opt[(original_prev_edge.index, vid)]
                        swap_constraint[(swap_edge.index, vid)] = pi_opt[(original_edge.index, vid)]

                        constraint_generators.append((swap_constraint, constrained_lambdas))

                except:
                    pass

        for k in K:
            if pi_opt.sum("*", k).getValue() > 0:
                veh_only_constraint_pis = constrained_eids.copy()
                veh_rem_constraint_pis = constrained_eids.copy()

                for eid, vid in constrained_eids.keys():
                    if vid != k:
                        veh_only_constraint_pis[eid, vid] = 0
                    else:
                        veh_rem_constraint_pis[eid, vid] = 0

                veh_only_constraint_lambdas = constrained_lambdas.copy()
                veh_rem_constraint_lambdas = constrained_lambdas.copy()

                for rid in constrained_lambdas.keys():
                    if all([not np.isclose(veh_flow[e, k], 1.0) for e in in_edges[rid] if (e ,k) in veh_flow.keys()]):
                        veh_only_constraint_lambdas[rid] = 0
                    else:
                        veh_rem_constraint_lambdas[rid] = 0

                constraint_generators.append((veh_only_constraint_pis, veh_only_constraint_lambdas))
                constraint_generators.append((veh_rem_constraint_pis, veh_rem_constraint_lambdas))

        if len(constrained_eids) > 0:
            primary_constr = gp.quicksum([constrained_eids[e, k] * (np.asscalar(scenario_times[e]) - (1e6 * (1 - upper_vars["x"][e, k]))) for e, k in constrained_eids.keys()]) - \
                gp.quicksum([constrained_lambdas[i] * latest_dropoffs[i] for i in constrained_lambdas.keys()])

        else:
            primary_constr = gp.LinExpr(0.0)

        secondary_constrs = list()
        for generator_pis, generator_lambdas in constraint_generators:
            constr = gp.quicksum([generator_pis[e, k] * (np.asscalar(scenario_times[e]) - (1e6 * (1 - upper_vars["x"][e, k]))) for e, k in generator_pis.keys()]) - \
                gp.quicksum([generator_lambdas[i] * latest_dropoffs[i] for i in generator_lambdas.keys()])
            secondary_constrs.append(constr)

        return primary_constr, secondary_constrs, pi_restrictors


    def createCombinatorialSubproblem(self, G_flow, heuristic_cutoff_flows=list()):
        m = gp.Model("combinatorial-subproblem-given-routing")
        m.setParam(GRB.Param.LogToConsole, 0)

        vehNodes = G_flow.vs.select(vtype="vehicle")
        reqONodes = G_flow.vs.select(vtype="origin")
        reqDNodes = G_flow.vs.select(vtype="destination")
        allReqNodes = set(reqONodes) | set(reqDNodes)
        sinkNode = G_flow.vs.find(vtype="sink")

        K = [veh["nid"] for veh in vehNodes]
        V = [req["nid"] for req in allReqNodes]
        V_O = [req["nid"] for req in reqONodes]
        V_D = [req["nid"] for req in reqDNodes]
        EIJ = [(edge.index, G_flow.vs[edge.source]["nid"], G_flow.vs[edge.target]["nid"]) for edge in G_flow.es if not edge["pruned"]]

        # The number of requests in the system. For an origin node index i, i+N_R is the corresponding destination node
        N_R = len(V_O)

        out_edges = gp.tupledict({vertex["nid"]: [edge.index for edge in G_flow.es.select(_source=vertex.index, pruned=False)] for vertex in G_flow.vs})
        in_edges = gp.tupledict({vertex["nid"]: [edge.index for edge in G_flow.es.select(_target=vertex.index, pruned=False)] for vertex in G_flow.vs})

        possible_flow = gp.tupledict({(edge.index, veh["nid"]): (G_flow.vs[edge.source]["vtype"] != "vehicle" or G_flow.vs[edge.source]["nid"] == veh["nid"]) for edge in G_flow.es for veh in vehNodes if not edge["pruned"]})
        for e, k in heuristic_cutoff_flows:
            possible_flow[e, k] = False

        u_e = gp.tuplelist([edge.index for edge in G_flow.es])
        v_ek = gp.tuplelist([(edge.index, veh["nid"]) for edge in G_flow.es for veh in vehNodes if possible_flow[edge.index, veh["nid"]] and edge.target is not sinkNode.index])
        g_i = gp.tuplelist([req["nid"] for req in reqONodes])
        h_ik = gp.tuplelist([(req["nid"], veh["nid"]) for req in allReqNodes for veh in vehNodes])

        order_lb_duals = m.addVars(u_e, vtype=GRB.CONTINUOUS, name="u1", lb=0)
        order_ub_duals = m.addVars(u_e, vtype=GRB.CONTINUOUS, name="u2", lb=0)
        occupancy_duals = m.addVars(v_ek, vtype=GRB.CONTINUOUS, name="v", lb=0)
        route_sequence_duals = m.addVars(g_i, vtype=GRB.CONTINUOUS, name="g", lb=0)
        capacity_duals = m.addVars(h_ik, vtype=GRB.CONTINUOUS, name="h", lb=0)

        m.update()

        var_dict = dict()
        var_dict["u1"] = order_lb_duals
        var_dict["u2"] = order_ub_duals
        var_dict["v"] = occupancy_duals
        var_dict["g"] = route_sequence_duals
        var_dict["h"] = capacity_duals

        m.addConstrs((gp.quicksum([order_ub_duals[e] - order_lb_duals[e] for e in out_edges[i]]) + gp.quicksum([order_lb_duals[e] - order_ub_duals[e] for e in in_edges[i]]) + route_sequence_duals[i] == 0 for i in V_O),
            "origin_dual_constraint")
        m.addConstrs((gp.quicksum([order_ub_duals[e] - order_lb_duals[e] for e in out_edges[i]]) + gp.quicksum([order_lb_duals[e] - order_ub_duals[e] for e in in_edges[i]]) - route_sequence_duals[i-N_R] == 0 for i in V_D),
            "destination_dual_constraint")
        m.addConstrs((gp.quicksum([occupancy_duals[e, k] for e in in_edges[i] if possible_flow[e, k]]) - gp.quicksum([occupancy_duals[e, k] for e in out_edges[i] if getTargetNodeId(G_flow, e) is not sinkNode.index]) - capacity_duals[i, k] == 0 for i in V for k in K),
            "vehicle_capacity_dual_constraint")
        m.addConstr(gp.quicksum([order_lb_duals[e] for e in in_edges[sinkNode["nid"]]]) == 0,
            "sink_edges_dont_count_lb")
        m.addConstr(gp.quicksum([order_ub_duals[e] for e in in_edges[sinkNode["nid"]]]) == 0,
            "sink_edges_dont_count_ub")

        m.update()

        m.setParam(GRB.Param.DualReductions, 0)

        return m, var_dict


    def updateAndSolveCombinatorialSubproblem(self, G_flow, s, lower_vars, veh_flow, upper_vars):
        m = gp.Model("combinatorial-subproblem-given-routing")
        m.setParam(GRB.Param.LogToConsole, 0)

        vehNodes = G_flow.vs.select(vtype="vehicle")
        reqONodes = G_flow.vs.select(vtype="origin")
        reqDNodes = G_flow.vs.select(vtype="destination")
        allReqNodes = set(reqONodes) | set(reqDNodes)
        sinkNode = G_flow.vs.find(vtype="sink")

        K = [veh["nid"] for veh in vehNodes]
        V = [req["nid"] for req in allReqNodes]
        V_O = [req["nid"] for req in reqONodes]
        V_D = [req["nid"] for req in reqDNodes]
        EIJ = [(edge.index, G_flow.vs[edge.source]["nid"], G_flow.vs[edge.target]["nid"]) for edge in G_flow.es if not edge["pruned"]]

        # The number of requests in the system. For an origin node index i, i+N_R is the corresponding destination node
        N_R = len(V_O)

        out_edges = gp.tupledict({vertex["nid"]: [edge.index for edge in G_flow.es.select(_source=vertex.index, pruned=False)] for vertex in G_flow.vs})
        in_edges = gp.tupledict({vertex["nid"]: [edge.index for edge in G_flow.es.select(_target=vertex.index, pruned=False)] for vertex in G_flow.vs})

        # Create dicts for the capacity of each vehicle and the amount of space that each request takes up when picked up
        veh_capacities = gp.tupledict({veh["nid"]: veh["capacity"] for veh in vehNodes})
        node_occupancies = gp.tupledict({vertex["nid"]: vertex["pod"] for vertex in G_flow.vs})

        # The maximum capacity of any vehicle in the network
        max_capacity = np.max(list(veh_capacities.values()))

        order_lb_duals = lower_vars["u1"]
        order_ub_duals = lower_vars["u2"]
        occupancy_duals = lower_vars["v"]
        route_sequence_duals = lower_vars["g"]
        capacity_duals = lower_vars["h"]

        order_lb_obj_sum = gp.quicksum([order_lb_duals[e]*(-1 - 2*N_R*(1-gp.quicksum([np.rint(veh_flow[e, k]) for k in K if (e, k) in veh_flow.keys()]))) for e, _, _ in EIJ])
        order_ub_obj_sum = gp.quicksum([order_ub_duals[e]*(1 - 2*N_R*(1-gp.quicksum([np.rint(veh_flow[e, k]) for k in K if (e, k) in veh_flow.keys()]))) for e, _, _ in EIJ])
        occupancy_sum = gp.quicksum([occupancy_duals[e, k]*(-node_occupancies[j] - max_capacity*(1-np.rint(veh_flow[e, k]))) for e, _, j in EIJ for k in K if j != sinkNode.index and (e, k) in veh_flow.keys()])
        route_sequenece_sum = route_sequence_duals.sum()
        capacity_sum = -1*gp.quicksum([capacity_duals[i, k]*veh_capacities[k] for i in V for k in K])

        objective_func = order_lb_obj_sum + order_ub_obj_sum + occupancy_sum + route_sequenece_sum + capacity_sum

        s.setObjective(objective_func, GRB.MAXIMIZE)

        feas_constr = s.addConstr(objective_func == 1, "enforce_feasibility")

        s.update()
        # s.write("output/gurobi/combinatorial-subproblem.lp")

        s.optimize()

        constrs = list()

        if s.status == GRB.INFEASIBLE:
            pass
        else:
            assert s.status == GRB.OPTIMAL

            opt_vars = dict()
            for var_name in lower_vars.keys():
                opt_vars[var_name] = s.getAttr("x", lower_vars[var_name])

            inf_lb_indices = [e.index for e in G_flow.es if not np.isclose(opt_vars["u1"][e.index], 0.0)]
            inf_ub_indices = [e.index for e in G_flow.es if not np.isclose(opt_vars["u2"][e.index], 0.0)]
            inf_occupancy_indices = [(e.index, k) for e in G_flow.es for k in K if (e.index, k) in opt_vars["v"].keys() and not np.isclose(opt_vars["v"][e.index, k], 0.0)]

            # assert all([np.isclose(veh_flow.sum(e, "*"), 1.0) for e in inf_lb_indices])
            # assert all([np.isclose(veh_flow.sum(e, "*"), 1.0) for e in inf_ub_indices])
            # assert all([np.isclose(veh_flow[e, k], 1.0) for e, k in inf_occupancy_indices])

            # print("Infeasible order indices:")
            # print([e for e in set(inf_lb_indices).union(set(inf_ub_indices))])
            # print("Infeasible capacity indices:")
            # print([(e, k) for e, k in inf_occupancy_indices])

            if len(inf_lb_indices) > 0 or len(inf_ub_indices) > 0:
                constrs.append(gp.quicksum([(1 - upper_vars["x"].sum(e, "*")) for e in set(inf_lb_indices).union(set(inf_ub_indices))]))

            if len(inf_occupancy_indices) > 0:
                constrs.append(gp.quicksum([(1 - upper_vars["x"][e, k]) for e, k in inf_occupancy_indices]))

        return constrs, feas_constr


    def bendersDecomposition(self, G_flow, N_scenarios, arc_paths, edge_time_array, initSoln=None, weights=[1, 1, 1], seed=None, title="", T=0):
        rs = np.random.RandomState(seed)

        flow_edge_times = edge_time_array * arc_paths.T # Or something like this...
        scenario_samples = rs.choice(list(range(len(flow_edge_times))), N_scenarios)
        print("Sampled scenarios: {}".format(", ".join([str(sample_num) for sample_num in scenario_samples])))

        min_edge_times = [np.min(flow_edge_times[scenario_samples, e.index]) for e in G_flow.es]
        scenario_edge_times = {s: flow_edge_times[int(scenario_samples[s]), :].T for s in range(N_scenarios)}

        numReqs = len(G_flow.vs.select(vtype="origin"))
        vehNodes = G_flow.vs.select(vtype="vehicle")

        print("Preliminary heuristic deletion of edges:")
        heuristic_cutoff_flows = list()
        # num_heuristic_constraints = 0
        t = time.time()
        # for e in G_flow.es:
        #     if G_flow.vs[e.source]["vtype"] == "vehicle" or G_flow.vs[e.target]["vtype"] == "sink" or G_flow.vs[e.source]["rid"] == G_flow.vs[e.target]["rid"] or e["pruned"]:
        #         continue
            
        #     veh_edge_min_delays = dict()
        #     for veh in vehNodes:
        #         k = veh.index
        #         subgraph, target_edge, edge_map = createSubgraphFromEdge(G_flow, e.index, k)
        #         subgraph_times = [np.min(flow_edge_times[:, edge_map[eid]]) for eid in range(len(subgraph.es))]

        #         veh_edge_min_delays[k] = optimalRouteOnSubgraph(subgraph, target_edge, subgraph_times, numReqs)

        #     cutoff_value = np.percentile(list(veh_edge_min_delays.values()), 75)
        #     high_delay_vehs = [k for k in veh_edge_min_delays.keys() if veh_edge_min_delays[k] > cutoff_value]

        #     heuristic_cutoff_flows.append((e.index, k))

        #     # heuristic_constraints = upper_level.addConstrs((veh_edge_flow[e.index, k] == 0 for k in high_delay_vehs), "heuristic_delay_cutoff")
        #     # num_heuristic_constraints += len(heuristic_constraints)

        heuristicTime = time.time() - t

        print("Added {} heuristic constraints in {:.2f} seconds".format(len(heuristic_cutoff_flows), heuristicTime))

        # Initialize upper-level problem
        upper_level, upper_vars = self.createUpperLevelProblem(G_flow, weights, N_scenarios, scenario_edge_times, initSoln, heuristic_cutoff_flows)
        
        veh_edge_flow = upper_vars["x"]
        theta = upper_vars["theta"]

        comb_subproblem, comb_sub_vars = self.createCombinatorialSubproblem(G_flow, heuristic_cutoff_flows)

        # Initialize a lower-level subproblem for each scenario to be evaluated
        scenario_dict = dict()
        for scenarioNum in range(N_scenarios):
            scenario_edge_times = flow_edge_times[int(scenario_samples[scenarioNum]), :]
            subproblem, pi, lam = self.createSubproblemWithoutObjective(G_flow, scenarioNum, heuristic_cutoff_flows)
            scenario_dict[scenarioNum] = {"scenario_id": scenario_samples[scenarioNum],
                                          "times": scenario_edge_times,
                                          "model": subproblem,
                                          "vars": {"pi": pi, "lambda": lam},
                                          "constraints": list()}

        self.benders_iter = 0
        upper_level.setParam(GRB.Param.LazyConstraints, 1)
        upper_level.setParam(GRB.Param.IntegralityFocus, 1)
        upper_level.setParam(GRB.Param.MIPFocus, 2)
        # upper_level.setParam(GRB.Param.MIPGap, 0.95)
        # upper_level.setParam(GRB.Param.Cuts, 2)
        # upper_level.setParam(GRB.Param.Heuristics, 0.01)
        upper_level.setParam(GRB.Param.Presolve, 2)
        upper_level.setParam(GRB.Param.PreSparsify, 1)
        upper_level.setParam(GRB.Param.SolFiles, "output/gurobi/{}-inter".format(title))
        upper_level.setParam(GRB.Param.ResultFile, "output/gurobi/{}-final.sol".format(title))
        # upper_level.setParam(GRB.Param.TimeLimit, 4500)

        # sol_routing_file = "output/{}-solution-progress-routing.csv".format(title)
        # routing_headers = ["Experiment Info", "Objective Value"]
        # for i in range(len(G_flow.vs.select(vtype="vehicle"))):
        #     routing_headers.append("Veh_{}_Reqs".format(i))
        # with open(sol_routing_file, "w", newline="") as outcsv:
        #     writer = csv.writer(outcsv)
        #     writer.writerow(routing_headers)

        def add_benders_cuts(model, where):
            if where == GRB.Callback.MIPSOL:
                sol_flow = model.cbGetSolution(veh_edge_flow)
                sol_theta = model.cbGetSolution(theta)

                all_sol_vals = model.cbGetSolution(model.getVars())
                sol_obj_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)

                print("  Optimal solution reached (Obj. Value {:.2f})".format(sol_obj_value))

                sol_routing = convertFlowAssignmentToRouting(G_flow, sol_flow)
                print("    " + "\n    ".join(["Vehicle {}: {}".format(veh, ", ".join([str(x) for x in sol_routing[veh]])) for veh in sol_routing.keys()]))

                sub_t = time.time()

                print("  Adding feasibility cuts")
                comb_subproblem.setParam(GRB.Param.LogFile, "output/gurobi/{}-combinatorial-subproblem-{}.txt".format(title, self.benders_iter))
                num_comb_sub_constrs = len(comb_subproblem.getConstrs())
                comb_constrs, feas_sub_constr = self.updateAndSolveCombinatorialSubproblem(G_flow, comb_subproblem, comb_sub_vars, sol_flow, upper_vars)

                added_comb_constrs = 0
                added_theta_constrs = 0

                for constr in comb_constrs:
                    model.cbLazy(constr >= 1)
                    added_comb_constrs += 1

                comb_subproblem.remove(feas_sub_constr)
                comb_subproblem.update()
                assert len(comb_subproblem.getConstrs()) == num_comb_sub_constrs

                if added_comb_constrs == 0:
                    print("  Adding optimality cuts", end="", flush=True)

                    # constrs = list()
                    sub_obj_value = 0

                    for scenarioNum in range(N_scenarios):
                        scenario_info = scenario_dict[scenarioNum]
                        subproblem = scenario_info["model"]
                        scenario_times = scenario_info["times"].T
                        pi = scenario_info["vars"]["pi"]
                        lam = scenario_info["vars"]["lambda"]

                        subproblem.setParam(GRB.Param.LogFile, "output/gurobi/{}-subproblems-{}.txt".format(title, self.benders_iter))

                        num_sub_constrs = len(subproblem.getConstrs())
                        primary_constr, secondary_constrs, added_sub_constrs = self.updateAndSolveSubproblem(G_flow, subproblem, pi, lam, scenario_times, sol_flow, upper_vars)

                        print(".", end="", flush=True)

                        model.cbLazy(theta[scenarioNum] >= primary_constr)
                        scenario_info["constraints"].append(primary_constr)
                        added_theta_constrs += 1

                        for constr in secondary_constrs:
                            model.cbLazy(theta[scenarioNum] >= constr)
                            scenario_info["constraints"].append(constr)
                            added_theta_constrs += 1

                        sub_obj_value += subproblem.getObjective().getValue() / N_scenarios
                        subproblem.remove(added_sub_constrs)
                        subproblem.update()
                        assert len(subproblem.getConstrs()) == num_sub_constrs

                sub_time = time.time() - sub_t
                print("\n    Subproblems solved in {:.1f} seconds ({} constraints added)".format(sub_time, added_comb_constrs+added_theta_constrs))

                if added_comb_constrs == 0:
                    print("    Calculated objective value for solution: {:.2f}".format(sub_obj_value))

                    avg_delay, scenario_delay = calcDelayFromSolution(G_flow, sol_flow, scenario_dict, T)
                    print("    Manually calculated delay at current solution: {:.2f}".format(avg_delay))

                    # routing_row = [title, sub_obj_value]
                    # if len(sol_routing) > 0:
                    #     for vid in range(len(G_flow.vs.select(vtype="vehicle"))):
                    #         veh_route = list()

                    #         if vid in sol_routing.keys():
                    #             route = sol_routing[vid]

                    #             for rid, pod, _, _ in route:
                    #                 veh_route.append((rid, pod))

                    #         routing_row.append(veh_route)

                    # with open(sol_routing_file, "a+", newline="") as outcsv:
                    #     writer = csv.writer(outcsv)
                    #     writer.writerow(routing_row)

                self.benders_iter += 1

        upper_level.setParam(GRB.Param.LogFile, "output/gurobi/{}-upper-level-log.txt".format(title))
        upper_level.write("output/gurobi/{}-upper-level-model.lp".format(title))
        upper_level.optimize(add_benders_cuts)

        if upper_level.status == GRB.OPTIMAL or upper_level.status == GRB.TIME_LIMIT:

            veh_edges = upper_level.getAttr('x', veh_edge_flow)
            veh_routes = convertFlowAssignmentToRouting(G_flow, veh_edges)

        else:

            upper_level.computeIIS()
            upper_level.write("output/gurobi/{}-infeasible-model.ilp".format(title))

            veh_routes = dict()

        return veh_routes, list()