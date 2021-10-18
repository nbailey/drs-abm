import gurobipy as gp
import igraph as ig
import math
import numpy as np
import random
import time

from gurobipy import GRB
from itertools import combinations, permutations


class StopLocation(object):
    '''
    A class to represent a stop location in a vehicle's itinerary
    lat: Location's latitude
    lng: Location's longitude
    rid: The request associated with this location
    order: This location's order in the associated request's itinerary.
           Generally 0 for pickup, 1 for dropoff. But if the request is
           already in the vehicle, then 0 is dropoff since it's the only
           action needed for that request.
    lbT: The earliest feasible time that this location can be arrived at.
    ubT: The latest feasible time that this location can be arrived at.
    pod: The change in vehicle number of passengers associated with this
         location. Usually pickup = +1, dropoff = -1.
    '''

    def __init__(self, lng, lat, rid=None, order=None, lbT=None, ubT=None, pod=None):
        self.lat = lat
        self.lng = lng
        self.rid = rid
        self.order = order
        self.lbT = lbT
        self.ubT = ubT
        self.pod = pod

    def checkTimeConstraints(self, T):
        return ((T >= self.lbT) and (T <= self.ubT))


def permutePassengerOrder(locations, startLoc=None):
    # Find all possible orderings of the pickup and dropoff operations
    # that a ridesharng vehicle needs to do, which can then be
    # iterated through.
    # Arguments:
    # - locations: A set or list of tuples (lat, lng, rid, order) representing
    #              the locations the vehicle must visit, the request ID associated
    #              with each, and the order of that location in that rid's itinerary
    #              0: first action (usually pickup), 1: second action (usually dropoff) etc.
    # - startLoc (optional): A tuple (lat, lng) representing the starting location of the
    #                        vehicle.
    # Returns:
    # - perms: A set of ordered lists of tuples (lat, lng, rid), each list
    #          representing one possible order of visiting the locations
    perms = set()

    # Create a list of request IDs in the vehicle's itinerary
    rids = [location.rid for location in locations]
    # Create a dict of locations indexed by rid and order
    ridLocs = {(location.rid, location.order): location for location in locations}

    for permutation in permutations(rids):
        permLocs = list()
        # Start each permutation with the fixed starting location
        # No request ID or pickup/dropoff order is associated with it
        if startLoc is not None:
            permLocs.append(startLoc)

        ridCounts = dict()
        for rid in permutation:
            if rid in ridCounts.keys():
                ridCounts[rid] += 1
            else:
                ridCounts[rid] = 0

            loc = ridLocs[rid, ridCounts[rid]]
            permLocs.append(loc)

        perms.add(tuple(permLocs))

    return perms


def createLocations(reqs, veh=None):
    stop_locations = list()
    for req in reqs:
        oLoc = StopLocation(req["origin"][0], req["origin"][1],
            req["rid"], 0, req["Cep"], req["Clp"], 1)
        dLoc = StopLocation(req["destination"][0], req["destination"][1],
            req["rid"], 1, req["Ced"], req["Cld"], -1)
        stop_locations.append(oLoc)
        stop_locations.append(dLoc)

    if veh is not None:
        for req in veh["pax_reqs"]:
            [Ced, Cld] = req.get_dropoff_time_window()
            dLoc = StopLocation(req.dlng, req.dlat, req.id, 0,
                Ced, Cld, -1)
            stop_locations.append(dLoc)

        for req in veh["wait_reqs"]:
            [Ced, Cld] = req.get_dropoff_time_window()
            oLoc = StopLocation(req.olng, req.olat, req.id, 0,
                req.Cep, req.Clp, 1)
            dLoc = StopLocation(req.dlng, req.dlat, req.id, 1,
                Ced, Cld, -1)
            stop_locations.append(oLoc)
            stop_locations.append(dLoc)

    return stop_locations


def findOptimalRouting(reqs, G, veh=None, T=0):
    stopLocs = createLocations(reqs, veh)

    if veh is not None:
        startTime = T + veh["t"]
        startLoc = StopLocation(veh["location"][0], veh["location"][1])
        capacity = veh["capacity"]
        currentPax = len(veh["pax_reqs"])

        # Create a set of all request IDs, whether being assigned in this timestep,
        # assigned in previous timesteps but still waiting, or already inside the vehicle
        allRids = set([loc.rid for loc in stopLocs])

        # If there are 3 or more requests in said set, perform pairwise feasibility checks
        if len(allRids) >= 3:
            # print("{} Requests + {} Current Pax + {} Assigned Reqs --> Pairwise Checks".format(
                # len(reqs), len(veh["pax_reqs"]), len(veh["wait_reqs"])))
            # print("All request IDs: {}".format(allRids))

            # Create a set of all request IDs already assigned to the vehicle (whether inside or
            # still waiting), since we know if 2 requests have already been assigned that they
            # are pairwise feasible
            vehRids = set([req.id for req in veh["pax_reqs"]]) | set([req.id for req in veh["wait_reqs"]])

            # print("Previously assigned request IDs: {}".format(vehRids))

            # Iterate through each combination of 2 requests
            for r1, r2 in combinations(allRids, 2):
                # Ignore requests if they were both assigned in previous timesteps
                if r1 in vehRids and r2 in vehRids:
                    continue

                # print("Testing Request {} + Request {}".format(r1, r2))
                # Find the optimal path from vehicle's current location to both locations
                comboLocs = [loc for loc in stopLocs if loc.rid == r1 or loc.rid == r2]
                _, comboRoute = minimalDelayPath(
                    comboLocs, G, startTime, startLoc)

                # If no feasible path is found, then we know that there will not be a feasible
                # path when even more requests are added into the picture. So return an infeasible
                # result.
                if comboRoute is None:
                    # print("Failure: Reqs {} & {}".format(r1, r2))
                    return 1e6, None
        return minimalDelayPath(
            stopLocs, G, startTime, startLoc, capacity, currentPax)
    else:
        return minimalDelayPath(stopLocs, G, T)


def tabuOptimalRouting(reqs, G, veh=None, T=0, maxTime=None):
    stopLocs = createLocations(reqs, veh)

    if veh is not None:
        startTime = T + veh["t"]
        startLoc = StopLocation(veh["location"][0], veh["location"][1])
        capacity = veh["capacity"]
        currentPax = len(veh["pax_reqs"])

        # # Create a set of all request IDs, whether being assigned in this timestep,
        # # assigned in previous timesteps but still waiting, or already inside the vehicle
        # allRids = set([loc.rid for loc in stopLocs])

        # # If there are 3 or more requests in said set, perform pairwise feasibility checks
        # if len(allRids) >= 3:
        #     # Create a set of all request IDs already assigned to the vehicle (whether inside or
        #     # still waiting), since we know if 2 requests have already been assigned that they
        #     # are pairwise feasible
        #     vehRids = set([req.id for req in veh["pax_reqs"]]) | set([req.id for req in veh["wait_reqs"]])

        #     # Iterate through each combination of 2 requests
        #     for r1, r2 in combinations(allRids, 2):
        #         # Ignore requests if they were both assigned in previous timesteps
        #         if r1 in vehRids and r2 in vehRids:
        #             continue

        #         # Find the optimal path from vehicle's current location to both locations
        #         comboLocs = [loc for loc in stopLocs if loc.rid == r1 or loc.rid == r2]
        #         # feas_check_t = time.time()
        #         _, comboRoute = tabuSearchPath(comboLocs, G, startTime, startLoc)
        #         # print("    Pairwise feasibility check for veh {} took {:.2f} seconds".format(
        #         #     veh["vid"], time.time()-feas_check_t))

        #         # If no feasible path is found, then we know that there will not be a feasible
        #         # path when even more requests are added into the picture. So return an infeasible
        #         # result.
        #         if comboRoute is None:
        #             return 1e6, None

        # Maximum tabu size is 3 times (2 + the number of stop locations)
        cost, route = tabuSearchPath(
            stopLocs, G, startTime, startLoc, capacity, currentPax, maxTabuSize=(3*(len(stopLocs)+2)), maxTime=maxTime)

    else:
        cost, route = tabuSearchPath(stopLocs, G, T, maxTabuSize=(3*(len(stopLocs)+2)), maxTime=maxTime)

    if cost > maxFeasibleCost(stopLocs):
        return 1e6, None
    else:
        return cost, route


def tabuSearchPath(locs, G, T=0, startLoc=None, capacity=1e6, startPax=0, maxTabuSize=30, maxIter=150, maxTime=None):
    # This function uses Tabu search algorithm to quickly find the best or near-best routing
    # for a specified set of locations.

    # If the starting location is specified, insert it into the 1st position in the array
    # of stop locations
    if startLoc is not None:
        locs.insert(0, startLoc)

    # Initialize two temperatures:
    # For feasible solution, we use 5% of the maximum feasible cost so that any feasible
    # solution is initially possible to jump to, although the chance of the best possible
    # solution jumping to the worst is ~1e-9.
    # For infeasible solutions, we use 1000 times the maximum feasible cost, so that there
    # is still resolution to differentiate better infeasible solutions from worse ones, but
    # that an adjacent infeasible solution is still highly likely to be accepted so that we
    # can quickly explore the infeasible space and find the feasible space.
    mfCost = maxFeasibleCost(locs)
    feasTemp = mfCost/20
    infTemp = mfCost * 1e3

    # Initialize the routing with a greedy algorithm solution, minimizing crow-flies distance
    # from each location to the next feasible one
    bestOrder = createInitialRouting(locs)
    # print("Initial solution is: {}".format(" -> ".join([str((loc.rid, loc.pod)) for loc in bestOrder if loc.rid is not None])))

    # Initialize the Tabu list with only the initial ordering
    tabuList = list()
    tabuList.append(bestOrder)

    # Compute the cost of the initial solution
    bestCost = computeRoutingCost(bestOrder, G, T, capacity, startPax, infStop=False, infCost=infTemp)

    # If the initial solution is already feasible, then we initialize the temperature as the
    # initial feasible temperature. Otherwise initialize temperature at the infeasible level.
    if bestCost <= mfCost:
        temp = feasTemp
    else:
        temp = infTemp

    curCost = bestCost
    curOrder = bestOrder

    terminate = False
    iterations = 0
    iterSinceNewBest = 0
    costEvals = 1

    swapCosts = dict()

    t_start = time.time()

    while not terminate:
        # print("Iteration {}".format(iterations))
        possibleSwaps = enumerateSwaps(curOrder)
        # print("    {} possible swaps found from current solution".format(len(possibleSwaps)))
        # If every possible swap from the current ordering is Tabu, then terminate the search
        if all([x in tabuList for x in possibleSwaps]):
            terminate = True
            continue

        # Iterate through all possible swaps to find the best allowable candidate
        bestCandidate = None
        bestCandCost = infTemp*1e3
        for candidate in possibleSwaps:
            if candidate in swapCosts.keys():
                candidateCost = swapCosts[candidate]
            else:
                candidateCost = computeRoutingCost(candidate, G, T, capacity, startPax, infStop=False, infCost=infTemp)
                # candidateCost = computeRoutingCost(candidate, G, T, capacity, startPax, infStop=True, infCost=infTemp)
                swapCosts[candidate] = candidateCost
                costEvals += 1
            # Aspirational Criterion: If a candidate is on the Tabu list, but its cost is better than all
            # previously found solutions, we can still accept that move. (I am still unclear how this ever happens...)
            if (candidate not in tabuList or candidateCost < bestCost) and candidateCost < bestCandCost:
                bestCandidate = candidate
                bestCandCost = candidateCost

        # print("    Best candidate is: {}".format(" -> ".join([str((loc.rid, loc.pod)) for loc in bestCandidate if loc.rid is not None])))
        # print("    Candidate cost: {:.2f}".format(bestCandCost))

        # If the candidate improves upon our current solution, accept it
        if bestCandCost < curCost:
            curCost = bestCandCost
            curOrder = bestCandidate

            # And if it's better than any previously found solution, it becomes the new best
            if curCost <= bestCost:
                bestCost = curCost
                bestOrder = curOrder
                iterSinceNewBest = 0
            #     print(" NEW BEST FOUND!")
            # else:
            #     print("  Improvement on current solution found!")

        # We accept candidates that are inferior to our current solution with some probability
        # that declines over time, based on the temperature.
        elif random.random() < math.exp((curCost - bestCandCost)/temp):
            # print("  Inferior candidate accepted through random chance")
            # print("    Probability of acceptance: {:.2f}%".format(100*threshold))

            curCost = bestCandCost
            curOrder = bestCandidate
            # Testing: Have the temperature decrease much faster for infeasible solutions
            # to speed up the identification of infeasibility
            if temp <= feasTemp:
                temp = temp * 0.75
            else:
                temp = temp * 0.1

        # If the current best cost now represents a feasible solution, replace
        # the temperature with the starting feasible temperature
        if temp > feasTemp and bestCost < mfCost:
            temp = feasTemp

        # Add the best candidate to the Tabu list to prevent exploring it in the future
        tabuList.append(bestCandidate)
        # If the Tabu list size exceeds the maximum, remove the oldest element in it
        if len(tabuList) >= maxTabuSize:
            tabuList.pop(0)

        iterations += 1
        iterSinceNewBest += 1

        elapsed = time.time() - t_start

        # If a maximum time is specified and that much time has elapsed, terminate
        if maxTime is not None and elapsed > maxTime:
            terminate = True

        # If too many iterations have passed, terminate
        if iterations >= maxIter:
            terminate = True

    elapsed = time.time() - t_start

    # if elapsed > 0.25:
    #     print("    Tabu search completed in {:.2f} seconds".format(elapsed), end="")
    #     print(" with {} route cost evaluations".format(costEvals), end="")
    #     print(" - Result: {}".format("INFEASIBLE" if bestCost > mfCost else "FEASIBLE"))

    # Return the route given by the best order as a list of (rid, pod, lng, lat) tuples
    # For use in the Agents.Model to build routes for all vehicles
    bestRoute = [(loc.rid, loc.pod, loc.lng, loc.lat) for loc in bestOrder if loc.rid is not None]

    return bestCost, bestRoute


def enumerateSwaps(locOrder):
    swaps = list()
    pax_rids = set()

    # Note: Always check location.pod == 1 or == -1 because the vehicle
    # location is part of the order but can't swap (and has pod=None)

    # Generate a set of rids that are already in the vehicle. Any request
    # that is in the location order only as a dropoff is a current pax
    for location in locOrder:
        if location.pod == 1:
            pax_rids.add(location.rid)
        elif location.pod == -1:
            if location.rid not in pax_rids:
                pax_rids.add(location.rid)
            else:
                pax_rids.remove(location.rid)

    # Iterate through the list of locations and generate the potential
    # other locations that location can swap with
    for i, location in enumerate(locOrder):
        if location.pod == -1:
            # Remove the location request ID from the set of pax in vehicle
            pax_rids.remove(location.rid)
            # Iterate through every subsequent location to see if this location
            # can swap with it while preserving pickup before dropoff constraints
            for j, other in enumerate(locOrder):
                if j > i:
                    # It can swap with any dropoff of a passenger already in
                    # the vehicle, or any pickup
                    if other.rid in pax_rids or other.pod == 1:
                        new_order = createSwap(locOrder, i, j)
                        swaps.append(tuple(new_order))

        elif location.pod == 1:
            # Add the location request ID to the set of pax in vehicle
            pax_rids.add(location.rid)
            # Iterate through every subsequent location to see if this location
            # can swap with it while preserving pickup before dropoff constraints
            for j, other in enumerate(locOrder):
                if j > i:
                    # If this is a pickup, it can swap with any location that
                    # precedes its own dropoff
                    if other.rid == location.rid:
                        break
                    # It can swap with any dropoff of a passenger already in
                    # the vehicle, or any pickup
                    if other.rid in pax_rids or other.pod == 1:
                        new_order = createSwap(locOrder, i, j)
                        swaps.append(tuple(new_order))


    assert len(pax_rids) == 0

    return swaps


def createSwap(l, i, j):
    update = [l[idx] for idx in range(i)] + [l[j]] + [l[idx] for idx in range(i+1, j)] + [l[i]] + [l[idx] for idx in range(j+1, len(l))]
    return update


def maxFeasibleCost(locations):
    # The maximum cost of a feasible solution is 100% of the delay constraint for every
    # request involved. Equal to the latest acceptable arrival time at each destination
    # minus that destination's earliest acceptable arrival time.
    return sum([loc.ubT - loc.lbT for loc in locations if loc.pod == -1])


def minimalDelayPath(locs, G, T=0, startLoc=None, capacity=1e6, startPax=0):
    stopPerms = permutePassengerOrder(locs, startLoc)

    # Store the minimum cost path found so far, its cost,
    # and the order of pickups/dropoffs generated
    bestCost = 1e6
    bestRoute = None

    # Iterate through all possible permutations of stop
    # orderings
    for locOrder in stopPerms:
        cost = computeRoutingCost(locOrder, G, T, capacity, startPax)

        # If the routing is feasible and the cost is improved,
        # store the new best cost and route
        if cost < bestCost:
            bestCost = cost
            bestRoute = [(loc.rid, loc.pod, loc.lng, loc.lat) for loc in locOrder if loc.rid is not None]
            # print("NEW BEST PATH FOUND!")

    # return bestPath, bestCost, bestRoute
    return bestCost, bestRoute


def computeRoutingCost(orderedLocs, G, T=0, capacity=1e6, startPax = 0, infStop=True, infCost=1e6):
    # Create placeholders for path cost, travel time, and whether
    # there exists a feasible routing or not
    totCost = 0
    # Time starts at the current system time, plus the time needed for the
    # vehicle to reach the next downstream node (used as its starting location)
    totTime = T
    feasible = True
    currentPax = startPax

    # If we somehow start with more passengers than veh capacity then all
    # paths are infeasible
    assert currentPax <= capacity

    # Create lists to store the segment sources and segment targets
    src = list()
    tgt = list()

    # Iterate through each pair (o, d) in the ordered locations
    for locIdx in range(len(orderedLocs)-1):
        # Create vertex names in maps graph from origin & destination locations
        origin = orderedLocs[locIdx]
        onodename = str((origin.lng, origin.lat))
        src.append(onodename)

        destination = orderedLocs[locIdx+1]
        dnodename = str((destination.lng, destination.lat))
        tgt.append(dnodename)

        # Add 1 if the destination is a pickup location, or subtract 1 if
        # it is a dropoff location, from our count of current passengers
        # If # current passengers exceeds capacity, this routing is infeasible
        currentPax += destination.pod
        if currentPax > capacity:
            feasible = False
            if infStop:
                break

        # This was a slower implementation that called shortest_paths()
        # for each segment origin->destination
        # # Get the shortest path between the origin and destination
        # # using the links' expected travel times, and check whether
        # # the total elapsed time at the end of this segment meets
        # # the time constraints at the destination
        # path_ett = G.shortest_paths(onodename, dnodename, weights="ttmedian")
        # assert len(path_ett) == 1
        # assert len(path_ett[0]) == 1
        # assert path_ett[0][0] >= 0
        # totTime += path_ett[0][0]
        # if not destination.checkTimeConstraints(totTime):
        #     feasible = False
        #     if infStop:
        #         break

        # if destination.pod < 0:
        #     totCost += totTime - destination.lbT


    if feasible or not infStop:
        # Get the shortest path lengths between all origins and destinations
        # using the links' expected travel times. Shortest paths are computed
        # between each src node and each destination node, but we extract only
        # the diagonal (src 1 -> tgt 1, src 2 -> tgt 2, etc.). This is still
        # faster than calling shortest_paths() several times with only 1 src and
        # 1 tgt each time.
        # print(src)
        # print(tgt)
        if len(set(tgt)) < len(tgt):
            # But if there are duplicate locations in tgt (i.e. two requests share
            # a pickup/dropoff location), then we must do the slower method.
            segLengths = list()
            for i in range(len(src)):
                pathLength = G.shortest_paths(src[i], tgt[i], weights="ttmedian")
                segLengths.append(pathLength[0][0])
        else:
            pathLengths = G.shortest_paths(src, tgt, weights="ttmedian")
            segLengths = [pathLengths[i][i] for i in range(len(src))]

        # Compute the arrival time at each destination using the cumulative sum
        # of segment travel times
        cumTimes = [T+sum(segLengths[:i+1]) for i in range(len(src))]

        # Check each destination node to see whether the arrival time meets
        # the destination time window constraints
        for dIdx, destination in enumerate(orderedLocs[1:]):
            dTime = cumTimes[dIdx]
            if not destination.checkTimeConstraints(dTime):
                feasible = False
                if infStop:
                    break

            # If this is a dropoff, add that request's delay beyond its earliest
            # acceptable dropoff time to the cost of this pairing
            if destination.pod < 0:
                totCost += dTime - destination.lbT

    if feasible:
        return totCost
    elif infStop:
        return infCost
    else:
        return infCost + totCost


def addTripToRTVGraph(rtvGraph, trip_info, veh):
    # Request vertices are named "req_X" where X is the rid
    rids = [int(req["name"][4:]) for req in trip_info["reqs"]]
    trip_name = "trip_" + "_".join([str(rid) for rid in rids])

    starting_num_trips = len(rtvGraph.vs.select(vtype="trip"))
    starting_trip_edges = 0

    try:
        trip = rtvGraph.vs.find(name=trip_name)
        starting_trip_edges += len(rtvGraph.incident(trip.index, mode=ig.ALL))
        assert len(rtvGraph.vs.select(vtype="trip")) == starting_num_trips
    except ValueError:
        rtvGraph.add_vertex(trip_name, vtype="trip")
        trip = rtvGraph.vs.find(name=trip_name)
        assert len(rtvGraph.vs.select(vtype="trip")) == starting_num_trips + 1

    try:
        edge = rtvGraph.es.find(_source=veh.index, _target=trip.index)
        assert (edge.index in rtvGraph.incident(veh.index, mode=ig.ALL) and
            edge.index in rtvGraph.incident(trip.index, mode=ig.ALL))
    except ValueError:
        rtvGraph.add_edge(veh, trip, cost=trip_info["cost"], route=trip_info["route"], etype="veh_trip")

    for req in trip_info["reqs"]:
        try:
            edge = rtvGraph.es.find(_source=req.index, _target=trip.index)
            assert (edge.index in rtvGraph.incident(req.index, mode=ig.ALL) and
                edge.index in rtvGraph.incident(trip.index, mode=ig.ALL))
        except ValueError:
            rtvGraph.add_edge(req, trip, etype="req_trip")

    return rtvGraph, trip


def createInitialRouting(locations):
    # Create an initial routing from the list of locations using a greedy algorithm.
    # Start at the first location provided (either the vehicle starting location or a
    # request pickup location) and choose the next feasible location in the stop locations
    # (i.e. ignoring dropoff locations where we haven't yet visited the corresponding
    # pickup) with the shortest distance until all locations are elaborated
    routing = list()
    pax_rids = set()

    # Generate a set of rids that are already in the vehicle. Any request
    # that is in the location order only as a dropoff is a current pax
    for location in locations:
        if location.pod == 1:
            pax_rids.add(location.rid)
        elif location.pod == -1:
            if location.rid not in pax_rids:
                pax_rids.add(location.rid)
            else:
                pax_rids.remove(location.rid)

    # print(pax_rids)

    nextLoc = locations[0]
    routing.append(nextLoc)
    remainingLocs = set(locations)
    remainingLocs.remove(nextLoc)

    while len(remainingLocs) > 0:
        rid = nextLoc.rid
        pod = nextLoc.pod
        olat = nextLoc.lat
        olng = nextLoc.lng

        if pod == 1:
            pax_rids.add(rid)
        elif pod == -1 and rid in pax_rids:
            pax_rids.remove(rid)

        nextLoc = None
        bestDist = 1e20
        for loc in remainingLocs:
            if loc.rid in pax_rids or loc.pod == 1:
                dist = greatCircleDistance(olat, olng, loc.lat, loc.lng)
                if dist < bestDist:
                    nextLoc = loc

        assert nextLoc is not None

        remainingLocs.remove(nextLoc)
        routing.append(nextLoc)

    assert len(pax_rids) == 0 or (nextLoc.pod == -1 and nextLoc.rid in pax_rids and len(pax_rids)==1)

    return routing


def greatCircleDistance(olat, olng, dlat, dlng):
    gc_dist = (6371000*2*math.pi/360 * np.sqrt( (math.cos((olat+dlat)*math.pi/360)*(olng-dlng))**2 + (olat-dlat)**2))
    return gc_dist


def shortestPathTime(G, olat, olng, dlat, dlng, weight="ttmedian"):
    onodename = str((olng, olat))
    dnodename = str((dlng, dlat))
    pathTimes = G.shortest_paths(onodename, target=dnodename, weights=weight)

    assert len(pathTimes) == 1
    assert len(pathTimes[0]) == 1

    return pathTimes[0][0]

# Helper functions for flow network -based assignment
def getEdgeInfoFromRoadGraph(onode, dnode, G):
    oloc = str(onode["location"])
    dloc = str(dnode["location"])

    path = G.get_shortest_paths(oloc, to=dloc, weights="ttmedian", output="epath")[0]

    path_length = sum([G.es[e]["length"] for e in path])
    path_ttmedian = sum([G.es[e]["ttmedian"] for e in path])

    return onode["name"], dnode["name"], path, path_length, path_ttmedian


def getTargetNodeId(G, eid):
    e = G.es[eid]
    tgtVid = e.target
    tgtVertex = G.vs[tgtVid]
    return tgtVertex["nid"]


def get_road_path_attrs(onode, dnode, G):
    oloc = str(onode["location"])
    dloc = str(dnode["location"])

    path = G.get_shortest_paths(oloc, to=dloc, weights="ttmedian", output="epath")[0]

    path_length = sum([G.es[e]["length"] for e in path])

    return path, path_length


def add_flow_edge(source, target, G_road, G_flow, time_table=None, otaThreshold=0, detourThreshold=1e6):
    v1 = source["name"]
    v2 = target["name"]

    path, length = get_road_path_attrs(source, target, G_road)

    # print("Path between {} and {} is of length {} and uses {} links".format(source["name"], target["name"], length, len(path)))

    if time_table is None and target["Cl"] >= 0 and sum([G_road.es[e]["ttmedian"] for e in path]) > detourThreshold*target["Cl"]:
        return None
    elif time_table is None:
        G_flow.add_edges([(v1, v2),], {"length": length})
        return path
    else:
        arrivalTimes = list()
        for scenario in time_table:
            arrivalTimes.append(sum([scenario[e] for e in path]))

        otaFreq = np.sum(np.array(arrivalTimes) <= target["Cl"])

        if otaFreq > otaThreshold:
            G_flow.add_edges([(v1, v2),], {"length": length})
        else:
            return None


def createLocationsFromFlowNodes(vehNode, flowNodes):
    startLoc = StopLocation(vehNode["lng"], vehNode["lat"])

    stopLocs = list()
    reqOrderMap = dict()

    for node in flowNodes:
        lng = node["lng"]
        lat = node["lat"]
        rid = node["rid"]
        pod = node["pod"]
        lbT = node["Ce"]
        ubT = node["Cl"]

        if node["vtype"] == "origin":
            order = 0
            reqOrderMap[rid] = 1
        else:
            if rid in reqOrderMap.keys():
                order = 1
            else:
                order = 0

        stopLoc = StopLocation(lng, lat, rid, order, lbT, ubT, pod)
        stopLocs.append(stopLoc)

    return startLoc, stopLocs

def getFlowsFromRoutes(veh_routes, G):
    veh_flows = dict()
    sinkNode = G.vs.find(vtype="sink")

    # A route is a (rid, pod, lng, lat) tuple
    for vid in veh_routes.keys():
        route = veh_routes[vid]
        
        veh_flow = list()
        
        vehNode = G.vs.find(name="veh_{}_start".format(vid))
        currLoc = vehNode

        for rid, pod, _, _ in route:
            if pod == 1:
                nodeName = "req_{}_pickup".format(rid)
            else:
                nodeName = "req_{}_dropoff".format(rid)

            reqNode = G.vs.find(name=nodeName)

            edge = G.es.find(_source=currLoc.index, _target=reqNode.index)
            veh_flow.append(edge.index)

            currLoc = reqNode

        veh_flow.append(G.es.find(_source=currLoc, _target=sinkNode.index).index)

        veh_flows[vid] = veh_flow

    return veh_flows

def createSubgraphFromEdge(G, e, k):    
    sink_node = G.vs.find(vtype="sink")
    sink_vid = sink_node.index

    src_vid = G.es[e].source
    tgt_vid = G.es[e].target

    src = G.vs[src_vid]
    tgt = G.vs[tgt_vid]

    nodes = [k, src_vid, tgt_vid, sink_vid]

    veh_node = G.vs[k]
    pax_rids = veh_node["pax_reqs"]

    for pax_rid in pax_rids:
        pax_orgn = G.vs.find("req_{}_pickup".format(pax_rid))
        nodes.append(pax_orgn.index)

        pax_dest = G.vs.find("req_{}_dropoff".format(pax_rid))
        nodes.append(pax_dest.index)

    for loc in (src, tgt):
        if loc["vtype"] == "origin":
            loc_dest = G.vs.find("req_{}_dropoff".format(loc["rid"]))
            nodes.append(loc_dest.index)
        else:
            loc_orgn = G.vs.find("req_{}_pickup".format(loc["rid"]))
            nodes.append(loc_orgn.index)

    subgraph = G.induced_subgraph(nodes)
    edge_map = dict()
    for edge in subgraph.es:
        e_src = subgraph.vs[edge.source]
        e_tgt = subgraph.vs[edge.target]

        upper_src = G.vs.find(e_src["name"])
        upper_tgt = G.vs.find(e_tgt["name"])
        upper_edge = G.es.find(_source=upper_src.index, _target=upper_tgt.index)

        edge_map[edge.index] = upper_edge.index

    sub_src = subgraph.vs.find(src["name"])
    sub_tgt = subgraph.vs.find(tgt["name"])
    target_edge = subgraph.es.find(_source=sub_src.index, _target=sub_tgt.index)

    return subgraph, target_edge.index, edge_map


def optimalRouteOnSubgraph(G, target_edge_idx, times, N_R):
    m = gp.Model("subgraph-routing")
    m.setParam(GRB.Param.LogToConsole, 0)

    reqONodes = G.vs.select(vtype="origin")
    reqDNodes = G.vs.select(vtype="destination")
    allReqNodes = set(reqONodes) | set(reqDNodes)

    vehNode = G.vs.find(vtype="vehicle")
    k = vehNode["nid"]

    sinkNode = G.vs.find(vtype="sink")
    sinkId = sinkNode["nid"]

    V_O = [reqO["nid"] for reqO in reqONodes]
    V_D = [reqD["nid"] for reqD in reqDNodes]
    V = [req["nid"] for req in allReqNodes]
    EIJ = [(edge.index, G.vs[edge.source]["nid"], G.vs[edge.target]["nid"]) for edge in G.es]

    out_edges = gp.tupledict({vertex["nid"]: [edge.index for edge in G.es.select(_source=vertex.index)] for vertex in G.vs})
    in_edges = gp.tupledict({vertex["nid"]: [edge.index for edge in G.es.select(_target=vertex.index)] for vertex in G.vs})
    latest_dropoffs = gp.tupledict({vertex["nid"]: vertex["Cl"] for vertex in allReqNodes})
    node_occupancies = gp.tupledict({vertex["nid"]: vertex["pod"] for vertex in G.vs})
    capacity = vehNode["capacity"]

    x_e = gp.tuplelist([edge.index for edge in G.es])
    p_i = gp.tuplelist([node["nid"] for node in G.vs if node["nid"] != sinkId])

    flow = m.addVars(x_e, vtype=GRB.BINARY, name="x")
    req_veh_order = m.addVars(p_i, vtype=GRB.INTEGER, name="p", lb=0)
    veh_occupancy = m.addVars(p_i, vtype=GRB.INTEGER, name="n", lb=0)
    arrival_time = m.addVars(p_i, vtype=GRB.CONTINUOUS, name="tau", lb=0)
    node_delay = m.addVars(p_i, vtype=GRB.CONTINUOUS, name="z", lb=0)

    m.addConstr(sum([flow[e] for e in out_edges[k]]) == 1, "veh_start_flow")
    m.addConstr(sum([flow[e] for e in in_edges[sinkId]]) == 1, "veh_sink_flow")
    m.addConstrs((sum([flow[e] for e in in_edges[i]]) - sum([flow[e] for e in out_edges[i]]) == 0 for i in V),
        "flow_conservation")
    m.addConstrs((sum([flow[e] for e in in_edges[i]]) - sum([flow[e] for e in out_edges[i+N_R]]) == 0 for i in V_O),
        "pickup_dropoff_flow")

    m.addConstrs((1e3*(flow[e] - 1) + req_veh_order[getTargetNodeId(G, e)] - req_veh_order[i] <= 1 for e, i, j in EIJ if j != sinkId),
        "req_veh_flow_order_lb")
    m.addConstrs((1e3*(1 - flow[e]) + req_veh_order[getTargetNodeId(G, e)] - req_veh_order[i] >= 1 for e, i, j in EIJ if j != sinkId),
        "req_veh_flow_order_ub")
    m.addConstrs((req_veh_order[i+N_R] - req_veh_order[i] >= 1 for i in V_O),
        "dropoff_after_pickup")

    m.addConstrs((1e3*(flow[e] - 1) + veh_occupancy[getTargetNodeId(G, e)] - veh_occupancy[i] <= node_occupancies[getTargetNodeId(G, e)] for i in V for e in out_edges[i] if getTargetNodeId(G, e) != sinkId),
        "pod_occupancy_ub")
    m.addConstrs((1e3*(1 - flow[e]) + veh_occupancy[getTargetNodeId(G, e)] - veh_occupancy[i] >= node_occupancies[getTargetNodeId(G, e)] for i in V for e in out_edges[i] if getTargetNodeId(G, e) != sinkId),
        "pod_occupancy_lb")
    m.addConstrs((veh_occupancy[i] <= capacity for i in V), "veh_capacity_constraint")

    m.addConstrs((sum([flow[e] for e in in_edges[i]]) == 1 for i in V_O), "exactly_one_assignment")

    m.addConstrs((arrival_time[j] - arrival_time[i] >= times[e] - 1e6*(1 - flow[e]) for e, i, j in EIJ if j != sinkId),
        "veh_node_arrival_time")
    m.addConstrs((node_delay[i] >= arrival_time[i] - latest_dropoffs[i] for i in V),
        "delay_from_pickup")

    m.addConstr(flow[target_edge_idx] == 1, "forced_target_edge_flow")

    m.setObjective(node_delay.sum(), GRB.MINIMIZE)

    m.optimize()

    return m.getObjective().getValue()


def calcDelayFromSolution(G, flow, scenario_dict, T=0):
    vehNodes = G.vs.select(vtype="vehicle")
    reqONodes = G.vs.select(vtype="origin")
    reqDNodes = G.vs.select(vtype="destination")
    allReqNodes = set(reqONodes) | set(reqDNodes)

    K = [veh["nid"] for veh in vehNodes]
    V = [req["nid"] for req in allReqNodes]

    avg_arrival_times = {v: 0 for v in V}
    avg_delays = {v: 0 for v in V}

    scenario_arrival_times = {(v, s): 0 for v in V for s in scenario_dict.keys()}
    scenario_delays = {(v, s): 0 for v in V for s in scenario_dict.keys()}

    for k in K:
        veh_flow_edges = G.es.select([e.index for e in G.es if (e.index, k) in flow.keys() and np.isclose(flow[(e.index, k)], 1.0)])

        flow_dests = set([G.vs[e.target]["nid"] for e in veh_flow_edges if G.vs[e.target]["nid"] in V])

        src = G.vs.find(nid=k)
        t = T
        scenario_t = [T]*len(scenario_dict)

        while len(flow_dests) > 0:
            edge = veh_flow_edges.find(_source=src.index)
            tgt = G.vs[edge.target]

            edge_times = list()
            for s in scenario_dict.keys():
                times = scenario_dict[s]["times"].T
                edge_times.append(np.asscalar(times[edge.index]))
            t += np.mean(edge_times)

            avg_arrival_times[tgt["nid"]] = t
            avg_delays[tgt["nid"]] = np.max([t - tgt["Cl"], 0])

            for s in scenario_dict.keys():
                times = scenario_dict[s]["times"].T
                scenario_t[s] += np.asscalar(times[edge.index])
                scenario_arrival_times[(tgt["nid"], s)] = scenario_t[s]
                scenario_delays[(tgt["nid"], s)] = np.max([scenario_t[s] - tgt["Cl"], 0])

                # print("In scenario {} vehicle {} arrives at {} (Cl={:.1f}) at t={:.1f}, delay={:.1f}".format(
                #     s, k, tgt["nid"], tgt["Cl"], scenario_arrival_times[(tgt["nid"], s)], scenario_delays[(tgt["nid"], s)]))

            src = tgt
            flow_dests.remove(tgt["nid"])

    avg_delay = np.sum(list(avg_delays.values()))
    scenario_delay = {s: np.sum([scenario_delays[(v, s)] for v in V]) for s in scenario_dict}

    return avg_delay, scenario_delay


def convertFlowAssignmentToRouting(G, flow, order=None, times=None, T=0):
    VTYPE_PODS = {
        "origin": 1,
        "destination": -1,
    }

    non_zero_flows = [f for f in flow.keys() if np.isclose(flow[f], 1.0)]

    veh_routes = dict()

    out_edges = gp.tupledict({vertex["nid"]: [edge.index for edge in G.es.select(_source=vertex.index, pruned=False)] for vertex in G.vs})

    for veh in G.vs.select(vtype="vehicle"):
        vid = veh["vid"]
        veh_route = list()
        current_node = veh

        t = T

        while current_node.index != G.vs.find(vtype="sink").index:
            out_flow_edges = [out_eid for out_eid in out_edges[current_node["nid"]] if (out_eid, veh["nid"]) in non_zero_flows]
            assert len(out_flow_edges) == 1

            for out_eid in out_flow_edges:
                out_edge = G.es[out_eid]
                tgt = G.vs[out_edge.target]
                if not tgt["dummy"]:
                    rid = tgt["rid"]
                    pod = VTYPE_PODS[tgt["vtype"]]
                    lng = tgt["lng"]
                    lat = tgt["lat"]
                    constr = tgt["Cl"]

                    if order is not None:
                        # P = order[tgt["nid"], veh["nid"]]
                        P = np.round(order[out_eid, veh["nid"]], 0)
                    else:
                        P = "n/a"

                    if times is not None:
                        t += np.asscalar(times[out_eid])
                        edge_time = np.round(np.asscalar(times[out_eid]), 2)
                        delay = np.max([np.round(t - constr, 2), 0])
                    else:
                        edge_time = "n/a"
                        delay = "n/a"

                    # veh_route.append((rid, pod, lng, lat, P, edge_time, constr, delay))
                    veh_route.append((rid, pod, lng, lat))

                current_node = tgt

        if len(veh_route) > 0:
            veh_routes[vid] = veh_route

    return veh_routes