"""
multiple classes for the AMoD system
"""

import numpy as np
import copy
import math
# import pdb # Debug
from collections import deque
import itertools

from lib.Demand import *
from lib.Constants import *
from lib.Optimization import AlonsoMora


# Diagnostic function
def print_for_rid(rid, prnt, target_rid=1090):
    if rid == target_rid:
        print(prnt)


class Step(object):
    """
    Step is a class for steps in a leg
    Attributes:
        d: distance
        t: duration
        et: expected duration
        geo: geometry, a list of coordinates
    """
    def __init__(self, d=0.0, t=0.0, et=0.0, geo=[]):
        self.d = d
        self.t = t
        self.et = et
        self.geo = geo
        assert len(self.geo) == 2

    def __str__(self):
        return "step: distance = {:.2f}, duration = {:.2f}, expected duration = {:.2f}".format(
            self.d, self.t, self.et)


class Leg(object):
    """
    Leg is a class for legs in the route
    A leg may consists of a series of steps
    Attributes:
        rid: request id (if rebalancing then -1)
        pod: pickup (+1) or dropoff (-1), rebalancing (0)
        tlng: target (end of leg) longitude
        tlat: target (end of leg) latitude
        d: total distance
        t: total duration
        steps: a list of steps
    """
    def __init__(self, rid, pod, tlng, tlat, d=0.0, t=0.0, steps=[]):
        self.rid = rid
        self.pod = pod
        self.tlng = tlng
        self.tlat = tlat
        self.d = d
        self.t = t
        self.steps = deque(steps)
        assert len(self.steps) == 0 or len(self.steps[0].geo) == 2

    def __str__(self):
        return "leg: distance = {:.2f}, duration = {:.2f}, number of steps = {}".format(
            self.d, self.t, len(self.steps))


class Veh(object):
    """
    Veh is a class for vehicles
    Attributes:
        id: sequential unique id
        idle: is idle
        rebl: is rebalancing
        T: system time at current state
        lat: current lngitude
        lng: current longtitude
        tlat: target (end of route) lngitude
        tlng: target (end of route) longtitude
        K: capacity
        S: speed (m/s)
        n: number of passengers on board
        route: a list of legs
        t: total duration of the route
        d: total distance of the route
        c: total cost (generalized time) of the passegners
        Ds: accumulated service distance traveled
        Ts: accumulated service time traveled
        Dp: accumulated pick-up distance traveled
        Tp: accumulated pick-up time traveled
        Dr: accumulated rebalancing distance traveled
        Tr: accumulated rebalancing time traveled
        Lt: accumulated load, weighed by service time
        Ld: accumulated load, weighed by service distance
        occ: Boolean variable whether the vehicle is occupied
        oes: number of occupancy events (OEs) defined as time in between two periods when vehicle is empty
    """
    def __init__(self, id, rs, K=4, S=6, T=0.0, ini_loc=None):
        self.id = id
        self.idle = True
        self.rebl = False
        self.T = T
        if ini_loc is None:
            self.lng = (Olng+Dlng)/2 + (Dlng-Olng)*rs.uniform(-0.35, 0.35)
            self.lat = (Olat+Dlat)/2 + (Dlat-Olat)*rs.uniform(-0.35, 0.35)
        else:
            self.lng = ini_loc[0]
            self.lat = ini_loc[1]
        # print('Vehicle initialized at ({}, {})'.format(self.lat, self.lng))
        self.tlng = self.lng
        self.tlat = self.lat
        self.K = K
        self.S = S
        self.n = 0
        self.route = deque([])
        self.t = 0.0
        self.d = 0.0
        self.c = 0.0
        self.Ds = 0.0
        self.Ts = 0.0
        self.Dp = 0.0
        self.Tp = 0.0
        self.Dr = 0.0
        self.Tr = 0.0
        self.Lt = 0.0
        self.Ld = 0.0
        self.oes = 0

    def get_location(self):
        return (self.lng, self.lat)

    def get_target_location(self):
        return (self.tlng, self.tlat)

    def jump_to_location(self, lng, lat):
        self.lng = lng
        self.lat = lat

    # build the route of the vehicle based on a series of quadruples (rid, pod, tlng, tlat)
    # update t, d, c, idle, rebl accordingly
    # rid, pod, tlng, tlat are defined as in class Leg
    def build_route(self, router, route, reqs=None, T=None, pre_drawn_tt=None):
        # Initialize set of requests already in vehicle by looking at existing route before
        # building a new one. This is used in setting the requests' predicted TT and WT
        reqs_in_veh, _ = self.get_passenger_requests()
        # Not all assigned requests are stored in the vehicle's route yet
        reqs_not_in_veh = set([leg[0] for leg in route if leg[0] not in reqs_in_veh])

        merge_first_leg = False
        if len(self.route)>0:
            self.clear_route_but_keep_next_node()
            merge_first_leg = True
        else:
            self.clear_route()

        # if the route is null, vehicle is idle
        if len(route) == 0:
            self.idle = True
            self.rebl = False
            self.t = 0.0
            self.d = 0.0
            self.c = 0.0
            return
        else:
            # print('Evaluating passengers for vehicle {}\'s new route...\n'.format(self.id))
            # Reset predicted travel times to reevaluate with new route
            for rid in reqs_in_veh:
                # Reset PVT and PWT from this point forward as 0. PVT will get added to
                # EEVT at end, PWT should remain 0 since req is already inside veh
                assert reqs[rid].Tp > 0
                reqs[rid].pvt = 0
                reqs[rid].pwt = 0
                if merge_first_leg:
                    # Add expected time of remaining step from prev route to PVT
                    reqs[rid].pvt += sum([step.et for step in self.route[0].steps])
                print_for_rid(rid, 'Request {} is inside vehicle {}; PVT reset to 0, PWT unchanged'.format(rid, self.id))
                print_for_rid(rid, 'Updating request {}\'s travel and waiting time predictions'.format(rid))
            for rid in reqs_not_in_veh:
                # Reset PVT and PWT from this point forward as 0. PWT will be added to
                # EEWT at the end.
                # TODO: Make sure this makes sense for in-advance requests
                assert np.isclose(reqs[rid].Tp, -1.0)
                reqs[rid].pvt = 0
                reqs[rid].pwt = 0
                if merge_first_leg:
                    # Add expected time of remaining step from prev route to PWT
                    reqs[rid].pwt += sum([step.et for step in self.route[0].steps])
                print_for_rid(rid, 'Request {} not in vehicle {}; PVT and PWT reset to 0.'.format(rid, self.id))

            for i, (rid, pod, tlng, tlat) in enumerate(route):
                # Add the new leg to the vehicle's route with a drawn travel time
                if pre_drawn_tt is not None:
                    added_leg = self.add_leg(router, rid, pod, tlng, tlat, reqs, T, pre_drawn_tt[i])
                else:
                    added_leg = self.add_leg(router, rid, pod, tlng, tlat, reqs, T)

                # print('Leg added to vehicle {}\'s route:'.format(self.id))
                # print(added_leg)
                # Then update the predicted journey times for each passenger in vehicle

                # Get the predicted leg duration without uncertainty using the average
                # travel time of each link on the route
                pred_t = sum([step.et for step in added_leg.steps])
                # print('The new leg has an expected travel time of {}'.format(pred_t))

                # Add to PVT for all requests in vehicle
                for req_in_veh_id in reqs_in_veh:
                    reqs[req_in_veh_id].pvt += pred_t
                    print_for_rid(req_in_veh_id, '{} added to request {}\'s PVT: new PVT {}'.format(pred_t, req_in_veh_id, reqs[req_in_veh_id].pvt))

                # Add to PWT for all requests yet to be picked up
                for req_outside_veh_id in reqs_not_in_veh:
                    reqs[req_outside_veh_id].pwt += pred_t
                    print_for_rid(req_outside_veh_id, '{} added to request {}\'s PWT: new PWT {}'.format(pred_t, req_outside_veh_id, reqs[req_outside_veh_id].pwt))

                # Add any picked up trip into vehicle or remove any dropped off trip
                # (Does not change actual occupants, just for computation)
                if pod == 1:
                    reqs_in_veh.add(rid)
                    reqs_not_in_veh.remove(rid)
                    print_for_rid(rid, 'Request {} picked up!'.format(rid))
                if pod == -1:
                    reqs_in_veh.remove(rid)
                    print_for_rid(rid, 'Request {} dropped off!'.format(rid))

            if merge_first_leg:
                # Remove first leg, which is just moving the vehicle to the immediately downstream node from its current location
                first_leg = self.route[0]
                del self.route[0]

                # Store distance and time of removed leg + new first leg
                x = self.route[0].d + first_leg.d
                y = self.route[0].t + first_leg.t

                # Add steps from the removed leg to the beginning of the new first leg
                self.route[0].steps.appendleft(first_leg.steps[0])
                # Update distance and time of first leg based on new steps
                self.route[0].d = sum([step.d for step in self.route[0].steps])
                self.route[0].t = sum([step.t for step in self.route[0].steps])

                # Check that the updated new first leg matches what we expect
                assert np.isclose(self.route[0].d, x)
                assert np.isclose(self.route[0].t, y)

            # if rid is -1, vehicle is rebalancing
            if self.route[0].rid == -1:
                self.idle = True
                self.rebl = True
                self.c = 0.0
                return
            # else, the vehicle is in service to pick up or dropoff
            else:
                c = 0.0
                self.idle = False
                self.rebl = False
                t = 0.0
                n = self.n

                for leg in self.route:
                    t += leg.t
                    c += n * leg.t * COEF_INVEH
                    n += leg.pod
                    c += t * COEF_WAIT if leg.pod == 1 else 0

                assert len(reqs_in_veh) == 0
                for rid in set([leg.rid for leg in self.route]):
                    if reqs[rid].has_predictions:
                        print_for_rid(rid, 'After updating the route for vehicle {}, request {} has revised rerouted expected times'.format(self.id, rid))
                        if reqs[rid].pwt > 0:
                            print_for_rid(rid, 'Including the elapsed expected waiting time of {:.2f}s, request {}\'s revised RWT is {:.2f}s'.format(
                                reqs[rid].eewt, rid, reqs[rid].pwt + reqs[rid].eewt))
                        # Add elapsed expected times to PVT and PWT to obtain RPVT and RPWT
                        reqs[rid].revt = reqs[rid].pvt + reqs[rid].eevt
                        reqs[rid].rewt = reqs[rid].pwt + reqs[rid].eewt
                        print_for_rid(rid, 'Including the elapsed expected travel time of {}s, request {}\'s revised RVT is {}s'.format(
                            reqs[rid].eevt, rid, reqs[rid].revt))
                    else:
                        # Include time since request in the vehicle's EEWT and FPWT
                        reqs[rid].fpvt = reqs[rid].pvt
                        reqs[rid].fpwt = reqs[rid].pwt + T - reqs[rid].Tr
                        reqs[rid].eewt = T - reqs[rid].Tr
                        reqs[rid].has_predictions = True
                        print_for_rid(rid, 'After updating the route for vehicle {}, request {} has predicted times'.format(self.id, rid))
                        print_for_rid(rid, 'Adding the elapsed time since request of {:.2f}s to the predicted remaining WT of {}s, request {}\'s predicted waiting time is {:.2f}'.format(
                            T-reqs[rid].Tr, reqs[rid].pwt, rid, reqs[rid].fpwt))
                        print_for_rid(rid, 'Predicted In-Vehicle Travel Time: {}'.format(reqs[rid].fpvt))

                assert n == 0
                self.c = c

                # print('\n')

    # remove the current route
    def clear_route(self):
        self.d = 0.0
        self.t = 0.0
        self.c = 0.0
        self.tlng = self.lng
        self.tlat = self.lat
        self.route.clear()

    # Get the immediate downstream node of this vehicle and the time it will take to reach
    def get_next_node(self):
        if len(self.route) > 0:
            tloc = self.route[0].steps[0].geo[1]
            t = self.route[0].steps[0].t
        else:
            tloc = (self.lng, self.lat)
            t = 0

        return tloc, t

    def clear_route_but_keep_next_node(self):
        tlng, tlat = self.route[0].steps[0].geo[1]
        rid = self.route[0].rid
        pod = 0 # Traveling to this next node downstream doesn't affect the number of pax in veh

        # Create new leg with only step being the previous route's first leg's first step
        step = self.route[0].steps[0]
        new_leg = Leg(rid, pod, tlng, tlat, step.d, step.t, steps=[])
        new_leg.steps.append(step)

        self.clear_route()
        self.route.append(new_leg)

        self.tlng = new_leg.steps[-1].geo[1][0]
        self.tlat = new_leg.steps[-1].geo[1][1]
        self.d += new_leg.d
        self.t += new_leg.t

    # add a leg based on (rid, pod, tlng, tlat)
    def add_leg(self, router, rid, pod, tlng, tlat, reqs, T, drawn_leg_tt=None, skip_routing=False):
        if not skip_routing:
            # If link travel times for the route have been pre-drawn, use them as the travel
            # times for the routing so that it matches the cost. Otherwise, draw them now
            # because the cost calculation used modal travel times
            if drawn_leg_tt is not None:
                l = router.get_routing(self.tlng, self.tlat, tlng, tlat, pre_drawn_tt=drawn_leg_tt)
            else:
                l = router.get_routing(self.tlng, self.tlat, tlng, tlat, use_uncertainty=LINK_UNCERTAINTY)

            leg = Leg(rid, pod, tlng, tlat,
                l['distance'], l['duration'], steps=[])
            t_leg = 0.0
            for s in l['steps']:
                step = Step(s['distance'], s['duration'], s['mean_duration'],
                    [[s['intersections'][0]['location'][0], s['intersections'][0]['location'][1]],
                     [s['intersections'][1]['location'][0], s['intersections'][1]['location'][1]]])
                t_leg += s['duration']
                leg.steps.append(step)

            assert np.isclose(t_leg, leg.t)

            if pod == 1:
                if T+self.t+leg.t < reqs[rid].Cep:
                    wait = reqs[rid].Cep - (T+self.t+leg.t)
                    leg.steps[-1].t += wait
                    leg.t += wait

            self.route.append(leg)
        else:
            d_, t_ = router.get_distance_duration(self.tlng, self.tlat, tlng, tlat, euclidean=True)
            leg = Leg(rid, pod, tlng, tlat, d_, t_, steps=[])
            leg.steps.append(Step(d_, t_, [[self.tlng, self.tlat],[tlng, tlat]]))
            self.route.append(leg)
        self.tlng = leg.steps[-1].geo[1][0]
        self.tlat = leg.steps[-1].geo[1][1]
        self.d += leg.d
        self.t += leg.t

        return leg

    # update the vehicle location as well as the route after moving to time T
    def move_to_time(self, T, reqs):
        dT = T - self.T
        if dT <= 0:
            return []
        # done is a list of finished legs
        done = []
        while dT > 0 and len(self.route) > 0:
            leg = self.route[0]
            rids_in_veh, rids_not_in_veh = self.get_passenger_requests()
            # if the first leg could be finished by then
            if leg.t < dT:
                dT -= leg.t
                self.T += leg.t
                if self.T >= T_WARM_UP and self.T <= T_WARM_UP+T_STUDY:
                    self.Ts += leg.t if leg.rid != -1 and self.n > 0 else 0
                    self.Ds += leg.d if leg.rid != -1 and self.n > 0 else 0
                    self.Tp += leg.t if leg.rid != -1 and self.n == 0 else 0
                    self.Dp += leg.d if leg.rid != -1 and self.n == 0 else 0
                    self.Tr += leg.t if leg.rid == -1 else 0
                    self.Dr += leg.d if leg.rid == -1 else 0
                    self.Lt += leg.t * self.n if leg.rid != -1 else 0
                    self.Ld += leg.d * self.n if leg.rid != -1 else 0

                # On pickup, add 1 occupancy event to the vehicle if either:
                # * There were no passengers in the vehicle prior to pickup
                # * This is the first pickup recorded within the study period (i.e. no previously recorded occupancy events)
                # NOTE: Request statistics are recorded for requests that request pickup within study period, a slightly
                #       different criteria than above for vehicle statistics.
                if reqs[leg.rid].Cep >= T_WARM_UP and reqs[leg.rid].Tr <= T_WARM_UP+T_STUDY and (len(rids_in_veh) == 0 or self.oes == 0):
                    self.oes += 1
                    reqs[leg.rid].new_occ = True

                self.jump_to_location(leg.tlng, leg.tlat)
                rids_in_veh, rids_not_in_veh = self.update_passengers_on_leg(leg, reqs, rids_in_veh, rids_not_in_veh)

                done.append( (leg.rid, leg.pod, self.T) )
                self.pop_leg()
            else:
                while dT > 0 and len(leg.steps) > 0:
                    step = leg.steps[0]
                    # if the first leg could not be finished, but the first step of the leg could be finished by then
                    if step.t < dT:
                        dT -= step.t
                        self.T += step.t
                        if self.T >= T_WARM_UP and self.T <= T_WARM_UP+T_STUDY:
                            self.Ts += step.t if leg.rid != -1 and self.n > 0 else 0
                            self.Ds += step.d if leg.rid != -1 and self.n > 0 else 0
                            self.Tp += step.t if leg.rid != -1 and self.n == 0 else 0
                            self.Dp += step.d if leg.rid != -1 and self.n == 0 else 0
                            self.Tr += step.t if leg.rid == -1 else 0
                            self.Dr += step.d if leg.rid == -1 else 0
                            self.Lt += step.t * self.n if leg.rid != -1 else 0
                            self.Ld += step.d * self.n if leg.rid != -1 else 0
                        self.jump_to_location(leg.tlng, leg.tlat)
                        self.update_passengers_on_step(step, reqs, rids_in_veh, rids_not_in_veh)
                        self.pop_step()
                        if len(leg.steps) == 0:
                            # corner case: leg.t extremely small, but still larger than dT
                            # this is due to the limited precision of the floating point numbers
                            self.jump_to_location(leg.tlng, leg.tlat)
                            self.update_passengers_on_leg(leg, reqs, rids_in_veh, rids_not_in_veh)
                            done.append( (leg.rid, leg.pod, self.T) )
                            self.pop_leg()
                            break
                    # the vehicle has to stop somewhere within the step
                    else:
                        pct = dT / step.t
                        if self.T >= T_WARM_UP and self.T <= T_WARM_UP+T_STUDY:
                            self.Ts += dT if leg.rid != -1 and self.n > 0 else 0
                            self.Ds += step.d * pct if leg.rid != -1 and self.n > 0 else 0
                            self.Tp += dT if leg.rid != -1 and self.n == 0 else 0
                            self.Dp += step.d * pct if leg.rid != -1 and self.n == 0 else 0
                            self.Tr += dT if leg.rid == -1 else 0
                            self.Dr += step.d * pct if leg.rid == -1 else 0
                            self.Lt += dT * self.n if leg.rid != -1 else 0
                            self.Ld += step.d * pct * self.n if leg.rid != -1 else 0
                        # Could put a lot of effort into conditioning the remaining expected time
                        # on the step based on the observed time so far.... But for now just
                        # scaling the expected time the same as the rest of the step attributes
                        partial_step = Step(step.d*pct, step.t*pct, step.et*pct, step.geo)
                        self.update_passengers_on_step(partial_step, reqs, rids_in_veh, rids_not_in_veh)
                        # find the exact location the vehicle stops and update the step
                        self.cut_step(pct)
                        # print(f'Step has been cut: {step.geo}')
                        self.jump_to_location(step.geo[0][0], step.geo[0][1])
                        self.T = T
                        return done
        assert dT > 0 or np.isclose(dT, 0.0)
        assert self.T < T or np.isclose(self.T, T)
        assert len(self.route) == 0
        assert self.n == 0
        assert np.isclose(self.d, 0.0)
        assert np.isclose(self.t, 0.0)

        self.T = T
        self.d = 0.0
        self.t = 0.0

        return done

    # get the location at time T
    # this function is similar to move_to_time(self, T), but it does not update the route
    def get_location_at_time(self, T):
        dT = T - self.T
        if dT <= 0:
            return self.lng, self.lat, self.n
        lng = self.lng
        lat = self.lat
        n = self.n
        route = copy.deepcopy(self.route)
        while dT > 0 and len(route) > 0:
            leg = route[0]
            if leg.t < dT:
                dT -= leg.t
                lng = leg.tlng
                lat = leg.tlat
                n += leg.pod
                route.popleft()
            else:
                while dT > 0 and len(leg.steps) > 0:
                    step = leg.steps[0]
                    if step.t < dT:
                        dT -= step.t
                        leg.steps.popleft()
                        if len(leg.steps) == 0:
                            # corner case: leg.t extremely small, but still larger than dT
                            lng = leg.tlng
                            lat = leg.tlat
                            n += leg.pod
                            route.popleft()
                            break
                    else:
                        pct = dT / step.t
                        self.cut_temp_step(step, pct)
                        lng = step.geo[0][0]
                        lat = step.geo[0][1]
                        return lng, lat, n
        assert dT > 0 or np.isclose(dT, 0.0)
        assert len(route) == 0
        assert n == 0
        return lng, lat, n

    # pop the first leg from the route list
    def pop_leg(self):
        leg = self.route.popleft()
        self.d -= leg.d
        self.t -= leg.t

    # pop the first step from the first leg
    def pop_step(self):
        step = self.route[0].steps.popleft()
        self.t -= step.t
        self.d -= step.d
        self.route[0].t -= step.t
        self.route[0].d -= step.d

    # find the exact location the vehicle stops and update the step
    def cut_step(self, pct):
        # print('Cutting step...')
        step = self.route[0].steps[0]
        save_step = (step.d, step.t, step.geo.copy())
        if np.isclose(step.d, 0.0):
            _pct = pct
        else:
            dis = 0.0
            sega = step.geo[0]
            for segb in step.geo[1:]:
                dis += np.sqrt( (sega[0] - segb[0])**2 + (sega[1] - segb[1])**2)
                sega = segb
            if dis != 0.0:
                dis_ = 0.0
                _dis = 0.0
                sega = step.geo[0]
                for segb in step.geo[1:]:
                    _dis = np.sqrt( (sega[0] - segb[0])**2 + (sega[1] - segb[1])**2)
                    dis_ += _dis
                    if dis_ / dis >= pct:
                        break
                    sega = segb
                while step.geo[0] != sega:
                    step.geo.pop(0)
                if _dis != 0.0:
                    _pct = (pct * dis - dis_ + _dis) / _dis
                else:
                    _pct = 1.0
                step.geo[0][0] = sega[0] + _pct * (segb[0] - sega[0])
                step.geo[0][1] = sega[1] + _pct * (segb[1] - sega[1])
        self.t -= step.t * pct
        self.d -= step.d * pct
        self.route[0].t -= step.t * pct
        self.route[0].d -= step.d * pct
        self.route[0].steps[0].t -= step.t * pct
        self.route[0].steps[0].d -= step.d * pct
        self.route[0].steps[0].et -= step.et * pct
        # print('After cutting, the first step on the route is now:')
        # print(self.route[0].steps[0])
        assert len(step.geo) == 2

    # similar to cut_step(self, pct), but always called by get_location_at_time(self, T)
    def cut_temp_step(self, step, pct):
        if step.d != 0:
            dis = 0.0
            sega = step.geo[0]
            for segb in step.geo[1:]:
                dis += np.sqrt( (sega[0] - segb[0])**2 + (sega[1] - segb[1])**2)
                sega = segb
            dis_ = 0.0
            _dis = 0.0
            sega = step.geo[0]
            for segb in step.geo[1:]:
                _dis = np.sqrt( (sega[0] - segb[0])**2 + (sega[1] - segb[1])**2)
                dis_ += _dis
                if dis_ / dis > pct:
                    break
                sega = segb
            while step.geo[0] != sega:
                step.geo.pop(0)
            _pct = (pct * dis - dis_ + _dis) / _dis
            step.geo[0][0] = sega[0] + _pct * (segb[0] - sega[0])
            step.geo[0][1] = sega[1] + _pct * (segb[1] - sega[1])

    # Determines which passengers are inside the vehicle at the given time
    # Returns a tuple of 2 sets of request IDs:
    # (requests currently in vehicle, requests currently awaiting vehicle)
    def get_passenger_requests(self):
        reqs_to_pick_up = set()
        reqs_to_drop_off = set()
        for leg in self.route:
            if leg.pod == 1:
                reqs_to_pick_up.add(leg.rid)
            if leg.pod == -1:
                reqs_to_drop_off.add(leg.rid)
        # Requests currently in vehicle are those we need to drop off that we don't need to pick up
        reqs_in_veh = reqs_to_drop_off - reqs_to_pick_up

        return reqs_in_veh, reqs_to_pick_up

    def update_passengers_on_leg(self, leg, reqs, rids_in_veh=None, rids_not_in_veh=None):
        # If no passengers specified, obtain them from the vehicle's actual route
        if rids_in_veh is None or rids_not_in_veh is None:
            rids_in_veh, rids_not_in_veh = self.get_passenger_requests()

        # Calculate the total elapsed _expected_ travel time for the completed leg,
        # taken by summing the expected travel time for each step of the leg
        total_elapsed_et = sum([step.et for step in leg.steps])
        total_time_dev = sum([abs(step.et - step.t) for step in leg.steps])

        # Add this elapsed expected travle time to current passengers' elapsed in-vehicle
        # expected travel time, and to awaiting passengers' elapsed expected wait time
        for rid in rids_in_veh:
            reqs[rid].eevt += total_elapsed_et
            reqs[rid].vtdev += total_time_dev
        for rid in rids_not_in_veh:
            reqs[rid].eewt += total_elapsed_et
            reqs[rid].wtdev += total_time_dev

        # Update the number of pickups and dropoffs seen by each passenger in the vehicle
        if leg.pod == 1:
            for rid in rids_in_veh:
                reqs[rid].NP += 1
            rids_in_veh.add(leg.rid)
            rids_not_in_veh.remove(leg.rid)
        elif leg.pod == -1:
            rids_in_veh.remove(leg.rid)
            for rid in rids_in_veh:
                reqs[rid].ND += 1

        # Update the number of passengers in the vehicle
        self.n += leg.pod

        return rids_in_veh, rids_not_in_veh

    def update_passengers_on_step(self, step, reqs, rids_in_veh=None, rids_not_in_veh=None):
        # If no passengers specified, obtain them from the vehicle's actual route
        if rids_in_veh is None or rids_not_in_veh is None:
            rids_in_veh, rids_not_in_veh = self.get_passenger_requests()

        # Update elapsed expected in-vehicle travel time and wait time for current passengers
        # and awaiting passengers, respectively, by adding the completed step's expected
        # travel time
        for rid in rids_in_veh:
            reqs[rid].eevt += step.et
            reqs[rid].vtdev += abs(step.et - step.t)
        for rid in rids_not_in_veh:
            reqs[rid].eewt += step.et
            reqs[rid].wtdev += abs(step.et - step.t)

    # visualize
    def draw(self):
        import matplotlib.pyplot as plt
        color = "0.50"
        if self.id == 0:
            color = "red"
        elif self.id == 1:
            color = "orange"
        elif self.id == 2:
            color = "yellow"
        elif self.id == 3:
            color = "green"
        elif self.id == 4:
            color = "blue"
        plt.plot(self.lng, self.lat, color=color, marker='o', markersize=4, alpha=0.5)
        count = 0
        for leg in self.route:
            count += 1
            plt.plot(leg.tlng, leg.tlat, color=color,
                     marker='s' if leg.pod == 1 else 'x' if leg.pod == -1 else None, markersize=3, alpha=0.5)
            for step in leg.steps:
                geo = np.transpose( step.geo )
                plt.plot(geo[0], geo[1], color=color, linestyle='-' if count<=1 else '--', alpha=0.5)

    def __str__(self):
        str =  "veh %d at (%.7f, %.7f) when t = %.3f; %s; occupancy = %d/%d" % (
            self.id, self.lng, self.lat, self.T, "rebalancing" if self.rebl else "idle" if self.idle else "in service", self.n, self.K)
        str += "\n  service dist/time: %.1f, %.1f; rebalancing dist/time: %.1f, %.1f" % (
            self.Ds, self.Ts, self.Dr, self.Tr)
        str += "\n  has %d leg(s), dist = %.1f, dura = %.1f, cost = %.1f" % (
            len(self.route), self.d, self.t, self.c)
        for leg in self.route:
            str += "\n    %s req %d at (%.7f, %.7f), dist = %.1f, dura = %.1f" % (
                "pickup" if leg.pod == 1 else "dropoff" if leg.pod == -1 else "rebalancing",
                leg.rid, leg.tlng, leg.tlat, leg.d, leg.t)
        return str


class Req(object):
    """
    Req is a class for requests
    Attributes:
        id: sequential unique id
        Tr: request time
        olng: origin longtitude
        olat: origin lngitude
        dlng: destination longtitude
        dlat: destination lngitude
        Ds: shortest travel distance from request's origin to destination
        Ts: shortest travel time from request's origin to destination
        Ps: shortest travel distance from assigned vehicle to request's origin
        Rs: shortest travel time from assigned vehicle to request's origin
        OnD: true if on-demand, false if in-advance
        Cep: constraint - earliest pickup
        Clp: constraint - latest pickup
        Tp: pickup time
        Td: dropoff time
        DF: detour factor
        DR: distance rejected (true if distance O->D is less than DISTANCE_THRESHOLD)
        NS: number of passengers in the vehicle when this passenger is dropped off
        NP: number of other requests picked up while this passenger is in vehicle
        ND: number of other requests dropped off while this passenger is in vehicle
        m_id: The row of the demand matrix used to generate this request
        assigned: True if request has been assigned to a vehicle, False otherwise
        assigned_veh: id of vehicle that serves this trip, if any
        new_occ: 1 if this is request is a new occupancy event, 0 otherwise
        has_predictions: True if any predictions (pred_tt/wt) have been made, False otherwise
        pvt: Current prediction of in-vehicle travel time
        pwt: Current prediction of waiting time
        fpvt: First prediction of in-vehicle travel time when assigned to vehicle
        fpwt: First prediction of waiting time when assigned to vehicle
        revt: Prediction of in-vehicle travel time when picked up
        rewt: Prediction of waiting time after last update to route was made before pickup
        eewt: Elapsed expected waiting time (the expected travel time along all links that the vehicle
              assigned to this request has already completed on its way to pick up the request)
        eevt: Elapsed expected in-vehicle travel time
        wtdev: Total absolute time deviation for the waiting time (i.e. the sum of all absolute differences
               between the actual travel times and expected travel times of links traveled on while waiting)
        vtdev: Total absolute time deviation for the in-vehicle time (i.e. the sum of all absolute differences
               between the actual travel times and expected travel times of links traveled while in vehicle)

    """
    def __init__(self, router, id, Tr, olng, olat, dlng, dlat, OnD=True, m_id=None):
        self.id = id
        self.Tr = Tr
        self.olng = olng
        self.olat = olat
        self.dlng = dlng
        self.dlat = dlat
        self.Ds, self.Ts = router.get_distance_duration(olng, olat, dlng, dlat, euclidean=False)
        self.Rs = -1.0

        self.OnD = OnD
        if self.OnD:
            self.Cep = Tr
            self.Clp = Tr + MAX_WAIT
        else:
            self.Cep = Tr + T_ADV_REQ
            self.Clp = Tr + T_ADV_REQ + MAX_WAIT
        self.Tp = -1.0
        self.Td = -1.0
        self.D = 0.0
        self.DR = False
        self.NS = 0
        self.NP = 0
        self.ND = 0
        self.m_id = m_id
        self.assigned = False
        self.assigned_veh = -1
        self.new_occ = False
        self.has_predictions = False
        self.pvt = 0.0
        self.pwt = 0.0
        self.fpvt = 0.0
        self.fpwt = 0.0
        self.revt = 0.0
        self.rewt = 0.0
        self.eevt = 0.0
        self.eewt = 0.0
        self.wtdev = 0.0
        self.vtdev = 0.0

    # return origin
    def get_origin(self):
        return (self.olng, self.olat)

    # return destination
    def get_destination(self):
        return (self.dlng, self.dlat)

    def get_shortest_path(self, router):
        return router.get_routing(self.olng, self.olat, self.dlng, self.dlat)

    def get_dropoff_time_window(self):
        return [self.Cep + self.Ts, self.Clp + MAX_DETOUR * self.Ts]

    # visualize
    def draw(self):
        import matplotlib.pyplot as plt
        plt.plot(self.olng, self.olat, 'r', marker='+')
        plt.plot(self.dlng, self.dlat, 'r', marker='x')
        plt.plot([self.olng, self.dlng], [self.olat, self.dlat], 'r', linestyle='--', dashes=(0.5,1.5))

    def __str__(self):
        str = "req %d from (%.7f, %.7f) to (%.7f, %.7f) at t = %.3f" % (
            self.id, self.olng, self.olat, self.dlng, self.dlat, self.Tr)
        str += "\n  earliest pickup time = %.3f, latest pickup at t = %.3f" % ( self.Cep, self.Clp)
        str += "\n  pickup at t = %.3f, dropoff at t = %.3f" % ( self.Tp, self.Td)
        return str


class Model(object):
    """
    Model is the class for the AMoD system
    Attributes:
        rs1: a seeded random generator for requests
        rs2: a seeded random generator for vehicle locations
        T: system time at current state
        M: demand matrix
        D: demand volume (trips/hour)
        V: number of vehicles
        K: capacity of vehicles
        vehs: the list of vehicles
        N: number of requests
        G: graph of road network (if graph is enabled)
        reqs: the list of requests
        rejs: the list of rejected requests
        distance_rejs: the list of requests rejected because the distance from O to D
            was below the distance threshold (not included in rejs)
        queue: requests in the queue
        assign: assignment method
        rebl: rebalancing method
    """
    def __init__(self, M, D, V=2, K=4, assign="ins", rebl="no", seeds=None, G=None):
        if seeds is None:
            seed1 = np.random.randint(0,1000000)
            seed2 = np.random.randint(0,1000000)
            print(' - Seed 1 : {}\n - Seed 2 : {}'.format(seed1, seed2), file=open('output/seeds.txt', 'a'))
            seeds = (seed1, seed2)
        else:
            assert len(seeds) == 2
        # two random generators, the seed of which could be modified for debug use
        self.rs1 = np.random.RandomState(seeds[0])
        self.rs2 = np.random.RandomState(seeds[1])
        self.T = 0.0
        self.M = M
        self.D = D
        self.V = V
        self.K = K
        self.G = G
        self.vehs = []
        for i in range(V):
            rand_node = self.rs2.choice(self.G.vs)
            self.vehs.append(Veh(i, None, K=K, ini_loc=(rand_node['lng'], rand_node['lat'])))
        self.N = 0
        self.reqs = []
        self.rejs = []
        self.distance_rejs = []
        self.queue = deque([])
        self.assign = assign
        self.rebl = rebl
        self.optimizer = AlonsoMora(params=OPT_PARAMS)

    # generate one request, following exponential arrival interval
    def generate_request(self, router):
        dt = 3600.0/self.D * self.rs1.exponential()
        rand = self.rs1.rand()
        for mid, demand_row in enumerate(self.M):
            if demand_row[5] > rand:
                # demand_row[1] is the origin latitude, so this says half of requests south
                # of 51.35N are in-advance requests.
                if IN_ADV_REQS:
                    OnD = False if demand_row[1] < 51.35 and self.rs1.rand() < 0.5 else True
                else:
                    OnD = True
                # if len(self.reqs) > 0 and self.reqs[-1].id == 28:
                #     pdb.set_trace()
                req = Req(router,
                          id=0 if self.N == 0 else self.reqs[-1].id+1,
                          Tr=dt if self.N == 0 else self.reqs[-1].Tr+dt,
                          olng=demand_row[0], olat=demand_row[1],
                          dlng=demand_row[2], dlat=demand_row[3],
                          OnD=OnD, m_id=mid) # move to last column for ttid in demand
                break
        return req

    # generate requests up to time T, following Poisson process
    def generate_requests_to_time(self, router, T):
        # if T == 90:
            # pdb.set_trace()
        if self.N > 0:
            self.queue.append(self.reqs[-1])
            # self.reqs[-1].assigned = False
        elif self.N == 0:
            req = self.generate_request(router)
            self.reqs.append(req)
            self.N += 1

        while self.reqs[-1].Tr <= T:
            req = self.generate_request(router)
            self.reqs.append(req)
            if req.Tr <= T:
                self.queue.append(self.reqs[-1])
            self.N += 1
        assert self.N == len(self.reqs)

    # dispatch the AMoD system: move vehicles, generate requests, assign, reoptimize and rebalance
    def dispatch_at_time(self, router, T):
        self.T = T
        for veh in self.vehs:
            # Calculate all vehicle movements until time T. Outputs a list of all finished legs in the timestep.
            done = veh.move_to_time(T, self.reqs)
            for (rid, pod, t) in done:
                # If a pickup leg was completed, ...
                if pod == 1:
                    # Update the request with pickup time
                    self.reqs[rid].Tp = t
                    # If no rerouted expected waiting time has been stored, that means that
                    # the initial prediction of waiting time was never updated, so use it for RWT
                    if np.isclose(self.reqs[rid].rewt, 0.0):
                        self.reqs[rid].rewt = self.reqs[rid].fpwt
                    # Reset the elapsed expected waiting time to 0 so that RWT won't update any more
                    self.eewt = 0.0
 
                # If a dropoff leg was completed, ...
                elif pod == -1:
                    # Update the request with dropoff time
                    self.reqs[rid].Td = t

                    # If no rerouted expected in-vehicle travel time has been stored, that means
                    # the initial prediction of IVTT was never updated, so use if for RVT
                    if np.isclose(self.reqs[rid].revt, 0.0):
                        self.reqs[rid].revt = self.reqs[rid].fpvt

                    # Set NS of request fulfilled to the sum of the number of pickups
                    # and dropoffs it saw as a passenger.
                    self.reqs[rid].NS = self.reqs[rid].NP + self.reqs[rid].ND

                    # If link-level travel time uncertainty is turned off (i.e. link TTs
                    # are deterministic) and the vehicle has capacity 1 (i.e. the trip
                    # wasn't pooled with others), then it must be the case that the
                    # request's shortest possible travel time (Ts) plus its pickup time (Tp)
                    # is equal to its dropoff time (t)
                    if not LINK_UNCERTAINTY and veh.K == 1:
                        assert np.isclose(self.reqs[rid].Tp + self.reqs[rid].Ts, t)


                    # If the trip was not from a location to the same location, then its detour
                    # is equal to the actual time it took from its origin to its destination
                    # divided by the time it would have taken along its shortest path
                    # TODO: Change this to account for travel time uncertainty, i.e. some
                    #       component of this "detour" comes from taking a different route
                    #       for pooling other requests, while some comes from variation in
                    #       the travel time along the shortest possible route
                    if self.reqs[rid].Ts != 0:
                        self.reqs[rid].D = (self.reqs[rid].Td - self.reqs[rid].Tp)/self.reqs[rid].Ts
                    else:
                        assert self.reqs[rid].Td == self.reqs[rid].Tp
                        self.reqs[rid].D = 1

            if len(veh.route) == 0:
                # The vehicle has no more route but hasn't yet been set
                # to idle - make it build an empty route for that goal
                veh.build_route(router, [])

        self.generate_requests_to_time(router, T)

        # Some testing for now of the Alonso-Mora optimization procedure
        # if np.isclose(T % 600, 0):
        #     print("Generating R-V graph for timestep {}".format(T))
        #     rvGraph = self.optimizer.generatePairwiseGraph(self.vehs, self.reqs, self.G, T)
        #     print("Graph successfully generated! |V|={}, |E|={}".format(len(rvGraph.vs), len(rvGraph.es)))
        #     rvGraph.write("output/graphs/alonsomora-rv-graph-{}.gml".format(T), "gml")

        #     print("Generating RTV graph for timestep {}".format(T))
        #     rtvGraph = self.optimizer.generateRTVGraph(rvGraph, self.G, T)
        #     print("Graph successfully generated! |V|={}, |E|={}".format(len(rtvGraph.vs), len(rtvGraph.es)))
        #     rtvGraph.write("output/graphs/alonsomora-rtv-graph-{}.gml".format(T), "gml")

        # Could add the previously rejected requests back into the pool if their Clp hasn't passed
        if PRINT_PROGRESS: print(self)
        if np.isclose(T % INT_ASSIGN, 0):
            if self.assign == "ins":
                self.insertion_heuristics(router, T)
            elif self.assign == "alonsomora":
                self.optimal_assignment(router, T)
        if np.isclose(T % INT_REBL, 0):
            if self.rebl == "sar":
                self.rebalance_sar(router)
            elif self.rebl == "orp":
                self.rebalance_orp(router, T)

    # Alonso-Mora anytime optimal assignment
    def optimal_assignment(self, router, T):
        for req in self.queue:
            reject_for_distance = False
            if req.Ts < TIME_THRESHOLD:
                reject_for_distance = True

            # if DISTANCE_THRESHOLD > 0:
            #     req_dist = router.get_distance(req.olng, req.olat, req.dlng, req.dlat)
            #     if req_dist < DISTANCE_THRESHOLD:
            #         reject_for_distance = True

            if reject_for_distance:
                self.distance_rejs.append(req)
                req.DR = True
                req.assigned = None
                print("Request {} rejected because shortest path time {:.1f} < threshold of {}s".format(
                    req.id, req.Ts, TIME_THRESHOLD))
                # print("Request {} rejected because distance {:.1f} < distance threshold of {}".format(
                #     req.id, req_dist, DISTANCE_THRESHOLD))

        veh_routes, rejs = self.optimizer.optimizeAssignment(
            self.vehs, self.reqs, self.G, T)

        for veh in veh_routes:
            route = veh_routes[veh]
            pax_reqs, wait_reqs = self.vehs[veh].get_passenger_requests()
            veh_reqs = pax_reqs | wait_reqs
            new_reqs = set()
            for rid, _, _, _ in route:
                if rid not in veh_reqs:
                    self.reqs[rid].assigned = True
                    self.reqs[rid].assigned_veh = veh
                    # Find the shortest path distance between the vehicle and the request's origin
                    # Since vehicle is extremely unlikely to be at exact node, calculate distance
                    # from the next downstream node and add the time it will take the vehicle to arrive there.
                    veh_next_loc, veh_step_t = self.vehs[veh].get_next_node()
                    self.reqs[rid].Ps, self.reqs[rid].Rs = router.get_distance_duration(
                        veh_next_loc[0], veh_next_loc[1], self.reqs[rid].olng, self.reqs[rid].olat)
                    self.reqs[rid].Rs += veh_step_t + T - self.reqs[rid].Cep

                    new_reqs.add(str(rid))
            text = "  Vehicle {} assigned to request".format(veh)
            if len(new_reqs) > 1:
                text += "s"
            text += " {}.".format(", ".join(new_reqs))
            print(text)
            self.vehs[veh].build_route(router, route, self.reqs, T)

        for rid in rejs:
            req = self.reqs[rid]
            self.rejs.append(req)
            req.assigned = True
            print("  Request {} rejected.".format(rid))

        l = len(self.queue)
        for _ in range(l):
            req = self.queue.popleft()
            # if not req.assigned:
            #     self.queue.append(req)

    # insertion heuristics
    def insertion_heuristics(self, router, T):
        l = len(self.queue)
        for _ in range(l):
            req = self.queue.popleft()
            reject_for_distance = False

            if DISTANCE_THRESHOLD > 0:
                # If the request distance O->D is shorter than the distance threshold, reject and do not assign to vehicle
                req_dist = router.get_distance(req.olng, req.olat, req.dlng, req.dlat)
                if req_dist < DISTANCE_THRESHOLD:
                    reject_for_distance = True

            if reject_for_distance:
                self.distance_rejs.append(req)
                req.DR = True
            elif not self.insert_heuristics(router, req, T):
                self.rejs.append(req)

            # Currently counting rejected trips as assigned because otherwise they continue to clog up the queue
            req.assigned=True

    # insert a request using the insertion heuristics method
    def insert_heuristics(self, router, req, T):
        dc_ = np.inf
        veh_ = None
        route_ = None
        pre_drawn_tt_ = None
        viol = None
        for veh in self.vehs:
            route = []
            if not veh.idle:
                for leg in veh.route:
                    route.append( (leg.rid, leg.pod, leg.tlng, leg.tlat) )
            else:
                assert veh.c == 0
            l = len(route)
            c = veh.c
            for i in range(l+1):
                for j in range(i+1, l+2):
                    route.insert(i, (req.id, 1, req.olng, req.olat) )
                    route.insert(j, (req.id, -1, req.dlng, req.dlat) )
                    flag, c_, viol, pre_drawn_tt = self.test_constraints_get_cost(router, route, veh, req, c+dc_)
                    if flag:
                        dc_ = c_ - c
                        veh_ = veh
                        route_ = copy.deepcopy(route)
                        if pre_drawn_tt is not None:
                            pre_drawn_tt_ = pre_drawn_tt.copy()
                    route.pop(j)
                    route.pop(i)
                    if viol > 0:
                        break
                if viol == 2:
                    break
        if veh_ != None:
            veh_.build_route(router, route_, self.reqs, T, pre_drawn_tt_)

            # print("    Insertion Heuristics: veh %d is assigned to req %d" % (veh_.id, req.id) )
            return True
        else:
            # print("    Insertion Heuristics: req %d is rejected!" % (req.id) )
            return False

    # test if a route can satisfy all constraints, and if yes, return the cost of the route
    def test_constraints_get_cost(self, router, route, veh, req, C):
        c = 0.0
        t = 0.0
        n = veh.n
        T = veh.T
        K = veh.K
        if PRE_DRAW_TTS:
            pre_drawn_tt = list()
        else:
            pre_drawn_tt = None

        # Vehicle's current position may not be a node if it is en route to another
        # destination, so we test the graph to see if current vehicle location is a node
        # and if not use the downstream node and add the time it needs to arrive at that
        # next node.
        try:
            cur_node = router.G.vs.find(name=str((veh.lng, veh.lat)))
            lng = cur_node['lng']
            lat = cur_node['lat']
        except ValueError:
            assert len(veh.route) > 0
            lng = veh.route[0].steps[0].geo[1][0]
            lat = veh.route[0].steps[0].geo[1][1]
            t += veh.route[0].steps[0].t

        for (rid, pod, tlng, tlat) in route:
            n += pod
            if n > K:
                return False, None, 1, None # over capacity
        n = veh.n
        for (rid, pod, tlng, tlat) in route:
            req_ = self.reqs[rid]

            # If the setting is enabled, draw travel times for the route
            # before testing costs so that the vehicle knows how long
            # the request will take to serve ahead of time. Otherwise,
            # the travel times will be drawn when the route is constructed
            # and are unknown to the vehicle when costs are evaluated
            dt = None
            if PRE_DRAW_TTS:
                path = router.G.get_shortest_paths(str((lng, lat)), to=str((tlng, tlat)), weights=router.ttweight, output='epath')[0]
                route_tt = 0
                edge_tt_dict = dict()
                for edge_index in path:
                    edge_tt = router.draw_link_travel_time(edge_index)
                    route_tt += edge_tt
                    edge_tt_dict[edge_index] = edge_tt
                dt = route_tt
                pre_drawn_tt.append((route_tt, edge_tt_dict, path))
            else:
                dt = router.get_duration(lng, lat, tlng, tlat)

            t += dt
            if pod == 1:
                if T + t < req_.Cep:
                    dt += req_.Cep - T - t
                    t += req_.Cep - T - t
                    req_.Cld = req_.Cep + MAX_DETOUR * req_.Ts
                elif T + t > req_.Clp:
                    return False, None, 2, None if rid == req.id else 0 # late pickup
                else:
                    req_.Cld = T + t + MAX_DETOUR * req_.Ts
            elif pod == -1 and T + t > req_.Cld:
                return False, None, 3, None if rid == req.id else 0 # late dropoff
            c += n * dt * COEF_INVEH
            n += pod
            assert n <= veh.K
            # Add the request's waiting time to the routing cost. If the request has already
            # been assigned but not yet picked up, it has a predicted WT > 0 and so the added
            # wait time beyond that previous prediction incurs an extra penalty (COEF_EXTRA_WAIT)
            if pod == 1:
                if req_.pwt > 0:
                    c += (T + t - req_.Tr - req_.pwt) * COEF_EXTRA_WAIT
                else:
                    c += t * COEF_WAIT
            if c > C:
                return False, None, 0, None
            lng = tlng
            lat = tlat
        return True, c, -1, pre_drawn_tt

    # rebalance using simple anticipatory rebalancing
    def rebalance_sar(self, router):
        for veh in self.vehs:
            if veh.idle:
                veh.clear_route()
                veh.rebl = False
                [d, v, s], center = self.get_state(veh)
                n = np.random.uniform(0, np.sum(d))
                m = 0
                for i,j in itertools.product(range(Mlat), range(Mlng)):
                    m += d[i][j]
                    if m > n:
                        break
                route = [(-1, 0, center[i][j][0], center[i][j][1])]
                veh.build_route(router, route)

    # rebalance using optimal rebalancing problem
    def rebalance_orp(self, router, T):
        d = np.zeros((Nlat, Nlng))
        c = np.zeros((Nlat, Nlng, 2))
        v = np.zeros((Nlat, Nlng))
        s = np.zeros((Nlat, Nlng))
        b = np.zeros((Nlat, Nlng))
        for m in self.M:
            for i,j in itertools.product(range(Nlat), range(Nlng)):
                if m[1] >= Dlat - (i+1)*Elat:
                    if m[0] <= Olng + (j+1)*Elng:
                        d[i][j] += m[4] * self.D
                        c[i][j][0] += m[0] * m[4] * self.D
                        c[i][j][1] += m[1] * m[4] * self.D
                        break
        for i,j in itertools.product(range(Nlat), range(Nlng)):
            if d[i][j] != 0:
                c[i][j][0] /= d[i][j]
                c[i][j][1] /= d[i][j]
        for veh in self.vehs:
            if veh.idle:
                veh.clear_route()
                veh.rebl = False
                for i,j in itertools.product(range(Nlat), range(Nlng)):
                    if veh.lat >= Dlat - (i+1)*Elat:
                        if veh.lng <= Olng + (j+1)*Elng:
                            v[i][j] += 1
                            break
            else:
                lng, lat, n = veh.get_location_at_time(T+INT_REBL)
                for i,j in itertools.product(range(Nlat), range(Nlng)):
                    if lat >= Dlat - (i+1)*Elat:
                        if lng <= Olng + (j+1)*Elng:
                            if n == 0:
                                s[i][j] += 0.8
                            elif n == 1:
                                s[i][j] += 0.4
                            elif n == 2:
                                s[i][j] += 0.2
                            elif n == 3:
                                s[i][j] += 0.1
                            else:
                                s[i][j] += 0.0
                            break
        for i,j in itertools.product(range(Nlat), range(Nlng)):
            if d[i][j] == 0:
                continue
            lamda = d[i][j] * INT_REBL/3600
            k = 0
            b[i][j] = 1.0
            while k <= s[i][j]:
                b[i][j] -= np.exp(-lamda) * (lamda**k) / np.math.factorial(k)
                k += 1
                if np.isclose(b[i][j], 0):
                    break
        while np.sum(v) > 0:
            i, j = np.unravel_index(b.argmax(), b.shape)
            if np.isclose(b[i][j], 0):
                return
            else:
                dis = np.inf
                vid = None
                for vid_, veh in enumerate(self.vehs):
                    if veh.idle and not veh.rebl:
                        dis_ = router.get_distance(veh.lng, veh.lat, c[i][j][0], c[i][j][1])
                        if dis_ < dis:
                            dis = dis_
                            vid = vid_
                route = [(-1, 0, c[i][j][0], c[i][j][1])]
                self.vehs[vid].build_route(router, route)
                for i_, j_ in itertools.product(range(Nlat), range(Nlng)):
                    if self.vehs[vid].lat >= Dlat - (i_+1)*Elat:
                        if self.vehs[vid].lng <= Olng + (j_+1)*Elng:
                            v[i_][j_] -= 1
                            break
                s[i][j] += 1
                lamda = d[i][j] * INT_REBL/3600
                k = int(s[i][j])
                '''
                print(k)
                print(lamda)
                print(b[i][j])
                '''
                b[i][j] -= np.exp(-lamda) * (lamda**k) / np.math.factorial(k)
                '''
                print(b[i][j])
                '''

    # get the state of a vehicle
    # a state is defined as the predicted demand, the number of vehicles and their locations, occupancy etc around a vehicle
    def get_state(self, veh):
        lng = veh.lng
        lat = veh.lat
        d = np.zeros((Mlat, Mlng))
        c = np.zeros((Mlat, Mlng,2))
        v = np.zeros((Mlat, Mlng))
        s = np.zeros((Mlat, Mlng))
        for m in self.M:
            for i,j in itertools.product(range(Mlat), range(Mlng)):
                if m[1] <= lat + Mlat*Elat/2 - i*Elat and m[1] >= lat + Mlat*Elat/2 - (i+1)*Elat:
                    if m[0] >= lng - Mlng*Elng/2 + j*Elng and m[0] <= lng - Mlng*Elng/2 + (j+1)*Elng:
                        d[i][j] += m[4] * self.D
                        c[i][j][0] += m[0] * m[4] * self.D
                        c[i][j][1] += m[1] * m[4] * self.D
                        break
        for i,j in itertools.product(range(Mlat), range(Mlng)):
            if d[i][j] != 0:
                c[i][j][0] /= d[i][j]
                c[i][j][1] /= d[i][j]
            else:
                c[i][j][0] = False
                c[i][j][1] = False
        for veh_ in self.vehs:
            if veh_.idle:
                for i,j in itertools.product(range(Mlat), range(Mlng)):
                    if veh_.lat <= lat + Mlat*Elat/2 - i*Elat and veh_.lat >= lat + Mlat*Elat/2 - (i+1)*Elat:
                        if veh_.lng >= lng - Mlng*Elng/2 + j*Elng and veh_.lng <= lng - Mlng*Elng/2 + (j+1)*Elng:
                            v[i][j] += 1
                            break
            else:
                lng_, lat_, n = veh_.get_location_at_time(self.T+INT_REBL)
                for i,j in itertools.product(range(Mlat), range(Mlat)):
                    if lat_ <= lat + Mlat*Elat/2 - i*Elat and lat_ >= lat + Mlat*Elat/2 - (i+1)*Elat:
                        if lng_ >= lng - Mlng*Elng/2 + j*Elng and lng_ <= lng - Mlng*Elng/2 + (j+1)*Elng:
                            if n == 0:
                                s[i][j] += 0.8
                            elif n == 1:
                                s[i][j] += 0.4
                            elif n == 2:
                                s[i][j] += 0.2
                            elif n == 3:
                                s[i][j] += 0.1
                            else:
                                s[i][j] += 0.0
                            break
        return [d,v,s], c

    # visualize
    def draw(self):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(5,6))
        plt.xlim((-0.02,0.18))
        plt.ylim((51.29,51.44))
        for veh in reversed(self.vehs):
            veh.draw()
        for req in self.queue:
            req.draw()
        plt.show()

    def __str__(self):
        str = "AMoD system at t = %.3f: %d requests, in which %d in queue" % ( self.T, self.N-1, len(self.queue) )
        # for r in self.queue:
        #     str += "\n" + r.__str__()
        return str
