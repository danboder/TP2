import math
import utils

import random
from itertools import permutations
import copy
import time

def solve_naive(n, k, W, points):
    """
    Greedy solution of the problem
    :param n: total number of point (storage included) (int)
    :param k: total number of vehicule (int)
    :param W: max weight available by vehicle (int)
    :param points: list of all the descriptions of the
                        points (demand d_i (int), coordinate (x_i,y_i) (tuple of floats)) (list of tuples (d_i, (x_i,y_i)))
    :return: a tuple (totValue, routes) where totValue is the cost value of the solution (float) and
             routes is a list of list of int describing the routes of the vehicules
    """
    storage = points[0]
    id_list = [(i, points[i]) for i in range(len(points))][1:]
    id_list = sorted(id_list,
                     key=lambda point: math.atan2(point[1][1][1] - storage[1][1], point[1][1][0] - storage[1][0]))
    print(id_list)
    routes = []
    cost = 0
    for i in range(k):
        routes.append([0])
        w = W
        j = 0
        while w > 0 and j < len(id_list):
            if w - id_list[j][1][0] >= 0:
                w -= id_list[j][1][0]
                routes[-1].append(id_list[j][0])
                id_list = id_list[:j] + id_list[j + 1:]
                cost += utils.dist(points[routes[-1][-2]][1], points[routes[-1][-1]][1])
            else:
                j += 1
        routes[-1].append(0)
        cost += utils.dist(points[routes[-1][-2]][1], points[routes[-1][-1]][1])

    return cost, routes

###################################
# CLASSES
###################################
class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __str__(self):
        return "("+"{:.1f}".format(self.x)+","+"{:.1f}".format(self.y)+")"
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    def __repr__(self):
        return self.__str__()
    
    def distance(self,other):
        return math.dist((self.x,self.y),(other.x,other.y))
class Client():
    def __init__(self, id, demande, point):
        self.id = id
        self.demande = demande
        self.point = point
    def __str__(self):
        return f"Id: {self.id}, demande: {self.demande}, Point: {str(self.point)}"
    def __repr__(self):
        return self.__str__()

###################################
# UTILS
###################################

# return True if routes are valid based on the vehicle capacity
def areRoutesValid(routes,W,clients):
    for r in routes:
        sum_w = 0
        for c in r:
            sum_w += clients[c].demande
        if sum_w > W: return False
    return True

# returns the index of a list of points of the closest to pt  
def closestPoint(pt,points):
    dist = 1e10
    closest = None
    for i,p in enumerate(points):
        d = p.distance(pt)
        if d < dist:
            dist = d
            closest = i
    return closest

# get the cost of one solution
def getCost(routes,clients):
    cost = 0
    for r in routes:
        for i in range(1, len(r)):
            p1 = clients[r[i-1]].point
            p2 = clients[r[i]].point
            cost += p1.distance(p2)
    return cost


###################################
# FUNCTIONS
###################################

# get some possible routes based on greedy algorithm
def greedy_routes(k,W,clients):
    routes = []
    for i in range(k):
        routes.append([0])
        w = W
        j = 1
        while w > 0 and j < len(clients):
            if w - clients[j].demande >= 0:
                w -= clients[j].demande
                routes[-1].append(clients[j].id)
                clients = clients[:j] + clients[j + 1:]
            else:
                j += 1
        routes[-1].append(0)
    return routes

def kMeans(k,W,clients):

    nb_tries_before_balancing = 50
    for _ in range(nb_tries_before_balancing):
        centroids = []
        for _ in range(k):
            centroids.append(Point(random.random()*200 - 100, random.random()*200 - 100)) # create k random points in space (x and y between -100 and 100)
        clients_belongs_to = [closestPoint(pt,centroids) for pt in map(lambda c: c.point, clients)]
        iterations = 0
        while iterations < 30:
            sum_of_pos = [[0,0] for _ in range(k)]
            nb_per_centroid = [0 for _ in range(k)]
            for i in range(len(clients)):
                index_belongingPoint = clients_belongs_to[i]
                sum_of_pos[index_belongingPoint][0] += clients[i].point.x
                sum_of_pos[index_belongingPoint][1] += clients[i].point.y
                nb_per_centroid[index_belongingPoint] += 1
            for i in range(k):
                ni = nb_per_centroid[i]
                si = sum_of_pos[i]
                if ni != 0:
                    centroids[i] = Point(si[0]/ni,si[1]/ni)
            clients_belongs_to = [closestPoint(pt,centroids) for pt in map(lambda c: c.point, clients)]
            iterations += 1
        
        routes = [[0] for _ in range(k)]
        for i,index_centroid in enumerate(clients_belongs_to):
            if i != 0:
                routes[index_centroid].append(clients[i].id)
        for r in routes:
            r.append(0)

        if areRoutesValid(routes,W,clients): 
            return routes


    # if routes not valid compared to total capacity of vehicles
    # get the difference to total capacity
    place_left = []
    index_place_left = []
    for i,r in enumerate(routes):
        s = 0
        for c in r:
            s += clients[c].demande
        diff = W-s
        place_left.append(diff)
        if diff > 0:
            index_place_left.append(i)

    # print(centroids)
    # get centroids of clusters having space left
    centroids_available = []
    for i in index_place_left:
        centroids_available.append(centroids[i])

    for i in range(len(routes)):
        while place_left[i] < 0: # where we exceed limit
            # max_value = max(routes[i][1:-1],key=lambda c:clients[c].demande) # get client with max demande
            max_value = max(routes[i][1:-1],key=lambda c: centroids[i].distance(clients[c].point)) # get client with max distance
            index = routes[i].index(max_value)
            # index = random.randint(1,len(routes[i])-1) # get random client
            c = routes[i].pop(index) # remove the client from route
            d = clients[c].demande # demand of that client
            
            # insert into the closest cluster ( = route ) with space
            inserted = False
            point_client = clients[c].point
            # print(list(map(lambda r: list(map(lambda c: clients[c].demande,r)),routes)))
            # print(point_client)
            # print("DEMANDE",d)
            # print(place_left)
            centroids_available_copy = copy.deepcopy(centroids_available)

            while not inserted:
                # print(centroids_available_copy)
                # get the closest centroid_available from the client
                index_centroid_available = closestPoint(point_client,centroids_available_copy)
                if index_centroid_available == None:
                    return False
                closest = centroids_available_copy[index_centroid_available]
                # print("closest",closest)

                # index = random.randint(0,len(index_place_left)-1) # insert in another route (than has weight < W) by random
                # find the route related to that centroid
                index = centroids.index(closest)
                # print("index",index)
                if place_left[index] - d >= 0: # if it doesn't go above W
                    list.insert(routes[index],1,c) # insert in route
                    inserted = True
                    place_left[index] -= d
                    place_left[i] += d      # update the place left list
                    if place_left[i] > 0:
                        index_place_left.append(i)
                else:
                    # if it goes above total capacity W, remove that centroid from the ones in available_copy
                    centroids_available_copy.pop(index_centroid_available)

    return routes

def optimize_route(routes,clients,nb_iterations): 
    # 2 opt
    for i in range(len(routes)):
        r = routes[i]
        best = r
        cost = getCost([r],clients)
        route = r[1:-1]
        for _ in range(nb_iterations):
            for permut in permutations(range(len(route)),2): # try all permutations
                new_route = copy.deepcopy(route)
                # change the order between the 2 points
                to_switch = new_route[permut[0]:permut[1]+1] # = change the order of to_switch
                to_switch.reverse()
                # l = permut[1]-permut[0]
                # for s in range(l):
                #     new_route[permut[0]+s] = to_switch[l-1-s]
                new_route = new_route[:permut[0]] + to_switch + new_route[permut[1] + 1:]
                new_route = [0] + new_route + [0]
                new_cost = getCost([new_route],clients)
                if new_cost < cost:
                    best = new_route
                    route = best[1:-1]
                    cost = new_cost
            # edge cases : if we want to switch with the storage ==> we slide the whole list
            for t in range(1,len(route)):
                new_route = route[t:] + route[:t]
                new_route = [0] + new_route + [0]
                new_cost = getCost([new_route],clients)
                if new_cost < cost:
                    best = new_route
                    route = best[1:-1]
                    cost = new_cost
        routes[i] = best
    return routes



def generate_neighboorhood(routes):
    # for each client
    # we create a new neighbor where that client can be in any other vehicle client list
    # at any position in the list
    neighbors = []
    for i in range(len(routes)):
        for j in range(1,len(routes[i])-1): # for each client
            for k in range(len(routes)): # for each vehicle
                if k != i: # we don't put the client in the same vehicle client list
                    for insert_at in range(1,len(routes)-1): # at each place in the list
                        new_routes = copy.deepcopy(routes)
                        id = new_routes[i].pop(j) # remove the client from inial positon
                        list.insert(new_routes[k],insert_at,id) # insert in new position
                        neighbors.append(new_routes)
    return neighbors

def generate_neighboorhood_faster(routes,W,clients):
    # for each client we create a new neighbor where that client can be in any other vehicle (taken at random) at a random position
    # as we don't create a neighbor for each case, this function is faster than the previous one
    # also we validate the neighbor before adding it, so no need to call validate neighborhood after
    neighbors = []
    for i in range(len(routes)):
        for j in range(1,len(routes[i])-1): # for each client
            insert_in = [var_ for var_ in range(len(routes))]
            insert_in.pop(i)
            k = random.choice(insert_in) # random route to insert client
            # sum = 0
            # r = routes[k]
            # for c in r:
            #     sum += clients[c].demande
            # if sum + clients[routes[i][j]].demande <= W and r[0] == 0 and r[-1] == 0:
            insert_at = random.randint(1,len(routes)-2)
            new_routes = copy.deepcopy(routes)
            id = new_routes[i].pop(j) # remove the client from inial positon
            list.insert(new_routes[k],insert_at,id) # insert in new position random
            neighbors.append(new_routes)
    return neighbors


def validate_neighboorhood(neighbors,clients,w):
    list_to_remove = []
    for i,neighbor in enumerate(neighbors):
        for r in neighbor:
            if r[0] != 0 or r[-1] != 0:
                list_to_remove.append(i)
                break
            sum = 0
            for c in r:
                sum += clients[c].demande
            if sum > w: 
                list_to_remove.append(i)
                break
    for i in range(len(list_to_remove)-1,-1,-1):
        neighbors.pop(list_to_remove[i])
    return neighbors



def solve_advance(n, k, W, points):
    """
    Advanced solution of the problem
    :param n: total number of point (storage included) (int)
    :param k: total number of vehicule (int)
    :param W: max weight available by vehicle (int)
    :param points: list of all the descriptions of the
                        points (demand d_i (int), coordinate (x_i,y_i) (tuple of floats)) (list of tuples (d_i, (x_i,y_i)))
    :return: a tuple (totValue, routes) where totValue is the cost value of the solution (float) and
             routes is a list of list of int describing the routes of the vehicules
    """
    # TODO implement here your solution

    total_time = time.time()

    clients = [Client(i,p[0],Point(*p[1],)) for i,p in enumerate(points)]
    # id = 0 is storage

    best_s = None
    best_f = 1e10

    nb_restart = 30
    for _ in range(nb_restart):
        print("START")
        # get time
        start_time = time.time()


        s = kMeans(k,W,clients)
        while s == False:
            s = kMeans(k,W,clients)
        # s = greedy_routes(k,W,clients)
        s = optimize_route(s,clients,15)

        T = 5
        maxT = T
        alphaT = 0.97
        betaT = 2 # scaling coefficient for reheating

        fs = getCost(s,clients)
        star = s
        fstar = fs

        re_count = 0
        re_lim = 30

        iterations_for_2opt = 1
        i = 0
        execution_time = 10 / nb_restart # in minutes
        # execution_time = 0.5 # in minutes
        change = True
        while time.time() - start_time < execution_time * 60:
            # if i % 500 == 0: print(i,fs)
            if re_count >= re_lim:
                # if restart limit has be reached, we regenerate neighborhood on best solution
                # Restart on previous best's neighborhood
                # G = generate_neighboorhood(star)
                re_count = 0 # reset counter
                T = min(T + betaT, maxT) # "reheat the algorithm" = increase T, but shouldn't go too hot
                print(T)
            if change:
                # if we keep the same s, no use to redo generation and validation
                # G = generate_neighboorhood_faster(s,W,clients)
                G = generate_neighboorhood(s)
                V = validate_neighboorhood(G, clients, W)
                change = False
            c = V[random.randint(0,len(V)-1)]
            c = optimize_route(c,clients,iterations_for_2opt)
            fc = getCost(c,clients)
            delta = fc - fs
            if delta <= 0 or random.random() < math.exp(-delta/T):
                change = True
                s = c
                fs = fc
                if fs < fstar:
                    # if we improve the best solution, restart counter resets
                    re_count = 0
                    star = s
                    fstar = fs
                    print(i,fstar,T)
            else:
                # count up to restart limit
                re_count += 1
            T = alphaT * T
            i += 1

        star = optimize_route(star,clients,2) # last optimization
        fstar = getCost(star,clients)
        if fstar < best_f:
            print("NEW BEST",fstar)
            best_f = fstar
            best_s = star

        print("--- %s seconds ---" % (time.time() - start_time).__round__())

    print("--- Total Time %s seconds ---" % (time.time() - total_time).__round__())

    return best_f, best_s
