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

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __str__(self):
        return "("+str(self.x)+","+str(self.y)+")"
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def distance(self,other):
        return math.dist((self.x,self.y),(other.x,other.y))
class Client():
    def __init__(self, id, demande, point):
        self.id = id
        self.demande = demande
        self.point = point

    def __str__(self):
        return f"Id: {self.id}, demande: {self.demande}, Point: {str(self.point)}"

def closestPoint(pt,point):
    dist = 1e10
    closest = None
    for i,p in enumerate(point):
        d = p.distance(pt)
        if d < dist:
            dist = d
            closest = i
    return closest



def kMeans(k,W,clients):
    centroids = []
    for _ in range(k):
        centroids.append(Point(random.random()*200 - 100, random.random()*200 - 100)) # create k random points in space (x and y between -100 and 100)
    clients_belongs_to = [closestPoint(pt,centroids) for pt in map(lambda c: c.point, clients)]
    iterations = 0
    while iterations < 10:
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
                new_centroid = Point(si[0]/ni,si[1]/ni)
            else:
                new_centroid = Point(0,0)
            if centroids[i] != new_centroid:
                centroids[i] = new_centroid
        clients_belongs_to = [closestPoint(pt,centroids) for pt in map(lambda c: c.point, clients)]
        iterations += 1
    
    routes = [[0] for _ in range(k)]
    for i,index_centroid in enumerate(clients_belongs_to):
        if i != 0:
            routes[index_centroid].append(clients[i].id)
    for r in routes:
        r.append(0)
    return routes



def getListClosest(pt,clients):
    sorted_list = sorted(clients, key = lambda c: c.point.distance(pt))
    if len(clients) <= 2:
        return list(map(lambda c : c.id, sorted_list))
    closest = sorted_list.pop(0)
    return [closest.id] + getListClosest(closest.point,sorted_list)

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

def getCost(routes,clients):
    cost = 0
    for r in routes:
        for i in range(1, len(r)):
            p1 = clients[r[i-1]].point
            p2 = clients[r[i]].point
            cost += p1.distance(p2)
    return cost

def two_opt(r,clients,nb_iterations):
    best = r
    cost = getCost([r],clients)
    route = r[1:-1]
    for _ in range(nb_iterations):
        for permut in permutations(range(len(route)),2):
            save = route[permut[0]]
            route[permut[0]] = route[permut[1]]
            route[permut[1]] = save
            new_route = [0] + route + [0]
            new_cost = getCost([new_route],clients)
            if new_cost < cost:
                best = new_route
                cost = new_cost
    return best

def optimize_route(routes,clients,nb_iterations): 
    # 2 opt, but with some tweaks
    for i in range(len(routes)):
        r = routes[i]
        best = r
        cost = getCost([r],clients)
        route = r[1:-1]
        for _ in range(nb_iterations):
            p = list(permutations(range(len(route)),2))
            random.shuffle(p)
            for permut in p:
                save = route[permut[0]]
                route[permut[0]] = route[permut[1]]
                route[permut[1]] = save # we keep the 2opt for the next one
                new_route = [0] + route + [0]
                new_cost = getCost([new_route],clients)
                if new_cost < cost:
                    best = new_route
                    cost = new_cost
        routes[i] = best
    return routes

def optimize_route2(routes,clients,nb_iterations): # real 2 opt
    for i in range(len(routes)):
        r = routes[i]
        best = r
        cost = getCost([r],clients)
        route = r[1:-1]
        for _ in range(nb_iterations):
            for permut in permutations(range(len(route)),2): # try all permutations
                new_route = copy.deepcopy(route)
                to_switch = new_route[permut[0]:permut[1]+1]
                l = permut[1]-permut[0]
                for s in range(l):
                    new_route[permut[0]+s] = to_switch[l-1-s]
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

def validate_neighboorhood(neighbors,clients,w):
    list_to_remove = []
    for i,neighbor in enumerate(neighbors):
        for r in neighbor:
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
    # get time
    start_time = time.time()

    clients = [Client(i,p[0],Point(*p[1],)) for i,p in enumerate(points)]
    # id = 0 is storage

    V = []
    while(len(V)==0):
        star = kMeans(k,W,clients)
        G = generate_neighboorhood(star)
        V = validate_neighboorhood(G, clients, W)
    s = V[random.randint(0,len(V)-1)]

    # s = greedy_routes(k,W,clients)

    # nb_iterations = 1000
    T = 30
    alphaT = 0.9

    fs = getCost(s,clients)
    star = s
    fstar = fs

    iterations_for_2opt = 5
    # for k in range(nb_iterations):
    k = 0
    execution_time = 5 # in minutes
    change = True
    while time.time() - start_time < execution_time * 60:
        if change:
            # if we keep the same s, no use to redo generation and validation
            G = generate_neighboorhood(s)
            V = validate_neighboorhood(G, clients, W)
            change = False
        c = V[random.randint(0,len(V)-1)]
        c = optimize_route2(c,clients,iterations_for_2opt)
        fc = getCost(c,clients)
        delta = fc - fs
        if delta <= 0 or random.random() < math.exp(-delta/T):
            s = c
            fs = fc
            if fs < fstar:
                star = s
                fstar = fs
                T = alphaT * T
                print(str(k),fstar)
            change = True
        k += 1

    print("before opt",fstar)
    star = optimize_route2(star,clients,50)
    fstar = getCost(star,clients)
    print("after opt",fstar)
    print("--- %s seconds ---" % (time.time() - start_time).__round__())
    return fstar, star
