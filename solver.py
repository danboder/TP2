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
    
    def distance(self,other):
        return math.dist((self.x,self.y),(other.x,other.y))
class Client():
    def __init__(self, id, demande, point):
        self.id = id
        self.demande = demande
        self.point = point

    def __str__(self):
        return f"Id: {self.id}, demande: {self.demande}, Point: {str(self.point)}"


def getListClosest(pt,clients):
    sorted_list = sorted(clients, key = lambda c: c.point.distance(pt))
    if len(clients) <= 2:
        return list(map(lambda c : c.id, sorted_list))
    closest = sorted_list.pop(0)
    return [closest.id] + getListClosest(closest.point,sorted_list)

def random_routes(k,w,clients):
    routes = [[0] for _ in range(k)]
    weights = [0 for _ in range(k)]
    for p in clients:
        if p.id != 0:
            classed = False
            while not classed:
                i = random.randint(0,k-1)
                if weights[i] + p.demande <= w:
                    routes[i].append(p.id)
                    weights[i] += p.demande
                    classed = True
    for r in routes:
        r.append(0)
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
    for i in range(len(routes)):
        route = two_opt(routes[i],clients,nb_iterations)
        routes[i] = route
    return routes



def generate_neighboorhood(routes):
    # for each client, create a neighbor where the client is on another vehicle client list
    neighbors = []

    for i in range(len(routes)):
        for j in range(1,len(routes[i])-1):
            new_routes = copy.deepcopy(routes)
            id = new_routes[i].pop(j)
            rand = random.randint(0,len(routes)-1)
            while rand == i: rand = random.randint(0,len(routes)-1)
            list.insert(new_routes[i],1,id)
            neighbors.append(new_routes)
    
    return neighbors

def validate_neighboorhood(neighbors,clients,w):
    list_to_remove = []
    for i,neighbor in enumerate(neighbors):
        for r in neighbor:
            sum = 0
            for c in r:
                sum += clients[c].demande
            if sum > w: list_to_remove.append(i)
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

    # for c in clients:
    #     print(c)
    initial_routes = random_routes(k,W,clients)

    nb_iterations = 1000
    T = 1
    alphaT = 0.97
    s = initial_routes
    fs = getCost(s,clients)
    star = s
    fstar = fs

    iterations_for_2opt = 30
    for k in range(nb_iterations):
        if k % 100 == 0: print(fstar)
        G = generate_neighboorhood(s)
        # print(G)
        V = validate_neighboorhood(G, clients, W)
        # print(V)
        c = V[random.randint(0,len(V)-1)]
        c = optimize_route(c,clients,iterations_for_2opt)
        fc = getCost(c,clients)
        delta = fc - fs
        if delta <= 0 or random.random() < math.exp(-delta/T):
            s = c
            fs = fc
        if fs < fstar:
            star = s
            fstar = fs
            T = alphaT * T

    # print(star)
    print("--- %s seconds ---" % (time.time() - start_time))
    return fstar, star
