import math
import utils


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
    return solve_naive(n, k, W, points)
