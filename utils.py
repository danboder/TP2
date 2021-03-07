import matplotlib.pyplot as plt
import math


def read_instance(filename):
    """
    Load a VRP instance
    :param filename: name of the file which stores the instance (string). The format is the following
                       n k W
                       d_0 x_0 y_0
                       d_1 x_1 y_1
                       ...
                       d_(n-1) x_(n-1) y_(n-1)
    :return: a tuple (n,k,W,points) where
                 - n is the number of points (storage facility included) (int)
                 - k is the number of vehicle available (int)
                 - W is the max weight available by vehicle (int)
                 - points is the list of all the descriptions of the
                        points (demand d_i (int), coordinate (x_i,y_i) (tuple of floats)) (list of tuples (d_i, (x_i,y_i)))
    """
    with open(filename) as file:
        hline = file.readline().strip().split()
        n = int(hline[0])
        k = int(hline[1])
        W = int(hline[2])
        points = []
        for _ in range(n):
            line = file.readline().strip().split()
            points.append((int(line[0]), (float(line[1]), float(line[2]))))
        return n, k, W, points


def write_solution(filename, totValue, routes):
    """
    Write the solution in a file
    :param filename: name of the file which will store the solution (string). The format is the following
                       totValue
                       0 c_1_0 c_2_0 ... 0
                       0 c_1_1 c_2_1 ... 0
                       ...
                       0 c_1_(k-1) c_2_(k-1) ... 0
    :param totValue: the value of the solution (float)
    :param routes: the list of the routes (list of list of int)
    :return: nothing
    """
    with open(filename, "w") as file:
        file.write(str(totValue))
        for route in routes:
            file.write('\n')
            file.write(" ".join([str(i) for i in route]))


def read_solution(filename):
    """
    Load a solution to the VRP instance
    :param filename: name of the file which stores the solution (string). The format is the following
                       totValue
                       0 c_1_0 c_2_0 ... 0
                       0 c_1_1 c_2_1 ... 0
                       ...
                       0 c_1_(k-1) c_2_(k-1) ... 0
    :return: a tuple (totValue, routes) where totValue is the cost value of the solution (float) and
             routes is a list of list of int describing the routes of the vehicules
    """
    with open(filename) as file:
        totValue = int(file.readline().strip())
        routes = []
        for route in file.readlines():
            routes.append([int(i) for i in route.strip().split()])
        return totValue, routes


def draw_solution(filename, points, routes, totValue):
    """
    Create a graph representing the solution
    :param filename: name of the file which will store the visual representation of a solution (string).
    :param points: points is the list of all the descriptions of the
                   points (demand d_i (int), coordinate (x_i,y_i) (tuple of floats)) (list of tuples (d_i, (x_i,y_i)))
    :param routes: a list of list of int describing the routes of the vehicules
    :param totValue: the value of the solution (float)
    :return: draw a graph representing the solution
    """
    plt.clf()
    plt.figure(figsize=(15, 15))
    plt.title("Solution with a cost of "+str(totValue))

    for id in range(len(points)):
        (c, (x, y)) = points[id]
        plt.plot(x, y, "co", color="m")
        plt.text(x, y, "L%d" % id, color="b", fontsize=8)

    # cycle_color = ["b", "g", "r", "c", "m", "y", "k", "w"]
    cycle_color = ["b", "g", "r", "c", "m", "y", "k", "grey"]
    for route_id in range(len(routes)):
        route = routes[route_id]
        route_color = cycle_color[route_id % len(cycle_color)]
        for id in range(len(route) - 1):
            (c0, (x0, y0)) = points[route[id]]
            (c1, (x1, y1)) = points[route[id + 1]]
            plt.plot([x0, x1], [y0, y1], color=route_color)
    plt.savefig(filename)


def dist(p1, p2):
    """
    Compute the euclidean distance between two points
    :param p1: first point (tuple of float (x,y))
    :param p2: first point (tuple of float (x,y))
    :return: the euclidean distance between the two (float)
    """
    return math.dist([p1[0], p1[1]], [p2[0], p2[1]])


def is_valid_solution(n, k, W, points, totValue, routes):
    """
    Verify the validity of a solution, wrt the input
    :param n: total number of point (storage included) (int)
    :param k: total number of vehicule (int)
    :param W: max weight available by vehicle (int)
    :param points: list of all the descriptions of the
                        points (demand d_i (int), coordinate (x_i,y_i) (tuple of floats)) (list of tuples (d_i, (x_i,y_i)))
    :param totValue: the value of the solution (float)
    :param routes: the list of the routes (list of list of int)
    :return:
    """
    cost = 0
    places = set()
    if len(routes) != k:
        print("The number of routes is not right")
        return False
    for route in routes:
        if route[0] != 0 or route[-1] != 0:
            print("Routes should start and stop at the storage facility")
            return False
        if 0 in route[1:-1]:
            print("Storage should only be visited at start and end of routes")
            return False
        w = 0
        for i in range(1, len(route)):
            cost += dist(points[route[i - 1]][1], points[route[i]][1])
            places.add(route[i])
            w += points[route[i]][0]
        if w > W:
            print("The weight of a route is exceeding the max")
            return False

    if len(places) != n:
        print("Not all places are visited")
        return False
    if min(places) != 0 or max(places)!= n-1:
        print("The routes visits places which does not exists")
        return False
    if cost != totValue:
        print("The cost of the solution is not right (cost given: {0}, cost recomputed: {1})".format(totValue,cost))
        return False

    return True


if __name__ == '__main__':
    ins = read_instance("./instances/test")
    sol = read_solution("./instances/testSol")
    draw_solution("sol.png", ins[3], sol[1],sol[0])
