import utils
import solver
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument('--infile', type=str, default='instances/test')
    parser.add_argument('--outfile', type=str, default='output')
    parser.add_argument('--visufile', type=str, default='sol.png')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    print("***********************************************************")
    print("[INFO] Start the solving of the VRP instance")
    print("[INFO] input file: %s" % args.infile)
    print("[INFO] output file: %s" % args.outfile)
    print("[INFO] visu file: %s" % args.visufile)
    print("***********************************************************")

    n, k, W, points = utils.read_instance(args.infile)

    totValue, routes = solver.solve_naive(n, k, W, points)
    # totValue, routes = solver.solve_advance(n, k, W, points)

    utils.is_valid_solution(n, k, W, points, totValue, routes)

    utils.write_solution(args.outfile, totValue, routes)
    utils.draw_solution(args.visufile, points, routes, totValue)
