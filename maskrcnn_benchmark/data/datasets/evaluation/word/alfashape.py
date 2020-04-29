import numpy as np
from scipy.spatial import Delaunay
from .area_of_polygon import area_of_polygon_crd
import networkx as nx

def sqrt_sum(a, b):
    x = (a[0]-b[0])
    y = (a[1]-b[1])
    return np.sqrt(x*x+y*y)

def shapeToSomePolygons(shape):
    G = nx.Graph()
    allnodes = set()
    for line in shape:
        G.add_nodes_from(line)
        G.add_edge(line[0], line[1])
        allnodes.add(line[0])
        allnodes.add(line[1])

    result = []

    while allnodes:
        node = allnodes.pop()
        new_node = next(iter(G[node]), None)
        if not new_node: continue

        G.remove_edge(node, new_node)
        temp = nx.shortest_path(G, node, new_node)
        for j,t in enumerate(temp):
            if t in allnodes:
                allnodes.remove(t)
        result.append(temp)
    return result

def getAlfaShapes(pts,alfas=1):
    tri_ind = [(0,1),(1,2),(2,0)]
    tri = Delaunay(pts)
    lenghts={}
    for s in tri.simplices:
        for ind in tri_ind:
            a = pts[s[ind[0]]]
            b = pts[s[ind[1]]]
            # print('a---', a)
            # print('b---', b)
            line = (a, b)
            # line = ((a[0], a[1]), (b[0], b[1]))
            lenghts[line] = sqrt_sum(a, b)

    ls = sorted(lenghts.values())

    mean_length = np.mean(ls)
    mean_length_index = ls.index(next(filter(lambda x: x>=mean_length, ls)))
    magic_numbers = [ls[i] for i in range(mean_length_index, len(ls))]
    magic_numbers[0] = 0
    sum_magic = np.sum(magic_numbers)
    for i in range(2, len(magic_numbers)):
        magic_numbers[i] += magic_numbers[i-1]
    magic_numbers = [m /sum_magic for m in magic_numbers]

    rez = []
    for alfa in alfas:
        i = magic_numbers.index(next(filter(lambda z: z > alfa, magic_numbers), magic_numbers[-1]))
        av_length = ls[mean_length_index+i]

        lines = {}

        for s in tri.simplices:
            used = True
            for ind in tri_ind:
                if lenghts[(pts[s[ind[0]]], pts[s[ind[1]]])] > av_length:
                    used = False
                    break
            if used == False: continue

            for ind in tri_ind:
                i,j= s[ind[0]],s[ind[1]]
                line = (pts[min(i,j)], pts[max(i,j)])
                lines[line] = line in lines

        good_lines = []
        for v in lines:
            if not lines[v]:
                good_lines.append(v)

        result = shapeToSomePolygons(good_lines)
        result.sort(key=area_of_polygon_crd, reverse=True)
        rez.append(result)
    return rez

