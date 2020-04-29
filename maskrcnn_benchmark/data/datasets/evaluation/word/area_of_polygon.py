import numpy as np

def area_of_polygon_xy(x, y):
    """Calculates the area of an arbitrary polygon given its verticies"""
    area = 0.0
    for i in range(-1, len(x)-1):
        area += x[i] * (y[i+1] - y[i-1])
    return abs(area) / 2.0

def area_of_polygon_crd(cordinates):
    """Calculates the area of an arbitrary polygon given its verticies"""
    x = [v[0] for v in cordinates]
    y = [v[1] for v in cordinates]
    return area_of_polygon_xy(x,y)

def area_of_polygon(**kwargs):
    if 'x' in kwargs and 'y' in kwargs:
        x = kwargs['x']
        y = kwargs['y']
        return area_of_polygon_xy(x, y)

    if 'coordinates' in kwargs:
        cordinates = kwargs['coordinates']
        return area_of_polygon_crd(cordinates)

    print("Wrong parameters")
    return None

def length_of_way(cordinates):
    """Length of the way"""
    if len(cordinates)<2:
        return 0
    leng = 0
    for i in range(1,len(cordinates)):
        crd = cordinates
        dist = distance(crd[i-1],crd[i-1])
        leng = leng + dist
    return leng