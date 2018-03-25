# -*- coding: utf-8 -*-
from get_data import cities
legal_cities = cities
from math import *
import numpy as np

def get_distance(Lat_A, Lng_A, Lat_B, Lng_B): #第一种计算方法
    ra=6378.140 #赤道半径
    rb=6356.755 #极半径 （km）
    flatten=(ra-rb)/ra  #地球偏率
    rad_lat_A=radians(Lat_A)
    rad_lng_A=radians(Lng_A)
    rad_lat_B=radians(Lat_B)
    rad_lng_B=radians(Lng_B)
    pA=atan(rb/ra*tan(rad_lat_A))
    pB=atan(rb/ra*tan(rad_lat_B))
    xx=acos(sin(pA)*sin(pB)+cos(pA)*cos(pB)*cos(rad_lng_A-rad_lng_B))
    c1=(sin(xx)-xx)*(sin(pA)+sin(pB))**2/cos(xx/2)**2
    c2=(sin(xx)+xx)*(sin(pA)-sin(pB))**2/sin(xx/2)**2
    dr=flatten/8*(c1-c2)
    distance=ra*(xx+dr)
    return distance

#get locations
city_locate = {}
with open("locate", "r", encoding="utf8") as f:
    for line in f.readlines():
        line = line.strip()
        r = line.split()
        if len(r) != 3:
            continue
        name, l1, l2 = r
        city_locate[name] = list(map(float, (l1[2:], l2[2:])))

for city in cities:
    if city not in city_locate:
        print(city)
        import ipdb; ipdb.set_trace()

distance = {}
for i in range(len(cities)):
    for j in range(i+1, len(cities)):
        l11, l12 = city_locate[cities[i]]
        l21, l22 = city_locate[cities[j]]
        d = get_distance(l11, l12, l21, l22)
        distance[(i, j)] = distance[(j, i)] = d


def build_kernel(f=None):
    # f [city_nums, 3]
    k = np.zeros((len(cities), len(cities)))
    for i in range(len(cities)):
        for j in range(i+1, len(cities)):
            d1 = distance[(i, j)] ** 2 / 100 ** 2 / 2
            if f is not None:
                d1 = d1 + np.linalg.norm(f[i] - f[j]) ** 2/ 100 ** 2 /2
            k[i, j] = k[j, i] = np.exp(-d1)
    return k