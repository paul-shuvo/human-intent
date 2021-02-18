# from sympy import Point3D, Line, Plane
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())
from time import time
from fg._3d import Plane, Line3D, Point3D
import numpy as np 

import pickle
with open('plane.pkl', 'rb') as f:
    pt_list = pickle.load(f)


points = np.array(pt_list)
s = 1
# print(pp())

# n = time()
# l1 = Line((3,5,-2), (1,-1,4))
# p = Point3D((3, 7, 9))
# print(l1)
# # print(f'line: {time()-n}')

# n = time()
# p1 = Plane((3,1,-10),(4,4,1),(3,5,-20))
# print(p1)
# # print(f'plane: {time()-n}')

# n = time()
# intersection1 = [float(i) for i in p1.intersection(l1)[0].args]
# print(f'sym in: {time()-n}')
# print(f'intersection: {intersection1}')
# import numpy as np
# now = time()

# pl1 = np.array([1,-1,4])
# pl2 = np.array([3,5,-2])

# # pl1 = np.array([1,1,1])
# # pl2 = np.array([3,3,3])

# l_eq = np.vstack((pl1 , (pl2-pl1)))


# pp1 = np.array([3,1,-10])
# pp2 = np.array([4,4,1])
# pp3 = np.array([3,5,-20])

# # pp1 = np.array([2,1,4])
# # pp2 = np.array([4,-2,7])
# # pp3 = np.array([5,3,-2])

# n = np.cross(pp2-pp1, pp3-pp1)
# # n = n/np.linalg.norm(n)
# print(f'n is: {n}')
# p_eq = np.vstack((pp1, n))
# d = np.dot(n,pp1)
# c, t_sum = np.dot(n, l_eq.T)

# t = (d-c)/t_sum


# intersection = l_eq[0] + l_eq[1]*t
# print(intersection)
# print(f'np in: {time()-now}')