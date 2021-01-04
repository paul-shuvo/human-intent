import numpy as np
import matplotlib.pyplot as plt
def points_on_triangle(v, n):
    x = np.sort(np.random.rand(2, n), axis=0)
    return np.column_stack([x[0], x[1]-x[0], 1.0-x[1]]) @ v
v = np.array([[0,0],[0,10],[10,10],[10,0]])

tri_2 = points_on_triangle(v[1:], 5)
tri_1 = points_on_triangle(v[:3], 5)
tri_merge = np.vstack((tri_1, tri_2))
print(f'{tri_1}\n{tri_2}\n{tri_merge}')
# print(points_on_triangle(v[1:], 10))
