import numpy as np

class Line:
    def __init__(self, points):
        self.p = points[0]
        self.v = points[1] - points[0]
    
    # https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    def intersect(self, geoentity):
        t = np.cross((geoentity.p - self.p), geoentity.v / np.cross(self.v, geoentity.v))
        # print(f'p is: {self.p}\nv is: {self.v}\nt is: {t}\npoint is: {self.p + self.v * t}')
        return self.p + self.v * t
        
        

# l1 = Line(np.array([[0,0], [5,5]]))
# l2 = Line(np.array([[0,5], [5,0]]))
# print(l1.intersect(l2))