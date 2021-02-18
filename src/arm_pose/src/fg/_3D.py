import numpy as np


class Point3D:
    def __init__(self, coordinates: np.ndarray):
        # shape is (sample x 3)
        assert isinstance(coordinates, np.ndarray), f'coordinates need to be of type ' \
                                                    f'numpy.ndarray, but found {type(coordinates)}'
        assert coordinates.shape[-1] == 3, f'Point should have 3 coordinate ' \
                                           f'values, got{coordinates.shape[-1]}'
        assert len(coordinates.shape) in [1,2], f'coordinates should have either 1 or 2 dimensions ' \
                                                f'for single and multiple points respectively, ' \
                                                f'not {len(coordinates.shape)}'

        if len(coordinates.shape) == 1:
            coordinates = np.expand_dims(coordinates, axis=0)

        self.points = coordinates

        self.shape = self.points.shape

    def __sub__(self, other):
        return Point3D(self.points - other.points)

    def __add__(self, other):
        return Point3D(self.points + other.points)

class Line3D:
    def __init__(self, points):
        # output shape is (sample x 2 x 3), 2 points needed for a line
        # [:,0,:] is the root of the line
        # [:,1,:] is the other point

        # check if it's the same point

        assert isinstance(points, (np.ndarray, list)), f'Points need to be either list of Point3D objects' \
                                                       f'or numpy.ndarray, found type: {type(points)}'

        if isinstance(points, list):
            assert len(points) == 2, f'The list should contain two sets of points, ' \
                                     f'found {len(points)}'
            assert points[0].shape == points[1].shape, \
                f'Shape mismatch, both set of points should' \
                f'have similar shape, but found {points[0].shape}' \
                f'and {points[1].shape}'

            self.points_r0 = points[0].points
            self.points_r1 = (points[1] - points[0]).points
        else:
            if len(points.shape) == 2:
                points = np.expand_dims(points, axis=0)

            assert points.shape[1] == 2 and points.shape[2] == 3, f'3D lines need two points having' \
                                                                  f'3 coordinate values'

            self.points_r0 = Point3D(points[:, 0, :]).points
            self.points_r1 = Point3D(points[:, 1, :] - points[:, 0, :]).points

        self.lines = np.stack((self.points_r0, self.points_r1), axis=1)
        self.shape = self.lines.shape

class Plane:
    def __init__(self, points, n=None):
        # check if collinear before making plane

        assert isinstance(points, (np.ndarray, list)), f'Points need to be either list of Point3D objects' \
                                                       f'or numpy.ndarray, found type: {type(points)}'
        if n is None:
            if isinstance(points, list):
                assert len(points) == 3, f'If no normal vector is provided, then exactly ' \
                                         f'3 points are needed for each sample line,' \
                                         f'found {len(points)}'
                assert points[0].shape == points[1].shape \
                       and points[0].shape == points[2].shape, f'All set of points should have same ' \
                                                               f'shape, found {points[0].shape}, ' \
                                                               f'{points[1].shape}, and {points[2].shape}'

                self.n = np.cross((points[1]-points[0]).points,
                                  (points[2]-points[0]).points)
                self.point_r = points[0].points

            else:
                if len(points.shape) == 2:
                    points = np.expand_dims(points, axis=0)

                assert len(points.shape) == 3 and points.shape[1] == 3 and points.shape[2] == 3,\
                    f'Each sample should contain 3 3D points' \
                    f'expected shape is (sample x 3 x 3), found' \
                    f'{points.shape}'

                # shape is sample x 3
                self.n = np.cross((points[:,1,:]-points[:,0,:]),
                                  (points[:,2,:]-points[:,0,:]))
                # shape is sample x 3
                self.points_r = points[:,0,:]
        else:
            if len(points.shape) == 2:
                points = np.expand_dims(points, axis=0)

            assert len(points.shape) == 3 and points.shape[1] == 3 and points.shape[2] == 3, \
                f'Each sample should contain 3 3D points' \
                f'expected shape is (sample x 3 x 3), found' \
                f'{points.shape}'
            self.n = n
            self.points_r = points[:,0,:]

        self.plane = np.stack((self.points_r, self.n), axis=1)
        self.shape = self.plane.shape

    def intersection(self, geo_entity):
        if isinstance(geo_entity, Line3D):
            # ax+by+cz=d
            # d = n . point_r
            # replace x, y, z with lines equation
            # write the equation
            d = np.einsum('ij, ij -> i', self.n, self.points_r)
            c, t_sum = np.einsum('ij, klj -> lik', self.n, geo_entity.lines)
            d = np.tile(np.expand_dims(d, axis=1), (1, c.shape[1]))
            t = (d - c) / t_sum
            r0 = np.tile(geo_entity.points_r0, (d.shape[0],1,1))
            r1 = np.einsum('ij, jk -> ijk', t, geo_entity.points_r1)
            return r0 + r1


# t = np.random.rand(4,2,3)
# p1 =
# s = Line3D(t)
# l = 0
# from numpy.random import rand
# from sympy import Line3D as _Line3D
# from sympy import Plane as _Plane
# s = rand(5)
# from time import time as t
# x1 = rand(4,2,3)
# y1 = rand(2,3,3)

# l1 = Line3D(x1)
# pl1 = Plane(y1)
# n = t()
# intersections = pl1.intersection(l1)
# np_t = t() - n
# print(f'np time: {t()-n}')
#
# n = t()
# x1 = rand(400,2,3)
# y1 = rand(200,3,3)
#
# l1 = Line3D(x1)
# pl1 = Plane(y1)
# intersections = pl1.intersection(l1)
# print(f'np time: {(t()-n)/np_t}')
#
#
# for i in range(x1.shape[0]):
#     l = _Line3D(*x1[i].tolist())
#     for j in range(y1.shape[0]):
#         p = _Plane(*y1[j].tolist())
#         ins = p.intersection(l)[0].evalf()
#         # print(f'sympy : {ins}, numpy: {intersections[j,i]}')
# # print(f'sympy time: {t() - n}')
# print(f'ratio is: {(t() - n)/np_t}')
# l = 0




# p0 = Point3D(np.zeros((4,3)))
# p1 = Point3D(np.ones((4,3)))
#
# pl1 = np.array([1,-1,4])
# pl2 = np.array([3,5,-2])
#
# pp1 = np.array([3,1,-10])
# pp2 = np.array([4,4,1])
# pp3 = np.array([3,5,-20])
#
# x1 = rand(4,2,3)
# y1 = rand(2,3,3)
#
# x1[0,0,:] = pl1
# x1[0,1,:] = pl2
#
# y1[0,0,:] = pp1
# y1[0,1,:] = pp2
# y1[0,2,:] = pp3
#
#
# l1 = Line3D(x1)
# p1 = Plane(y1)
#
#
# p = p1.intersection(l1)
# # p = p1+p1
# print(p[0,0])
# l = 0