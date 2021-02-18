import pytest
from g3d import *
def test_point3d_init():
    coordinates = np.random.rand(2,3).tolist()
    with pytest.raises(AssertionError):
        Point3D(coordinates)

    coordinates = np.random.rand(2)
    with pytest.raises(AssertionError):
        Point3D(coordinates)

    coordinates = np.random.rand(2,2)
    with pytest.raises(AssertionError):
        Point3D(coordinates)

    coordinates = np.random.rand(2,2,3)
    with pytest.raises(AssertionError):
        Point3D(coordinates)

    coordinates = np.random.rand(3)
    p1 = Point3D(coordinates)
    assert p1.points.shape == (1,3)

    coordinates = np.random.rand(5,3)
    p2 = Point3D(coordinates)
    assert p2.points.shape == (5, 3)

def test_line3d_init():
    with pytest.raises(AssertionError):
        Line3D({1,2,3})

    coordinates0 = np.random.rand(2,3)
    coordinates1 = np.random.rand(3,3)
    p1 = Point3D(coordinates0)
    p2 = Point3D(coordinates1)

    with pytest.raises(AssertionError):
        Line3D([p1])

    with pytest.raises(AssertionError):
        Line3D([p1, p2, p1])

    with pytest.raises(AssertionError):
        Line3D([p1,p2])

    coordinates1 = np.random.rand(2,3)
    p2 = Point3D(coordinates1)
    l1 = Line3D([p1, p2])
    assert l1.lines.shape == (p1.points.shape[0], 2, p1.points.shape[1])
    assert np.allclose(l1.lines, np.stack((p1.points, (p2-p1).points), axis=1))

    coordinates = np.random.rand(4,2,3)
    l1 = Line3D(coordinates)
    assert l1.lines.shape == coordinates.shape
    assert np.allclose(l1.lines, np.stack((
        coordinates[:,0,:], coordinates[:,1,:]-coordinates[:,0,:]), axis=1))

    coordinates = np.random.rand(2,2)
    with pytest.raises(AssertionError):
        Line3D(coordinates)

    coordinates = np.random.rand(2, 3)
    l1 = Line3D(coordinates)
    assert l1.shape == (1,2,3)

def test_plane_init():
    with pytest.raises(AssertionError):
        Plane({1,2,3})

    with pytest.raises(AssertionError):
        Plane([])

    p1 = Point3D(np.random.rand(4,3))
    p2 = Point3D(np.random.rand(4, 3))
    p3 = Point3D(np.random.rand(3, 3))

    with pytest.raises(AssertionError):
        Plane([p1,p2,p3])