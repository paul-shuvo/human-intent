import numpy as np 

# radius of the circle
circle_r = 20
# center of the circle (x, y)
box = np.random.random((4,2)) * 50
center = (box[0] + box[3]) / 2
# center = np.array([20,10])

# random angle
alpha = 2 * np.pi * np.random.random((20,1))
# random radius
r = circle_r * np.sqrt(np.random.random())
# calculating coordinates
x = r * np.cos(alpha) + center[0]
y = r * np.sin(alpha) + center[1]
x[0] = center[0]
y[0] = center[1]
s = np.hstack((x, y), dtype=np.int16).astype(np.int32).tolist()
l = 1