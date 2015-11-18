import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

def init_tracker():
    tracker = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1.0   # time step

    tracker.F = np.array([[1, dt, 0,  0],
                          [0,  1, 0,  0],
                          [0,  0, 1, dt],
                          [0,  0, 0,  1]])

    tracker.H = np.array([[1, 0, 0, 0],
                          [0, 0, 1, 0]])

    tracker.R = np.eye(2) * 1000
    q = Q_discrete_white_noise(dim=2, dt=dt, var=1)
    tracker.Q = block_diag(q, q)
    tracker.x = np.array([[2000, 0, 700, 0]]).T
    tracker.P = np.eye(4) * 50.
    return tracker


my_filter = init_tracker()
predictions = np.load("predictions.npy")

xs_filtered = []
ys_filtered = []

for i, pred in enumerate(predictions):
    my_filter.update(np.array([pred]).T)

    # do something with the output
    x = my_filter.x
    print(x[2][0])
    xs_filtered.append(x[0][0])
    ys_filtered.append(x[2][0])
    my_filter.predict()
    

#print(xs_filtered)
#print(predictions[:, 1])

#print(len(xs_filtered))

plt.plot(xs_filtered, ys_filtered)

filtered_preds = []

for i, (x, y) in enumerate(zip(xs_filtered, ys_filtered)):
    filtered_preds.append([x, y])

print(ys_filtered)
arr = np.array([[xs_filtered], [ys_filtered]])
print(np.array(filtered_preds))
print(predictions)
plt.plot(predictions[:,0], predictions[:,1])
plt.show()

np.save("filtered", filtered_preds)
