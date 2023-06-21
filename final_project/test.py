import numpy as np


f = np.load('./results/BCQ_Hopper-v3_0.npy')

# smooth the curve
def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            # print(previous)
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

# plot the results
import matplotlib.pyplot as plt
# plot the smoothed curve and original curve in the same figure
plt.plot(smooth_curve(f), label='smoothed curve')
# plot the original curve and set the alpha value
plt.plot(f, alpha=0.3, label='original curve')
plt.legend()
plt.show()