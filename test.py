import numpy as np

x_coors = np.arange(1, 100)
y_coors = [0]*99
print(np.polyfit(y_coors, x_coors, 1))