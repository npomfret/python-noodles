import numpy as np
from fracdiff import fdiff

a = np.array([1, 2, 4, 7, 0])
print(fdiff(a, 0.5))