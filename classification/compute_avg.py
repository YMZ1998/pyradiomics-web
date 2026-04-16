import numpy as np

l = [11 / 12,
     11 / 11,
     10 / 11,
     11 / 11,
     8 / 11,
     10 / 11,
     8 / 10,
     8 / 10,
     9 / 10,
     8 / 10]
print(np.mean(l), np.std(l))
print("{:.2f}±{:.2f}".format(np.mean(l) * 100, np.std(l) * 100))
