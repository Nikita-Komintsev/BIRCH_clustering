import random
def generate_random_pairs(num_pairs, filename):
    x1 = ["{:.2f}".format(random.uniform(0, 1)) for _ in range(num_pairs)]
    x2 = ["{:.2f}".format(random.uniform(0, 1)) for _ in range(num_pairs)]

    with open(filename, 'w') as file:
        file.write("x1 = [\n" + "\t" + ",\n\t".join(x1) + "\n" + "]\n\n")
        file.write("x2 = [\n" + "\t" + ",\n\t".join(x2) + "\n" + "]\n")

generate_random_pairs(200, 'dataset.py')

from dataset import x1, x2
import numpy as np
from sklearn.cluster import Birch
import time

print("Compute birch clustering...")
st = time.time()

X = np.stack([x1, x2], axis=1)
X = np.reshape(X, (-1, 2))

n_clusters = 3
birch = Birch(n_clusters=n_clusters, threshold=0.02, branching_factor=10)
birch.fit(X)

# label = birch.labels_
label = birch.predict(X)

print("Elapsed time: ", time.time() - st)
print("Number of clusters: ", np.unique(label).size)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.scatter(x1, x2, c=label)

ax.set_xlabel(r"$x1$", fontsize=15)
ax.set_ylabel(r"$x2$", fontsize=15)
ax.set_title(f"Birch clustering $K = {n_clusters} Scatter plot")

ax.grid(True)

plt.show()