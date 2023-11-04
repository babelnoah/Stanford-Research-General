import multiprocessing
from functools import partial
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

convergent_points = []

# Define coordinates of the tetrahedron vertices
a_vertex = np.array([np.sqrt(8/9), 0, -1/3])
b_vertex = np.array([-np.sqrt(2/9), np.sqrt(2/3), -1/3])
c_vertex = np.array([-np.sqrt(2/9), -np.sqrt(2/3), -1/3])
d_vertex = np.array([0,0,1])

vertices = np.array([a_vertex, b_vertex, c_vertex, d_vertex])

def is_point_in_tetrahedron(point, a_vertex, b_vertex, c_vertex, d_vertex, buffer=5):
    vertices = [a_vertex, b_vertex, c_vertex, d_vertex]
    max_dist = max(np.linalg.norm(v1-v2) for v1 in vertices for v2 in vertices)
    
    for vertex in vertices:
        if np.linalg.norm(point - vertex) > max_dist + buffer:
            return False
    return True

def calculate_next_generation(x, r, delta, alpha, beta, gamma):
    D = x[0]*x[3] - x[1]*x[2] 
    w_bar = 1 - delta*(x[0]**2 + x[3]**2) - alpha*(x[1]**2 + x[2]**2) - 2*beta*(x[2]*x[3] + x[0]*x[1]) - 2*gamma*(x[0]*x[2] + x[1]*x[3])
    x_next = np.zeros(4)
    x_next[0] = (x[0] - delta*x[0]**2 - beta*x[0]*x[1] - gamma*x[0]*x[2] - r*D)/w_bar
    x_next[1] = (x[1] - beta*x[0]*x[1] - alpha*x[1]**2 - gamma*x[1]*x[3] + r*D)/w_bar
    x_next[2] = (x[2] - gamma*x[0]*x[2] - alpha*x[2]**2 - beta*x[2]*x[3] + r*D)/w_bar
    x_next[3] = (x[3] - gamma*x[1]*x[3] - beta*x[2]*x[3] - delta*x[3]**2 - r*D)/ w_bar

    # Normalize x_next so that it sums to 1
    x_next /= np.sum(x_next)

    return x_next

def iterate_generations(x, delta, alpha, beta, gamma):
    stable_generations = 0  # Initialize counter for stable generations
    change_threshold = 2.5*10**-4  # Set change threshold

    points = []
    while True:
        # Save current x for later comparison
        x_old = x.copy()
        #r = 0.05
        r = np.random.uniform(0,0.08)
        x = calculate_next_generation(x, r, delta, alpha, beta, gamma)

        # Check if absolute change in all components of x is less than threshold
        if np.all(np.abs(x - x_old) < change_threshold):
            stable_generations += 1
        else:
            # Reset counter if change is above threshold
            stable_generations = 0

        points.append(x.tolist())

        # Break loop if x has been stable for 100 generations
        if stable_generations >= 100:
            break

    return points

def run_iteration(r_tuple, total_iterations, d, a, b, g, r_values):
    r_idx, r_value = r_tuple
    local_convergences = {}
    for i in range(total_iterations):
        r = np.random.uniform(r_value, r_values[r_idx + 1])
        initial_point = np.random.dirichlet(np.ones(4), size=1)[0]
        points = iterate_generations(initial_point, d, a, b, g) 

        # Store the initial point and its corresponding convergent point
        local_convergences[tuple(initial_point)] = points[-1]

    # Total progress calculation
    progress = ((r_idx * total_iterations + i + 1) / (total_iterations * len(r_values))) * 100
    print("Total progress: {:.2f}%".format(progress))
    return local_convergences

a = 0.02
b = 0.04
d = 0.03
g = b

total_iterations = 10000  # Number of tests
points_convergences = {}
# Define r_value ranges
r_values = np.linspace(0, 0.5, 8)

if __name__ == '__main__':  # Protects your forked worker processes from recursively importing your main module.
    pool = multiprocessing.Pool(processes=4)  # Example using 4 processes

    points_convergences = {}

    # partial function to fix the common parameters across all r_values
    run_iteration_partial = partial(run_iteration, total_iterations=total_iterations, d=d, a=a, b=b, g=g, r_values=r_values)

    # map the function over the list of r_values and collect results
    results = pool.map(run_iteration_partial, enumerate(r_values[:-1]))


    for res in results:
        points_convergences.update(res)

# Convert dict to two separate lists
initial_points = list(points_convergences.keys())
convergent_points = list(points_convergences.values())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract convergent points for plotting
convergent_points = np.array(convergent_points)

# Convert to Cartesian coordinates
cartesian_convergent_points = []
for i, element in enumerate(convergent_points):
    cartesian_convergent_points.append(np.dot(vertices.T, element))
cartesian_convergent_points = np.array(cartesian_convergent_points)

x_conv = cartesian_convergent_points[:, 0]
y_conv = cartesian_convergent_points[:, 1]
z_conv = cartesian_convergent_points[:, 2]

mesh_points = np.column_stack((x_conv, y_conv, z_conv))

# Apply DBSCAN to identify clusters
db = DBSCAN(eps=0.3, min_samples=10).fit(mesh_points)
labels = db.labels_

# Identify the number of clusters
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# Initialize a list to store the points for each cluster
clusters = [mesh_points[labels == i] for i in range(n_clusters)]

# Initialize a dict to store the initial points for each cluster
initial_clusters = {i: [] for i in range(n_clusters)}

# Build the mapping of initial points to their cluster
for initial_point, convergent_point in zip(initial_points, convergent_points):
    # Transform the convergent point to Cartesian coordinates
    cartesian_convergent_point = np.dot(vertices.T, convergent_point)
    # Find the label of the convergent_point
    label = db.labels_[np.where((mesh_points == cartesian_convergent_point).all(axis=1))[0][0]]
    if label != -1:  # Exclude noise points
        # Append the initial_point to the corresponding cluster
        initial_clusters[label].append(initial_point)

# Convert lists to np arrays for easier manipulation and transform to Cartesian coordinates
for label in initial_clusters.keys():
    initial_clusters[label] = np.array([np.dot(vertices.T, point) for point in initial_clusters[label]])

# Get colormap
cmap = cm.get_cmap("Spectral", len(r_values))

# Plot the points for each cluster
for i, cluster in enumerate(clusters):
    # Compute the Convex Hull for the current cluster
    hull = ConvexHull(cluster)
    
    # Plot the Convex Hull for the cluster
    for s in hull.simplices:
        ax.plot_trisurf(cluster[s, 0], cluster[s, 1], cluster[s, 2], alpha=0.8, color=cmap(i))

    # Plot the initial points for the cluster with more opacity
    initial_cluster = initial_clusters[i]
    
    # Compute the Convex Hull for the initial points of the current cluster
    hull_initial = ConvexHull(initial_cluster)

    # Plot the Convex Hull for the initial points
    for s in hull_initial.simplices:
        ax.plot_trisurf(initial_cluster[s, 0], initial_cluster[s, 1], initial_cluster[s, 2], alpha=0.25, color=cmap(i))

# Plot the outline of the tetrahedron
vertices = np.array([a_vertex, b_vertex, c_vertex, d_vertex])
ax.plot_trisurf(*vertices.T, color='r', alpha=0.1)

ax.set_title("Partitioned Tetrahedral")
plt.savefig('output_figure.png')
plt.show()