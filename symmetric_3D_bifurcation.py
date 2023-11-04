from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

convergent_points = []
convex_hull_volumes = {}

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
    #D_next = x_next[0]*x_next[3] - x_next[1]*x_next[2]

    # Normalize x_next so that it sums to 1
    x_next /= np.sum(x_next)

    return x_next

def iterate_generations(x, delta, alpha, beta, gamma):
    stable_generations = 0  # Initialize counter for stable generations
    change_threshold = 10**-2  # Set change threshold

    #Trackers
    change_history = []  # Initialize change history
    lowest_max_change = np.inf  # Initialize variable to track lowest max change
    running_sum = 0  # Initialize running sum for average calculation

    while True:
        # Save current x for later comparison
        x_old = x.copy()
        #r = 0.05
        r = np.random.uniform(0,0.08)
        x = calculate_next_generation(x, r, delta, alpha, beta, gamma)

        # # Trackers
        # change = np.abs(x - x_old)
        # change_history.append(change)
        # if len(change_history) > 100:
        #     change_history.pop(0)
        # max_change_last_100_gens = max(np.max(ch) for ch in change_history)
        # lowest_max_change = min(lowest_max_change, max_change_last_100_gens)
        # running_sum += np.mean(change)
        # average_change = running_sum / len(points)
        # print(f'\rMax change across last 100 generations: {max_change_last_100_gens:.2e}, Lowest max change: {lowest_max_change:.2e}, Running average change: {average_change:.2e}', end='')

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

a = 0.1
b = 0.4
d = 0.1
g = 0.4


points = []
total_iterations = 100  # Number of tests
points_convergences = {}
# Define r_value ranges
r_values = np.linspace(0, 0.5, 5)
# Declare mapping from initial points to r-values
initial_point_r_values = {}

# Iterate over each r_value range
for r_idx, r_value in enumerate(r_values[:-1]):
    for i in range(total_iterations):
        r = np.random.uniform(r_value, r_values[r_idx + 1])
        print(r_values[r_idx + 1])
        initial_point = np.random.dirichlet(np.ones(4), size=1)[0]
        iterate_generations(initial_point, d, a, b, g) 

        # Store the initial point, its corresponding convergent point and r-value
        points_convergences[tuple(initial_point)] = points[-1]
        initial_point_r_values[tuple(initial_point)] = (r_value, r_values[r_idx + 1])  # Store r-value range


        # Total progress calculation
        progress = ((r_idx * total_iterations + i + 1) / (total_iterations * len(r_values))) * 100
        print("Total progress: {:.2f}%".format(progress))

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

# Initialize a dict to store the r-values for each cluster
r_value_clusters = {i: [] for i in range(n_clusters)}

# Build the mapping of initial points to their cluster and store r-values
for initial_point, convergent_point in zip(initial_points, convergent_points):
    # Transform the convergent point to Cartesian coordinates
    cartesian_convergent_point = np.dot(vertices.T, convergent_point)
    # Find the label of the convergent_point
    label = db.labels_[np.where((mesh_points == cartesian_convergent_point).all(axis=1))[0][0]]
    if label != -1:  # Exclude noise points
        # Append the initial_point to the corresponding cluster and store r-value
        initial_clusters[label].append(initial_point)
        r_value_clusters[label].append(initial_point_r_values[initial_point])  # Store r-value ranges

# Compute mean r-values for each cluster
mean_r_value_ranges = {label: np.mean(np.array(r_values), axis=0) for label, r_values in r_value_clusters.items()}

# Normalize mean r-values to range [0, 1] for use with colormap
min_r_value_range, max_r_value_range = np.min([range[0] for range in mean_r_value_ranges.values()]), np.max([range[1] for range in mean_r_value_ranges.values()])
normalized_mean_r_value_ranges = {label: [(value[0] - min_r_value_range) / (max_r_value_range - min_r_value_range), 
                                          (value[1] - min_r_value_range) / (max_r_value_range - min_r_value_range)] 
                                  for label, value in mean_r_value_ranges.items()}

# Convert lists to np arrays for easier manipulation and transform to Cartesian coordinates
for label in initial_clusters.keys():
    initial_clusters[label] = np.array([np.dot(vertices.T, point) for point in initial_clusters[label]])

# Get colormap
cmap = cm.get_cmap("Spectral")

# Initialize a dictionary to store volumes by r-value ranges for each cluster
volumes_by_cluster = {i: [] for i in range(n_clusters)}

fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)


# Plot the points for each cluster
for i, cluster in enumerate(clusters):
    # Compute the Convex Hull for the current cluster
    hull = ConvexHull(cluster)
    # Store the volume of the convex hull
    convex_hull_volumes[i] = hull.volume

    # Plot the Convex Hull for the cluster using normalized mean r-value range as color
    color = cmap(np.mean(normalized_mean_r_value_ranges[i]))
    for s in hull.simplices:
        ax1.plot_trisurf(cluster[s, 0], cluster[s, 1], cluster[s, 2], alpha=0.8, color=color)

    # Plot the initial points for the cluster with more opacity
    initial_cluster = initial_clusters[i]

    # Compute the Convex Hull for the initial points of the current cluster
    hull_initial = ConvexHull(initial_cluster)

    # Plot the Convex Hull for the initial points using normalized mean r-value range as color
    for s in hull_initial.simplices:
        ax1.plot_trisurf(initial_cluster[s, 0], initial_cluster[s, 1], initial_cluster[s, 2], alpha=0.25, color=color)

    # Now that the volume for cluster i has been computed, append it to volumes_by_cluster
    volumes_by_cluster[i].append(convex_hull_volumes[i])

# Map each cluster to its r-value range and store the corresponding volume
for label, value in mean_r_value_ranges.items():
    volumes_by_cluster[label].append(convex_hull_volumes[label])

# Plot the outline of the tetrahedron
vertices = np.array([a_vertex, b_vertex, c_vertex, d_vertex])
ax1.plot_trisurf(*vertices.T, color='r', alpha=0.1)

ax1.set_title("Partitioned Tetrahedral")

# Plot the volumes by r-value range for each cluster
for label, volumes in volumes_by_cluster.items():
    color = cmap(np.mean(normalized_mean_r_value_ranges[label]))  # Use the same color mapping as in the other plot
    ax2.plot(mean_r_value_ranges[label], volumes, color=color)

ax2.set_xlabel('Mean r-value range')
ax2.set_ylabel('Volume of convex hull')
ax2.set_title('Volume by r-value range')

plt.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax2, label='Mean r-value')

plt.savefig('output_figure.png')
plt.show()
