from scipy.spatial.qhull import QhullError
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import os

# Create a directory to save the images, if it doesn't exist
if not os.path.exists('tetrahedral_images'):
    os.makedirs('tetrahedral_images')

convergent_points = []
points = []
points_convergences = {}

a = 0.02
b = 0.04
d = 0.03
g = b

# Define coordinates of the tetrahedron vertices
a_vertex = np.array([np.sqrt(8/9), 0, -1/3])
b_vertex = np.array([-np.sqrt(2/9), np.sqrt(2/3), -1/3])
c_vertex = np.array([-np.sqrt(2/9), -np.sqrt(2/3), -1/3])
d_vertex = np.array([0,0,1])

vertices = np.array([a_vertex, b_vertex, c_vertex, d_vertex])

def create_matrix():
    W = np.zeros((4, 4))

    # Fill in the matrix based on the conditions given
    W[0][0] = np.random.rand() #w11
    W[1][1] = np.random.rand() #w22
    W[0][1] = W[1][0] = np.random.rand() # w12 = w21
    W[0][2] = W[2][0] = np.random.rand() # w13 = w31
    W[1][2] = W[2][1] = W[3][0] = W[0][3] = np.random.rand() # w23 = w32 = w41 = w14
    W[1][3] = W[3][1] = np.random.rand() # w24 = w42
    W[2][3] = W[3][2] = np.random.rand() # w34 = w43
    W[2][2] = np.random.rand() #w33
    W[3][3] = np.random.rand() #w44

    
    
    return W

def calculate_next_generation(x, r, W):

    # Compute W_i (average fitness values)
    W_avg = np.zeros(4)  # Initialize an array with zeros
    for i in range(4):
        for j in range(4):
            W_avg[i] += W[i][j] * x[j]
    
    # w_bar calculation
    w_bar = sum([W_avg[i] * x[i] for i in range(4)])

    # D (Differential of x)
    D = x[0]*x[3] - x[1]*x[2]

    # Next Generation calculations
    x_next = np.zeros(4)
    x_next[0] = (x[0] * W_avg[0] - W[0][3] * r * D) / w_bar
    x_next[1] = (x[1] * W_avg[1] + W[0][3] * r * D) / w_bar
    x_next[2] = (x[2] * W_avg[2] + W[0][3] * r * D) / w_bar
    x_next[3] = (x[3] * W_avg[3] - W[0][3] * r * D) / w_bar

    return x_next

def iterate_generations(x, matrix):
    stable_generations = 0  # Initialize counter for stable generations
    change_threshold = 10**-10  # Set change threshold

    #Trackers
    change_history = []  # Initialize change history
    lowest_max_change = np.inf  # Initialize variable to track lowest max change
    running_sum = 0  # Initialize running sum for average calculation

    while True:
        # Save current x for later comparison
        x_old = x.copy()
        r = 0.05
        #r = np.random.uniform(0,0.5)
        x = calculate_next_generation(x, r, matrix)

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

for r in np.arange(0, 0.5, 0.01):
    convergent_points = []
    points = []
    points_convergences = {}
    print(str(200*r) + "% Complete")
    total_iterations = 10000  # Number of tests
    for i in range(total_iterations):
        matrix = create_matrix()
        initial_point = np.random.dirichlet(np.ones(4), size=1)[0]
        iterate_generations(initial_point, matrix) 

        # Store the initial point and its corresponding convergent point
        points_convergences[tuple(initial_point)] = points[-1]

        # # Total progress calculation
        # progress = (i + 1) / total_iterations * 100
        # print(f"Total progress: {progress:.2f}%")

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

    # Make sure it's an array
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
        
    # Plot the points for each cluster
    for i, cluster in enumerate(clusters):
        
        unique_cluster_points = np.unique(cluster, axis=0)
        
        if len(unique_cluster_points) == 1:
            # If all points are the same, plot a single point.
            ax.scatter(*unique_cluster_points[0], color=f'C{i}', s=100)
        elif len(unique_cluster_points) == 2:
            # If there are two unique points, plot a line between them.
            ax.plot(*zip(*unique_cluster_points), color=f'C{i}', linewidth=3)
        elif len(unique_cluster_points) > 2:
            try:
                # Try to plot the Convex Hull.
                hull = ConvexHull(cluster)
                for s in hull.simplices:
                    ax.plot_trisurf(cluster[s, 0], cluster[s, 1], cluster[s, 2], alpha=1, color=f'C{i}')
            except QhullError:
                # If the ConvexHull computation fails, plot the points.
                ax.scatter(*unique_cluster_points.T, color=f'C{i}', s=10)

        # Plot the initial points for the cluster with more opacity
        initial_cluster = initial_clusters[i]
        
        # Compute the Convex Hull for the initial points of the current cluster
        if len(np.unique(initial_cluster, axis=0)) > 3:
            try:
                hull_initial = ConvexHull(initial_cluster)

                # Plot the Convex Hull for the initial points
                for s in hull_initial.simplices:
                    ax.plot_trisurf(initial_cluster[s, 0], initial_cluster[s, 1], initial_cluster[s, 2], alpha=0.20, color=f'C{i}')
            except QhullError:
                # If the ConvexHull computation fails, plot the points.
                ax.scatter(*np.unique(initial_cluster, axis=0).T, color=f'C{i}', s=10, alpha=0.5)

    # Plot the outline of the tetrahedron
    vertices = np.array([a_vertex, b_vertex, c_vertex, d_vertex])
    ax.plot_trisurf(*vertices.T, color='r', alpha=0.1)

    ax.set_title("Partitioned Tetrahedral")
    plt.savefig(f'tetrahedral_images/tetrahedral_{int(r*100)}.png')
    plt.close()  # Close the plot to free up memory