from scipy.spatial.qhull import QhullError
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans

convergent_points = []
points = []
points_convergences = {}
number_generations = []

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
    total_generations = 0
    stable_generations = 0  # Initialize counter for stable generations
    change_threshold = 10**-20  # Set change threshold

    #Trackers
    change_history = []  # Initialize change history
    lowest_max_change = np.inf  # Initialize variable to track lowest max change
    running_sum = 0  # Initialize running sum for average calculation

    while True:
        total_generations +=1
        # Save current x for later comparison
        x_old = x.copy()
        #r = 0.05
        r = np.random.uniform(0,0.5)
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
            number_generations.append(total_generations)
            break
    
total_iterations = 10000  # Number of tests
for i in range(total_iterations):
    matrix = create_matrix()
    initial_point = np.random.dirichlet(np.ones(4), size=1)[0]
    iterate_generations(initial_point, matrix) 

    # Store the initial point and its corresponding convergent point
    points_convergences[tuple(initial_point)] = points[-1]

    # Total progress calculation
    progress = (i + 1) / total_iterations * 100
    print(f"Total progress: {progress:.2f}%")

# Convert dict to two separate lists
initial_points = list(points_convergences.keys())
convergent_points = list(points_convergences.values())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Convert lists to np arrays for easier manipulation
initial_points = np.array([np.dot(vertices.T, point) for point in initial_points])
convergent_points = np.array([np.dot(vertices.T, point) for point in convergent_points])

# Compute displacement vectors from initial points to convergent points
displacement_vectors = convergent_points - initial_points

# Normalize displacement vectors for color mapping
colors = displacement_vectors / np.max(np.linalg.norm(displacement_vectors, axis=1))

# Scale displacement vectors by their lengths
lengths = np.linalg.norm(displacement_vectors, axis=1)
scaled_displacement_vectors = displacement_vectors / np.linalg.norm(displacement_vectors, axis=1, keepdims=True) * lengths[:, np.newaxis]

# Subsample the data for clearer visualization
sample_size = 500  # change this as desired
idx = np.random.choice(range(len(initial_points)), sample_size, replace=False)

initial_points_subsample = initial_points[idx]
displacement_vectors_subsample = scaled_displacement_vectors[idx]
colors_subsample = colors[idx]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting displacement vectors with consistent arrow size
arrow_length = 0.1
ax.quiver(initial_points_subsample[:, 0], initial_points_subsample[:, 1], initial_points_subsample[:, 2],
          displacement_vectors_subsample[:, 0], displacement_vectors_subsample[:, 1], displacement_vectors_subsample[:, 2],
          length=arrow_length, color=cm.jet(np.linalg.norm(colors_subsample, axis=1)))

# Optionally, plot the start and end points
ax.scatter(initial_points_subsample[:, 0], initial_points_subsample[:, 1], initial_points_subsample[:, 2], c='blue', marker='o', label='Initial Points')
ax.scatter(convergent_points[idx, 0], convergent_points[idx, 1], convergent_points[idx, 2], c='red', marker='x', label='Convergent Points')
print(np.average(number_generations))


# Plot the outline of the tetrahedron
vertices = np.array([a_vertex, b_vertex, c_vertex, d_vertex])
ax.plot_trisurf(*vertices.T, color='r', alpha=0.1)

# Adding edges for the tetrahedron
edges = [
    (a_vertex, b_vertex),
    (a_vertex, c_vertex),
    (a_vertex, d_vertex),
    (b_vertex, c_vertex),
    (b_vertex, d_vertex),
    (c_vertex, d_vertex)
]

for start, end in edges:
    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='black')

# Calculate the barycentric coordinates for these points
barycentric_coords = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
labels = ['AB', 'Ab', 'aB', 'ab']
barycentric_to_cartesian = []

for i, element in enumerate(barycentric_coords):
    barycentric_to_cartesian.append(np.dot(vertices.T, element))

barycentric_to_cartesian = np.array(barycentric_to_cartesian)

for i in range(len(barycentric_to_cartesian)):
    x, y, z = barycentric_to_cartesian[i]
    ax.scatter(x, y, z, color='black')
    ax.text(x, y, z, labels[i], color='black')

ax.legend()
ax.set_title("Vector Field of Displacements")
plt.show()
