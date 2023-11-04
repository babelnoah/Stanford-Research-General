from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

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
    #D_next = x_next[0]*x_next[3] - x_next[1]*x_next[2]

    # Normalize x_next so that it sums to 1
    x_next /= np.sum(x_next)

    return x_next

def iterate_generations(x, delta, alpha, beta, gamma, r_range):
    stable_generations = 0  # Initialize counter for stable generations
    change_threshold = 2*10**-3  # Set change threshold

    while True:
        #print(change_threshold)
        change_threshold += 10**-4
        # Save current x for later comparison
        x_old = x.copy()

        # The r value for each generation is chosen randomly from the given range
        r = np.random.uniform(r_range[0], r_range[1])
        x = calculate_next_generation(x, r, delta, alpha, beta, gamma)

        # Check if absolute change in all components of x is less than threshold
        if np.all(np.abs(x - x_old) < change_threshold):
            stable_generations += 1
        else:
            # Reset counter if change is above threshold
            stable_generations = 0

        # Break loop if x has been stable for 100 generations
        if stable_generations >= 100:
            break

    return x

a = 0.03
b = 0.004
d = 0.005
g = b

points = []
points_convergences = {}

# Create a list to store the r ranges
total_points = 500
r_ranges = np.linspace(0.01, 0.5, total_points)

# Create a dictionary to store the converged points for each r range
r_convergences = {r_range: [] for r_range in r_ranges}

total_iterations = 1000  # Number of tests for each r range
for r_range in r_ranges:
    for _ in range(total_iterations):
        initial_point = np.random.dirichlet(np.ones(4), size=1)[0]
        converged_point = iterate_generations(initial_point, d, a, b, g, [0 , r_range])
        r_convergences[r_range].append(converged_point)

    # Total progress calculation
    progress = (np.where(r_ranges == r_range)[0][0] + 1) / len(r_ranges) * 100
    print(f"Total progress: {progress:.2f}%")

# Convert dict to two separate lists
r_values = list(r_convergences.keys())
convergent_points = list(r_convergences.values())

# Create subplot
fig, axs = plt.subplots(4, figsize=(10, 20))

# List to store x labels
x_labels = ['x1', 'x2', 'x3', 'x4']

# Plot for each x
for i in range(4):
    for r_range, converged_points in r_convergences.items():
        converged_x_values = [point[i] for point in converged_points]
        axs[i].scatter([r_range]*len(converged_x_values), converged_x_values, alpha=0.1,color='k')
    axs[i].set_xlabel('r range')
    axs[i].set_ylabel(f'Converged {x_labels[i]} values')
    axs[i].set_title(f'Converged {x_labels[i]} values vs r range')

plt.tight_layout()
plt.show()


