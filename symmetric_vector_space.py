from scipy.spatial.qhull import QhullError
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from PIL import Image##########################################
number_generations = []
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
    change_threshold = 10**-5  # Set change threshold
    total_generations = 0

    #Trackers
    change_history = []  # Initialize change history
    lowest_max_change = np.inf  # Initialize variable to track lowest max change
    running_sum = 0  # Initialize running sum for average calculation

    while True:
        total_generations+=1
        # Save current x for later comparison
        x_old = x.copy()
        r = 0.05
        #r = np.random.uniform(0,0.1)
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
        # if stable_generations >= 100:
        #     number_generations.append(total_generations)
        #     break
        if total_generations >= 100:
            number_generations.append(total_generations) 
            break

a = 0.03
b = 0.004
d = 0.005
g = b
    
points_convergences = {}

total_iterations = 10000  # Number of tests
for i in range(total_iterations):
    initial_point = np.random.dirichlet(np.ones(4), size=1)[0]
    iterate_generations(initial_point, d, a, b, g) 

    # Store the initial point and its corresponding convergent point
    points_convergences[tuple(initial_point)] = points[-1]

    # Total progress calculation
    progress = (i + 1) / total_iterations * 100
    print(f"Total progress: {progress:.2f}%")

# Convert dict to two separate lists
initial_points = list(points_convergences.keys())
convergent_points = list(points_convergences.values())

# Convert lists to np arrays for easier manipulation
initial_points = np.array([np.dot(vertices.T, point) for point in initial_points])
convergent_points = np.array([np.dot(vertices.T, point) for point in convergent_points])

# Compute displacement vectors from initial points to convergent points
displacement_vectors = convergent_points - initial_points

normalized_displacement = displacement_vectors / np.linalg.norm(displacement_vectors, axis=1, keepdims=True)

# Use magnitudes for color mapping
magnitudes = np.linalg.norm(displacement_vectors, axis=1)
colors = cm.jet(magnitudes / np.max(magnitudes))

fig = plt.figure(figsize=(10, 10))##################################################
ax = fig.add_subplot(111, projection='3d')

# Calculate the barycentric coordinates for these points
barycentric_coords = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
labels = ['AB', 'Ab', 'aB', 'ab']
barycentric_to_cartesian = []

for i, element in enumerate(barycentric_coords):
    barycentric_to_cartesian.append(np.dot(vertices.T, element))

barycentric_to_cartesian = np.array(barycentric_to_cartesian)
#####################################################################################################################################################
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
font_properties = {'weight': 'normal', 'size': 12}

offsets = {
    "AB": (0.03, 0, 0),   # Slightly to the right
    "Ab": (-0.1, 0, 0), # Slightly to the left
    "aB": (-0.07, -0.14, 0), # Slightly down
    "ab": (0, 0.1, 0)   # Slightly up
}

for i in range(len(barycentric_to_cartesian)):
    x, y, z = barycentric_to_cartesian[i]
    offset = offsets[labels[i]]
    ax.scatter(x, y, z, color='black')
    ax.text(x + offset[0], y + offset[1], z + offset[2], labels[i], color='black', **font_properties)
#####################################################################################################################################################

sample_rate = 10  # for example, taking every 10th point
initial_points_subsample = initial_points[::sample_rate]
displacement_vectors_subsample = displacement_vectors[::sample_rate]
colors_subsample = colors[::sample_rate]  # if you have colors array
max_length = 0.1
norms = np.linalg.norm(displacement_vectors_subsample, axis=1)
normalized_displacement = np.where(norms[:, np.newaxis] != 0, 
                                  displacement_vectors_subsample * (max_length / norms[:, np.newaxis]),
                                  displacement_vectors_subsample)
ax.quiver(initial_points_subsample[:, 0], initial_points_subsample[:, 1], initial_points_subsample[:, 2],
          normalized_displacement[:, 0], normalized_displacement[:, 1], normalized_displacement[:, 2],
          color=cm.jet(np.linalg.norm(colors_subsample, axis=1)))


# Initial and convergent points
ax.scatter(initial_points[:, 0], initial_points[:, 1], initial_points[:, 2], c='blue', marker='o', label='Initial Points')
ax.scatter(convergent_points[:, 0], convergent_points[:, 1], convergent_points[:, 2], c='red', marker='x', label='Convergent Points')


# Remove tick labels to declutter:
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.grid(color="white", linestyle='solid')
plt.tight_layout()
# Save the image to your desktop with high DPI for better quality
filename = "/Users/noah/Desktop/your_figure_name.png"
plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)

# # Close the plt figure to release memory
# plt.close()

# # Crop the image to remove any whitespace (using PIL)
# img = Image.open(filename)
# cropped_img = img.crop(img.getbbox())
# cropped_img.save(filename)
plt.show()