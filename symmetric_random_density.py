import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from PIL import Image

convergent_points = []

# Define coordinates of the tetrahedron vertices
a_vertex = np.array([np.sqrt(8/9), 0, -1/3])
b_vertex = np.array([-np.sqrt(2/9), np.sqrt(2/3), -1/3])
c_vertex = np.array([-np.sqrt(2/9), -np.sqrt(2/3), -1/3])
d_vertex = np.array([0,0,1])

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
    change_threshold = 10**-4  # Set change threshold

    #Trackers
    change_history = []  # Initialize change history
    lowest_max_change = np.inf  # Initialize variable to track lowest max change
    running_sum = 0  # Initialize running sum for average calculation

    while True:
        # Save current x for later comparison
        x_old = x.copy()
        #print(x_old)
        r = 0.04
        #r = np.random.uniform(0,0.04)
        x = calculate_next_generation(x, r, delta, alpha, beta, gamma)

        # Trackers
        change = np.abs(x - x_old)
        change_history.append(change)
        if len(change_history) > 100:
            change_history.pop(0)
        max_change_last_100_gens = max(np.max(ch) for ch in change_history)
        lowest_max_change = min(lowest_max_change, max_change_last_100_gens)
        running_sum += np.mean(change)
        average_change = running_sum / len(points)
        #print(f'\rMax change across last 100 generations: {max_change_last_100_gens:.2e}, Lowest max change: {lowest_max_change:.2e}, Running average change: {average_change:.2e}', end='')

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

a = 0.02
b = 0.04
d = 0.03
g = b

# Define the starting points
starting_points = [
    [0.248, 0.251, 0.249, 0.252],
    [0.497, 0.001, 0.002, 0.500],
    [0.001, 0.497, 0.500, 0.002]
]

points = []

# Iterate over each starting point
# for i, start_point in enumerate(starting_points):
#     print(f"Starting point: {start_point}")
#     iterate_generations(np.array(start_point),d,a,b,g)  

#     # Print the final convergent point
#     print(f"Convergent point: {points[-1]}")
#     print("Sum of convergent point: " + str(np.sum(points[-1])))
#     convergent_points.append(points[-1])

#     # Total progress calculation
#     progress = (i + 1) / len(starting_points) * 100
#     print(f"Total progress: {progress:.2f}%")

total_iterations = 150  # Number of tests
for i in range(total_iterations):
    print(i)
    iterate_generations(np.random.dirichlet(np.ones(4), size=1)[0],d,a,b,g) 

    # Print the final convergent point
    print(f"Convergent point: {points[-1]}")
    print("Sum of convergent point: " + str(np.sum(points[-1])))
    convergent_points.append(points[-1])

    # Total progress calculation
    progress = (i + 1) / total_iterations * 100
    print(f"Total progress: {progress:.2f}%")


# Convert points list to numpy array for easier manipulation
vertices = np.array([a_vertex, b_vertex, c_vertex, d_vertex])
points = np.array(points)

cartesian_points = []  # Define the cartesian_points list
total_points = len(points)

for i, element in enumerate(points):
    cartesian_points.append(np.dot(vertices.T, element))
    progress = (i + 1) / total_points * 100
    

filtered_points = []

for point in cartesian_points:
    if is_point_in_tetrahedron(point, a_vertex, b_vertex, c_vertex, d_vertex):
        filtered_points.append(point)

points = np.array(filtered_points)

# Extract coordinates for plotting
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

print("Density")
# Convert points to a 2D array with each row being a point
points = np.vstack([x, y, z])



# # Calculate the point density
density = stats.gaussian_kde(points)(points)
print("Density done")
# Start plotting
fig = plt.figure(figsize=(10, 10))##################################################
ax = fig.add_subplot(111, projection='3d')
#ax.set_facecolor((0.9, 0.9, 0.9))  # Adjust background color )##################################################

# Sort the points by density, so that the densest points are plotted last
idx = density.argsort()
x, y, z, density = x[idx], y[idx], z[idx], density[idx]

# Use density as the color
sc = ax.scatter(x, y, z, c=density, cmap='jet', alpha=0.005)


sc = ax.scatter(x, y, z, cmap='jet', alpha=0.005)
print("Outline")

# Plot the outline of the tetrahedron
vertices = np.array([a_vertex, b_vertex, c_vertex, d_vertex])
ax.plot_trisurf(*vertices.T, color='r', alpha=0.1)

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

# Extract convergent points for plotting
cartesian_convergent_points = []  
convergent_points = np.array(convergent_points)
print(convergent_points)

for i, p in enumerate(convergent_points):
    cartesian_convergent_points.append(np.dot(vertices.T, p))

cartesian_convergent_points = np.array(cartesian_convergent_points)
x_conv = cartesian_convergent_points[:, 0]
y_conv = cartesian_convergent_points[:, 1]
z_conv = cartesian_convergent_points[:, 2]

# Plot convergent points in different color and size
convergent_scatter = ax.scatter(x_conv, y_conv, z_conv, c='green', s=30, label='Convergent Points')

#Display convergent point barycentric coords on graph
convergent_points = np.around(convergent_points, 4)
unique_points = [convergent_points[0]]

# Grid and Axes enhancements######################################################################################################################################################
ax.view_init(elev=25, azim=-105)
# Remove tick labels to declutter:
ax.w_xaxis.pane.fill = ax.w_yaxis.pane.fill = ax.w_zaxis.pane.fill = False
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.grid(color="white", linestyle='solid')
plt.tight_layout()
# Save the image to your desktop with high DPI for better quality
filename = "/Users/noah/Desktop/3.png"
plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)

# Close the plt figure to release memory
plt.close()

# Crop the image to remove any whitespace (using PIL)
img = Image.open(filename)
cropped_img = img.crop(img.getbbox())
cropped_img.save(filename)

# #Bezier Curve
# P0_i = np.array([1, 0, 0, 0])
# P1_i = unique_points[0]
# P2_i = unique_points[1]
# P3_i = np.array([0, 0, 0, 1])
# P0 = np.dot(vertices.T, P0_i)
# P1 = np.dot(vertices.T, P1_i)
# P2 = np.dot(vertices.T, P3_i)
# C = 2*P1 - 0.5*(P0 + P2)
# def adjusted_bezier_curve(t, P0, C, P2):
#     return (1-t)**2 * P0 + 2 * (1-t) * t * C + t**2 * P2
# t = np.linspace(0, 1, 100)
# curve = np.array([adjusted_bezier_curve(ti, P0, C, P2) for ti in t])
# ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], 'b-', label='Bezier Curve')
# ax.scatter([P0[0], P1[0], P2[0]], [P0[1], P1[1], P2[1]], [P0[2], P1[2], P2[2]], color='red', s=60, label='Given Points')
# ax.scatter([C[0]], [C[1]], [C[2]], color='green', s=60, label='New Control Point')