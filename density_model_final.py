import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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
    D_next = x_next[0]*x_next[3] - x_next[1]*x_next[2]
    return x_next, D_next

def iterate_generations(x, delta, alpha, beta, gamma):
    points.append([x[0], x[1], x[2], x[3]])

    while True:
        #r = np.random.uniform(0, 0.05)
        r = 0.05
        x, D = calculate_next_generation(x, r, delta, alpha, beta, gamma)
        points.append([x[0], x[1], x[2], x[3]])
        if abs(D - calculate_next_generation(x, r, delta, alpha, beta, gamma)[1]) < 10**-10:
            break
    return D

a = 0.03
b = 0.004
d = 0.005
g = b

total_iterations = 10  # Number of tests
points = []

for i in range(total_iterations):
    progress = (i + 1) / total_iterations * 100
    print(f"Progress: {progress:.2f}%")

    iterate_generations(np.random.dirichlet(np.ones(4), size=1)[0],d,a,b,g)

# Convert points list to numpy array for easier manipulation
vertices = np.array([a_vertex, b_vertex, c_vertex, d_vertex])
points = np.array(points)

cartesian_points = []  # Define the cartesian_points list
total_points = len(points)

for i, element in enumerate(points):
    cartesian_points.append(np.dot(vertices.T, element))
    progress = (i + 1) / total_points * 100
    print(f"Progress: {progress:.2f}% completed.")

filtered_points = []

for point in cartesian_points:
    if is_point_in_tetrahedron(point, a_vertex, b_vertex, c_vertex, d_vertex):
        filtered_points.append(point)

points = np.array(filtered_points)

# Extract coordinates for plotting
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

# Convert points to a 2D array with each row being a point
points = np.vstack([x, y, z])
print("Density")
# Calculate the point density
density = stats.gaussian_kde(points)(points)

# Create 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
print("Density")
# Sort the points by density, so that the densest points are plotted last
idx = density.argsort()
x, y, z, density = x[idx], y[idx], z[idx], density[idx]
print("Density")
# Use density as the color
sc = ax.scatter(x, y, z, c=density, cmap='jet', alpha=0.003)
print("Density")
# Plot the outline of the tetrahedron
vertices = np.array([a_vertex, b_vertex, c_vertex, d_vertex])
ax.plot_trisurf(*vertices.T, color='r', alpha=0.1)

# Add a colorbar for the density
fig.colorbar(sc, ax=ax, label='Density')

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

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Randomly Generated Points')

# Show the plot
plt.show()