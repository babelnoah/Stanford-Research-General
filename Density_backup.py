import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from tqdm import tqdm
from matplotlib import cm
import random
from scipy import stats


def system_of_equations(p, a, b, c, d, r1, r2, r3, r4):
    x, y, z = p
    return [(x - a[0])**2 + (y - a[1])**2 + (z - a[2])**2 - r1**2,
            (x - b[0])**2 + (y - b[1])**2 + (z - b[2])**2 - r2**2,
            (x - c[0])**2 + (y - c[1])**2 + (z - c[2])**2 - r3**2,
            (x - d[0])**2 + (y - d[1])**2 + (z - d[2])**2 - r4**2]

def calculate_distances(point, verticies):
    distances = []
    for p in verticies:
        distance = np.linalg.norm(point - p)
        distances.append(distance)
    return distances

def filter_points(points, reference_points, distance_threshold):
    filtered_points = []
    
    for point in points:
        count = 0
        for ref_point in reference_points:
            distance = np.linalg.norm(point - ref_point)
            if distance <= distance_threshold:
                count += 1
                if count >= 4:
                    filtered_points.append(point)
                    break
                else:
                    print("Off point: " + str(np.sum(calculate_distances(point,[a_vertex, b_vertex, c_vertex, d_vertex]))))
            else:
                    print("Off point: " + str(np.sum(calculate_distances(point,[a_vertex, b_vertex, c_vertex, d_vertex]))))
                    
    
    return np.array(filtered_points)

def generate_point_in_tetrahedron(v0, v1, v2, v3):
    s = random.random()
    t = random.random()
    u = random.random()

    if s + t > 1.0:
        s = 1.0 - s
        t = 1.0 - t

    if t + u > 1.0:
        tmp = u
        u = 1.0 - s - t
        t = 1.0 - tmp
    elif s + t + u > 1.0:
        tmp = u
        u = s + t + u - 1.0
        s = 1.0 - t - tmp

    a = 1 - s - t - u  # a, s, t, u are the barycentric coordinates of the random point.
    return v0 * a + v1 * s + v2 * t + v3 * u

# Set the initial guess
initial_guess = np.array([0, 0, 0])

# Define coordinates of the tetrahedron vertices
a_vertex = np.array([np.sqrt(8/9), 0, -1/3])
b_vertex = np.array([-np.sqrt(2/9), np.sqrt(2/3), -1/3])
c_vertex = np.array([-np.sqrt(2/9), -np.sqrt(2/3), -1/3])
d_vertex = np.array([0,0,1])

# Normalize factor (side length)
side_length = np.sqrt(8/3)

def calculate_next_generation(x, r, delta, alpha, beta, gamma):
    D = x[0]*x[3] - x[1]*x[2] 
    w_bar = 1 - delta*(x[0]**2 + x[3]**2) - alpha*(x[1]**2 + x[2]**2) - 2*beta*(x[2]*x[3] + x[0]*x[1]) - 2*gamma*(x[0]*x[2] + x[1]*x[3])
    x_next = np.zeros(4)
    x_next[0] = (x[0] - delta*x[0]**2 - beta*x[0]*x[1] - gamma*x[0]*x[2] - r*D)/w_bar
    x_next[1] = (x[1] - beta*x[0]*x[1] - alpha*x[1]**2 - gamma*x[1]*x[3] + r*D)/w_bar
    x_next[2] = (x[2] - gamma*x[0]*x[2] - alpha*x[2]**2 - beta*x[2]*x[3] + r*D)/w_bar
    x_next[3] = (x[3] - gamma*x[1]*x[3] - beta*x[2]*x[3] - delta*x[3]**2 - r*D)/ w_bar
    D_next = x_next[0]*x_next[3] - x_next[1]*x_next[2]

    # Normalize x_next so that it sums to 1
    x_next /= np.sum(x_next)
    
    return x_next, D_next

def iterate_generations(x, delta, alpha, beta, gamma):
    points.append([x[0], x[1], x[2], x[3]])

    while True:
        r = 0.05
        x, D = calculate_next_generation(x, r, delta, alpha, beta, gamma)
        points.append([x[0], x[1], x[2], x[3]])
        if abs(D - calculate_next_generation(x, r, delta, alpha, beta, gamma)[1]) < 10**-7:
            break
    return D

d = 0.005
a = 0.03
g = 0.004
b = g 

a_vertex = np.array([np.sqrt(8/9), 0, -1/3])
b_vertex = np.array([-np.sqrt(2/9), np.sqrt(2/3), -1/3])
c_vertex = np.array([-np.sqrt(2/9), -np.sqrt(2/3), -1/3])
d_vertex = np.array([0,0,1])

total_iterations = 100  # Number of tests
points = []
for i in range(total_iterations):
    progress = (i + 1) / total_iterations * 100
    print(f"Progress: {progress:.2f}%")

    iterate_generations(np.random.dirichlet(np.ones(4), size=1)[0],d,a,b,g)
    #calculate_distances(generate_point_in_tetrahedron(a_vertex,b_vertex,c_vertex,d_vertex),[a_vertex, b_vertex, c_vertex, d_vertex]), d, a, b, g

# Convert points list to numpy array for easier manipulation
vertices = np.array([a_vertex, b_vertex, c_vertex, d_vertex])
points = np.array(points)

cartesian_points = []  # Define the cartesian_points list
total_points = len(points)

for i, element in enumerate(points):
    cartesian_points.append(np.dot(vertices.T, element))
    progress = (i + 1) / total_points * 100
    print(f"Progress: {progress:.2f}% completed.")

points = np.array(cartesian_points)

# Extract coordinates for plotting
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

print("x: " + str(len(x)))
print("y: " + str(len(y)))
print("z: " + str(len(z)))

# Convert points to a 2D array with each row being a point
points = np.vstack([x, y, z])
# Calculate the point density
density = stats.gaussian_kde(points)(points)

# Create 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Sort the points by density, so that the densest points are plotted last
idx = density.argsort()
x, y, z, density = x[idx], y[idx], z[idx], density[idx]

# Use density as the color
sc = ax.scatter(x, y, z, c=density, cmap='jet', alpha=0.003)

# Plot the outline of the tetrahedron
vertices = np.array([a_vertex, b_vertex, c_vertex, d_vertex])
ax.plot_trisurf(*vertices.T, color='r', alpha=0.1)

# Add a colorbar for the density
fig.colorbar(sc, ax=ax, label='Density')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Randomly Generated Points')

# Show the plot
plt.show()
