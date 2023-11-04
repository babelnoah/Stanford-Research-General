import numpy as np
import matplotlib.pyplot as plt
import random

a = 0.02
b = 0.04
d = 0.03
g = b

a_vertex = np.array([np.sqrt(8/9), 0, -1/3])
b_vertex = np.array([-np.sqrt(2/9), np.sqrt(2/3), -1/3])
c_vertex = np.array([-np.sqrt(2/9), -np.sqrt(2/3), -1/3])
d_vertex = np.array([0,0,1])

total_iterations = 100  # Number of tests
convergent_points = []
points = []

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
    #D_next = x_next[0]*x_next[3] - x_next[1]*x_next[2]

    # Normalize x_next so that it sums to 1
    x_next /= np.sum(x_next)
    
    return x_next

def iterate_generations(x, delta, alpha, beta, gamma):
    stable_generations = 0  # Initialize counter for stable generations
    change_threshold = 10**-15  # Set change threshold

    #Trackers
    # change_history = []  # Initialize change history
    # lowest_max_change = np.inf  # Initialize variable to track lowest max change
    # running_sum = 0  # Initialize running sum for average calculation

    while True:
        # Save current x for later comparison
        x_old = x.copy()
        r = 0.04
        #r = np.random.uniform(0,0.04)
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

for i in range(total_iterations):
    progress = (i + 1) / total_iterations * 100
    print(f"Progress: {progress:.2f}%")
    iterate_generations(np.random.dirichlet(np.ones(4), size=1)[0],d,a,b,g)
    convergent_points.append(points[-1])

# Convert points list to numpy array for easier manipulation
vertices = np.array([a_vertex, b_vertex, c_vertex, d_vertex])
points = np.array(points)


#GRAPH


# Create 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the outline of the tetrahedron
vertices = np.array([a_vertex, b_vertex, c_vertex, d_vertex])
ax.plot_trisurf(*vertices.T, color='r', alpha=0.1)

# Calculate the barycentric coordinates for these points
barycentric_coords = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0.248,0.251,0.249,0.252],[0.497,0.001,0.002,0.500],[0.001,0.497,0.500,0.002],[0.1539,0.6562,0.036,0.1539],[0.1539,0.036,0.6562,0.1539],[0.8888,0.05,0.05,0.0112],[0.0112,0.05,0.05,0.8888]])
labels = ['AB', 'Ab', 'aB', 'ab','s1','s2','s3','p1','p2','p3','p4']
barycentric_to_cartesian = []

for i, element in enumerate(barycentric_coords):
    barycentric_to_cartesian.append(np.dot(vertices.T, element))

barycentric_to_cartesian = np.array(barycentric_to_cartesian)

for i in range(len(barycentric_to_cartesian)):
    x, y, z = barycentric_to_cartesian[i]
    ax.scatter(x, y, z, color='black')
    ax.text(x, y, z, labels[i], color='black')

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
convergent_scatter = ax.scatter(x_conv, y_conv, z_conv, c='green', s=85, label='Convergent Points')

#Display convergent point barycentric coords on graph
convergent_points = np.around(convergent_points, 4)
unique_points = [convergent_points[0]]
for point in convergent_points[1:]:
    if not any(np.allclose(point, unique_point, atol=0.0175) for unique_point in unique_points):
        unique_points.append(point)
unique_points_str = '\n'.join(map(str, unique_points))
ax.text2D(0.05, 0.95, f'Convergent points:\n{unique_points_str}', transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Add a legend
ax.legend()

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'Random k (k = [0,0.04], alpha: {a}, beta: {b}, delta: {d})')

# Show the plot
plt.show()
print("Done")