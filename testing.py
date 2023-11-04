import numpy as np
import matplotlib.pyplot as plt

def calculate_lambdas(x, z, R):
    x4 = 1 - x[0] - x[1] - x[2]

    lambdas = {
        'lambda11': (z[0] - 1) * x[0]**2,
        'lambda12': z[0] * x[1],
        'lambda13': z[0] * x[2]**2,
        'lambda14': z[0] * x4**2,
        'lambda15': (2 * z[0] - 1) * x[0] * x[1],
        'lambda16': (2 * z[0] - 1) * x[0] * x[2],
        'lambda17': 2 * z[0] * x[1] * x4,
        'lambda18': 2 * z[0] * x[2] * x4,
        'lambda19': (2 * z[0] - (1 - R)) * x[0] * x4 + (2 * z[0] - R) * x[1] * x[2],

        'lambda21': z[1] * x[0]**2,
        'lambda22': (z[1] - 1) * x[1],
        'lambda23': z[1] * x[2]**2,
        'lambda24': z[1] * x4**2,
        'lambda25': (2 * z[1] - 1) * x[0] * x[1],
        'lambda26': 2 * z[1] * x[0] * x[2],
        'lambda27': (2 * z[1] - 1) * x[1] * x4,
        'lambda28': 2 * z[1] * x[2] * x4,
        'lambda29': (2 * z[1] - R) * x[0] * x4 + (2 * z[1] - (1 - R)) * x[1] * x[2],

        'lambda31': z[2] * x[0]**2,
        'lambda32': z[2] * x[1],
        'lambda33': (z[2] - 1) * x[2]**2,
        'lambda34': z[2] * x4**2,
        'lambda35': 2 * z[2] * x[0] * x[1],
        'lambda36': (2 * z[2] - 1) * x[0] * x[2],
        'lambda37': 2 * z[2] * x[1] * x4,
        'lambda38': (2 * z[2] - 1) * x[2] * x4,
        'lambda39': (2 * z[2] - R) * x[0] * x4 + (2 * z[2] - (1 - R)) * x[0] * x4
    }

    return lambdas
def Q(z, x, R):

    # Using Monte Carlo to determine if (A,B,C,D,E,F,G,H,K,R) are within H(x,z)
    # Here, we assume some large number of samples for the simulation
    N = 10000
    count = 0

    for _ in range(N):
        A, B, C, D, E, F, G, H, K = np.random.uniform(0, 1, 9)
        R = np.random.uniform(0, 0.5)
        lambdas = calculate_lambdas(x, z, R)

        E1 = A*lambdas['lambda11'] + B*lambdas['lambda12'] + C*lambdas['lambda13'] + D*lambdas['lambda14'] + E*lambdas['lambda15'] + F*lambdas['lambda16'] + G*lambdas['lambda17'] + H*lambdas['lambda18'] + K*lambdas['lambda19']
        E2 = A*lambdas['lambda21'] + B*lambdas['lambda22'] + C*lambdas['lambda23'] + D*lambdas['lambda24'] + E*lambdas['lambda25'] + F*lambdas['lambda26'] + G*lambdas['lambda27'] + H*lambdas['lambda28'] + K*lambdas['lambda29']
        E3 = A*lambdas['lambda31'] + B*lambdas['lambda32'] + C*lambdas['lambda33'] + D*lambdas['lambda34'] + E*lambdas['lambda35'] + F*lambdas['lambda36'] + G*lambdas['lambda37'] + H*lambdas['lambda38'] + K*lambdas['lambda39']

        if E1 >= 0 and E2 >= 0 and E3 >= 0:
            count += 1


    probability = count / N

    return probability

def transition_density(z, x, h, R):  # Switch the roles of x and z
    Q1 = (Q(z + np.array([h, 0, 0]), x, R) - Q(z, x, R)) / h
    Q2 = (Q(z + np.array([0, h, 0]), x, R) - Q(z, x, R)) / h
    Q3 = (Q(z + np.array([0, 0, h]), x, R) - Q(z, x, R)) / h
    return np.array([Q1, Q2, Q3])

# Create a grid of x values to evaluate the transition density
N_x = 20
z_values = [np.array([i, j, k]) for i in np.linspace(0, 1, N_x) for j in np.linspace(0, 1, N_x) for k in np.linspace(0, 1, N_x) if i + j + k <= 1]
h = 10**-3

# Fix z to the corners
corners = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]

# For each z (corner), calculate the transition densities across the x grid
densities_per_corner = {}
for corner in corners:
    densities = []
    num = 0
    for z in z_values:
        num +=1
        print(str(100*num/len(z_values)) + "% Done (" + str(corner) + ")")
        R = np.random.uniform(0, 0.5)
        densities.append(transition_density(z, corner, h, R))
    densities_per_corner[str(corner)] = densities

# Plotting (only showing for one corner for brevity)
corner_to_plot = np.array([1,0,0])  # Change this to the desired corner
densities = densities_per_corner[str(corner_to_plot)]

# Convert densities to magnitudes
density_magnitudes = [np.linalg.norm(d) for d in densities]

# 3D Plotting
x1_values = [x[0] for x in z_values]
x2_values = [x[1] for x in z_values]
x3_values = [x[2] for x in z_values]

x4_values = []
for i in range(len(x1_values)):
    x4_values.append(1 - x1_values[0] - x2_values[0] - x3_values[0])

# Stack them together for easier iteration
barycentric_coords = np.stack((x1_values, x2_values, x3_values, x4_values), axis=-1)

# Define coordinates of the tetrahedron vertices
vertices = np.array([
    [np.sqrt(8/9), 0, -1/3],
    [-np.sqrt(2/9), np.sqrt(2/3), -1/3],
    [-np.sqrt(2/9), -np.sqrt(2/3), -1/3],
    [0,0,1]
])

# Convert to Cartesian coordinates
cartesian_coords = np.dot(barycentric_coords, vertices)

# Extract x, y, z cartesian coordinates
x_cart = cartesian_coords[:, 0]
y_cart = cartesian_coords[:, 1]
z_cart = cartesian_coords[:, 2]

# Create a 4-subplot figure (2x2 layout)
fig = plt.figure(figsize=(10, 10))

for idx, corner in enumerate(corners, start=1):
    densities = densities_per_corner[str(corner)]
    density_magnitudes = [np.linalg.norm(d) for d in densities]

    ax = fig.add_subplot(2, 2, idx, projection='3d')
    plot = ax.scatter(x_cart, y_cart, z_cart, c=density_magnitudes, cmap='viridis')
    ax.set_title(f'Corner: {corner}')
    ax.set_xlabel('Z1')
    ax.set_ylabel('Z2')
    ax.set_zlabel('Z3')
    fig.colorbar(plot, ax=ax)

# Define coordinates of the tetrahedron vertices
a_vertex = np.array([np.sqrt(8/9), 0, -1/3])
b_vertex = np.array([-np.sqrt(2/9), np.sqrt(2/3), -1/3])
c_vertex = np.array([-np.sqrt(2/9), -np.sqrt(2/3), -1/3])
d_vertex = np.array([0,0,1])

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

plt.tight_layout()
plt.show()