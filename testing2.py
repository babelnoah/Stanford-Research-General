import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates

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

def Q(x, z, R):
    lambdas = calculate_lambdas(x, z, R)

    # Define Ei conditions
    E1 = lambdas['lambda11'] + lambdas['lambda12'] + lambdas['lambda13'] + lambdas['lambda14'] + lambdas['lambda15'] + lambdas['lambda16'] + lambdas['lambda17'] + lambdas['lambda18'] + lambdas['lambda19']
    E2 = lambdas['lambda21'] + lambdas['lambda22'] + lambdas['lambda23'] + lambdas['lambda24'] + lambdas['lambda25'] + lambdas['lambda26'] + lambdas['lambda27'] + lambdas['lambda28'] + lambdas['lambda29']
    E3 = lambdas['lambda31'] + lambdas['lambda32'] + lambdas['lambda33'] + lambdas['lambda34'] + lambdas['lambda35'] + lambdas['lambda36'] + lambdas['lambda37'] + lambdas['lambda38'] + lambdas['lambda39']

    # Using Monte Carlo to determine if (A,B,C,D,E,F,G,H,K,R) are within H(x,z)
    # Here, we assume some large number of samples for the simulation
    N = 100
    count = 0

    for _ in range(N):
        A, B, C, D, E, F, G, H, K = np.random.uniform(0, 1, 9)
        R_sample = np.random.uniform(0, 0.5)

        if E1 >= 0 and E2 >= 0 and E3 >= 0:
            count += 1

    probability = count / N

    return probability
def partial_derivative_Q(x, z, h, R):
    Q1 = Q(x + np.array([h, 0, 0]), z, R)
    Q2 = Q(x + np.array([0, h, 0]), z, R)
    Q3 = Q(x + np.array([0, 0, h]), z, R)
    Q4 = Q(x, z, R)
    
    Q12 = Q(x + np.array([h, h, 0]), z, R)
    Q13 = Q(x + np.array([h, 0, h]), z, R)
    Q23 = Q(x + np.array([0, h, h]), z, R)

    Q123 = Q(x + np.array([h, h, h]), z, R)

    return (Q123 - Q13 - Q23 - Q12 + 2*Q4 - Q1 - Q2 - Q3) / h**3

N_x = 100  # Number of samples in x space
N_z = 50  # Number of samples in z space

# Generate random samples for x, ensuring they sum to less than 1
x_values = [np.random.dirichlet(np.ones(3), size=1)[0] for _ in range(N_x)]

z_values = [np.array([i, j, k]) for i in np.linspace(0, 1, N_z) for j in np.linspace(0, 1, N_z) for k in np.linspace(0, 1, N_z) if i + j + k <= 1]

val = 0
for x in x_values:
    val +=1
    print (str(100*val/len(x_values)) + "% Done")
    p_values = []

    for z in z_values:
        R = np.random.uniform(0, 0.5)
        h = 0.01
        p = partial_derivative_Q(x, z, h, R)
        p_values.append(p)

    avg_p = np.mean(p_values)
    
    # Create the 3D scatter plot colored by average p value
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 3D Plotting
x1_values = [z[0] for z in z_values]
x2_values = [z[1] for z in z_values]
x3_values = [z[2] for z in z_values]

# Convert x1, x2, x3 to 4D using x4 = 1 - x1 - x2 - x3
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


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot = ax.scatter(x_cart, y_cart, z_cart, c=p_values, cmap='viridis')
fig.colorbar(plot)
ax.set_xlabel('Z1')
ax.set_ylabel('Z2')
ax.set_zlabel('Z3')
plt.show()