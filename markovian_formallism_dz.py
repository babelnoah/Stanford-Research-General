#FINAL

import numpy as np
import matplotlib.pyplot as plt

def calculate_lambdas(x, z, R):
    x4 = 1 - x[0] - x[1] - x[2]

    lambdas = {
        'lambda11': (z[0] - 1) * x[0]**2,
        'lambda12': z[0] * x[1]**2,
        'lambda13': z[0] * x[2]**2,
        'lambda14': z[0] * x4**2,
        'lambda15': (2 * z[0] - 1) * x[0] * x[1],
        'lambda16': (2 * z[0] - 1) * x[0] * x[2],
        'lambda17': 2 * z[0] * x[1] * x4,
        'lambda18': 2 * z[0] * x[2] * x4,
        'lambda19': (2 * z[0] - (1 - R)) * x[0] * x4 + (2 * z[0] - R) * x[1] * x[2],

        'lambda21': z[1] * x[0]**2,
        'lambda22': (z[1] - 1) * x[1]**2,
        'lambda23': z[1] * x[2]**2,
        'lambda24': z[1] * x4**2,
        'lambda25': (2 * z[1] - 1) * x[0] * x[1],
        'lambda26': 2 * z[1] * x[0] * x[2],
        'lambda27': (2 * z[1] - 1) * x[1] * x4,
        'lambda28': 2 * z[1] * x[2] * x4,
        'lambda29': (2 * z[1] - R) * x[0] * x4 + (2 * z[1] - (1 - R)) * x[1] * x[2],

        'lambda31': z[2] * x[0]**2,
        'lambda32': z[2] * x[1]**2,
        'lambda33': (z[2] - 1) * x[2]**2,
        'lambda34': z[2] * x4**2,
        'lambda35': 2 * z[2] * x[0] * x[1],
        'lambda36': (2 * z[2] - 1) * x[0] * x[2],
        'lambda37': 2 * z[2] * x[1] * x4,
        'lambda38': (2 * z[2] - 1) * x[2] * x4,
        'lambda39': (2 * z[2] - R) * x[0] * x4 + (2 * z[2] - (1 - R)) * x[1] * x[2]
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

# Create a grid of x values to evaluate the transition density
N_z = 20
z_values = [np.array([i, j, k]) for i in np.linspace(0, 1, N_z) for j in np.linspace(0, 1, N_z) for k in np.linspace(0, 1, N_z) if i + j + k <= 1]
p_values = []

x = np.array([0.99,0.001,0.001])
h = 10**-2

num=0
for z in z_values:
    num+=1
    print(str(100*num/len(z_values)) + "% Complete")
    R = np.random.uniform(0, 0.5)
    p = partial_derivative_Q(x, z, h, R)
    p_values.append(p)

# 3D Plotting
z1_values = [x[0] for x in z_values]
z2_values = [x[1] for x in z_values]
z3_values = [x[2] for x in z_values]

z4_values = []
for i in range(len(z1_values)):
    z4_values.append(1 - z1_values[0] - z2_values[0] - z3_values[0])

# Stack them together for easier iteration
barycentric_coords = np.stack((z1_values, z2_values, z3_values, z4_values), axis=-1)

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