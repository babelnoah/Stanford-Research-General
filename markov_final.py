from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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

def euclidean_distance(a, b):
    return np.linalg.norm(a-b)

def Q(z, x, R):

    N = 1
    #########################################################################
    count = 0
    distance_x_to_z = euclidean_distance(x, z)

    for _ in range(N):
        initial_x = np.copy(x) # Store the initial x value
        for i in range(1000): # outer loop
            x = initial_x # Reset x to its initial value at the start of each outer iteration
            for _ in range(50):
                W = create_matrix()
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
                x_next[0] = (x[0] * W_avg[0] - W[0][3] * R * D) / w_bar
                x_next[1] = (x[1] * W_avg[1] + W[0][3] * R * D) / w_bar
                x_next[2] = (x[2] * W_avg[2] + W[0][3] * R * D) / w_bar
                x_next[3] = (x[3] * W_avg[3] - W[0][3] * R * D) / w_bar
                x = x_next

        dot_product = np.dot((z-initial_x),(x_next-initial_x))
        if dot_product < 0:
            count += 1

    probability = count / N
    return probability



def partial_derivative_Q(x, z, h, R):
    # Compute partial derivatives for each x_i
    Q1 = Q(x + np.array([h, 0, 0, 0]), z, R) - Q(x, z, R)
    Q2 = Q(x + np.array([0, h, 0, 0]), z, R) - Q(x, z, R)
    Q3 = Q(x + np.array([0, 0, h, 0]), z, R) - Q(x, z, R)
    Q4 = Q(x + np.array([0, 0, 0, h]), z, R) - Q(x, z, R)

    # Normalize by h to get the derivative
    Q1 /= h
    Q2 /= h
    Q3 /= h
    Q4 /= h

    return Q1, Q2, Q3, Q4

# Create a grid of x values to evaluate the transition density
z_values = np.random.dirichlet(np.ones(4), size=250)
#########################################################################
p_values = []

# Define the four x values
x_values = np.array([0.99,0.003,0.003,0.004])
#x_values = np.array([0.25,0.25,0.25,0.25])
# x_values = np.array([0.49,0.49,0.01,0.01])


h = 10**-5

num=0
p1_values, p2_values, p3_values, p4_values = [], [], [], []
for z in z_values:
    R = np.random.uniform(0, 0.5)
    p1, p2, p3, p4 = partial_derivative_Q(x_values, z, h, R)
    p1_values.append(p1)
    p2_values.append(p2)
    p3_values.append(p3)
    p4_values.append(p4)
    num+=1
    print("Progress: " + str(100*num/len(z_values)) + " Complete; x:" + str(x_values))

# 3D Plotting
z1_values = [x[0] for x in z_values]
z2_values = [x[1] for x in z_values]
z3_values = [x[2] for x in z_values]
z4_values = [x[3] for x in z_values]


# Stack them together for easier iteration
barycentric_coords = np.stack((z1_values, z2_values, z3_values, z4_values), axis=-1)

# Define coordinates of the tetrahedron vertices
a_vertex = np.array([np.sqrt(8/9), 0, -1/3])
b_vertex = np.array([-np.sqrt(2/9), np.sqrt(2/3), -1/3])
c_vertex = np.array([-np.sqrt(2/9), -np.sqrt(2/3), -1/3])
d_vertex = np.array([0,0,1])

vertices = np.array([a_vertex, b_vertex, c_vertex, d_vertex])

# Convert to Cartesian coordinates
cartesian_coords = np.dot(barycentric_coords, vertices)

# Extract x, y, z cartesian coordinates
x_cart = cartesian_coords[:, 0]
y_cart = cartesian_coords[:, 1]
z_cart = cartesian_coords[:, 2]

average_p_values = [(p1 + p2 + p3 + p4) / 4 for p1, p2, p3, p4 in zip(p1_values, p2_values, p3_values, p4_values)]

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')

plot = ax.scatter(x_cart, y_cart, z_cart, c=average_p_values, cmap='viridis', edgecolor='k', linewidth=0.02)


ax.view_init(elev=25, azim=-285)
ax.w_xaxis.pane.fill = ax.w_yaxis.pane.fill = ax.w_zaxis.pane.fill = False
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.grid(color="white", linestyle='solid')
plt.tight_layout()
plt.show()
# # Save the image to your desktop with a unique name for each x value
# filename = f"/Users/noah/Desktop/5.png"
# plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)

# # Close the plt figure to release memory
# plt.close()

# # Crop the image to remove any whitespace (using PIL)
# img = Image.open(filename)
# cropped_img = img.crop(img.getbbox())
# cropped_img.save(filename)