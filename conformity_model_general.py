import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def calculate_next_generation(x_initial, y_initial, r, a):
    try:
        x_next = x_initial * (1 - r)**2 + a * r**2 + 2 * r * (1 - r) * np.sqrt(x_initial) * y_initial
        y_next = y_initial * (1 - r) + r * a * np.sqrt(x_initial)
        return x_next, y_next
    except ValueError:  # raised when calculating sqrt of a negative number
        return np.nan, np.nan  # we return NaN to indicate an invalid operation

def check_divergence(x_initial, y_initial, r, a):
    x, y = x_initial, y_initial
    eps = 1e-20
    for i in range(1000):
        x_next, y_next = calculate_next_generation(x, y, r, a)
        if np.isnan(x_next) or np.isnan(y_next):
            return 2  # invalid operation
        elif np.isinf(x_next) or np.isinf(y_next):
            return 1  # diverges
        elif np.abs(x - x_next) < eps and np.abs(y - y_next) < eps:
            return 0  # converges
        x, y = x_next, y_next
    return 0  # assume it converges if we didn't already return

# Create a grid of (r, a) values
r_values = np.linspace(-2,2, 200)
a_values = np.linspace(-1, 1, 100)

result = np.empty((len(r_values), len(a_values)))

var=0
# Calculate the status of the function for each (r, a) pair
for i, r in enumerate(r_values):
    for j, a in enumerate(a_values):
        result[i, j] = check_divergence(0.1, 0.5, r, a)
        print(str((100*var)/(len(r_values)*len(a_values))) + "% complete")
        var +=1

# Define a colormap
cmap = ListedColormap(['blue', 'red', 'yellow'])

# Plot the result
plt.imshow(result, extent=[0, 2, 0, 2], origin='lower', cmap=cmap)
plt.xlabel('k-values')
plt.ylabel('alpha values')
plt.title('Divergence map')
plt.colorbar(ticks=[0, 1, 2], label='0=Converges, 1=Negative in a sqrt, 2=Diverges')
plt.show()
