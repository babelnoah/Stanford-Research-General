import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Create a directory for saving the plots
os.makedirs('results', exist_ok=True)

# Parameters
r_values = [2]
a_values = [0.25]
x_initial_values = [0.1, 0.5, 0.8, 1.2]
y_initial_values = [-0.5, -0.2, 0, 0.2, 0.5, 1.0]
eps = 1e-20  # small number for convergence check

# Fitness function
def calculate_next_generation(x_initial, y_initial, r, a):
    x_next = x_initial * (1 - r)**2 + a * r**2 + 2 * r * (1 - r) * np.sqrt(x_initial) * y_initial
    y_next = y_initial * (1 - r) + r * a * np.sqrt(x_initial)
    print("X: " + str(x_next) + " Y: " + str(y_next))
    return x_next, y_next

# Iterate over parameter sets
for i, r in enumerate(r_values):
    for j, a in enumerate(a_values):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(f"r={r}, a={a}")
        ax.set_xlabel('Time (iteration)')
        ax.set_ylabel('x')

        for x_initial in x_initial_values:
            for y_initial in y_initial_values:
                x, y = x_initial, y_initial
                x_values = [x]
                iteration = [0]
                counter = 0
                fail_flag = False
                
                for i in range(30000):
                    counter += 1
                    x_next, y_next = calculate_next_generation(x, y, r, a)
                    if np.iscomplex(x_next) or np.iscomplex(y_next):
                        print("Square Root of Negative Number")
                        break
                    if np.isinf(x_next) or np.isinf(y_next) or np.isnan(x_next) or np.isnan(y_next):
                        fail_flag = True
                        print("Overflow")
                        break  # break if overflow encountered
                    if np.abs(x - x_next) < eps and np.abs(y - y_next) < eps:
                        print("Convergence")
                        break  # convergence reached
                    x, y = x_next, y_next
                    x_values.append(x)
                    iteration.append(counter)


                ax.plot(iteration, x_values, label=f"x0={x_initial}, y0={y_initial}")
                
                if fail_flag:
                    ax.annotate(f'Failed: x0={x_initial}, y0={y_initial}', xy=(0.5, 0.5), 
                                       xycoords='axes fraction', fontsize=10, ha='center')
        
                ax.legend(fontsize='x-small')

        plt.show()
