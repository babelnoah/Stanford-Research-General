import numpy as np
import matplotlib.pyplot as plt

def calculate_next_generation(x_c_initial, x_n_initial, r, p,delta_n,delta_c,epsilon):
    x_c_next = x_c_initial + r*(1-p)*(x_n_initial - x_c_initial) + r*(delta_c)*(np.sqrt(p*(1-p)))*(x_n_initial - x_c_initial) + r*(delta_c)*(epsilon)
    x_n_next = x_n_initial + r*(-p)*(x_n_initial - x_c_initial) + r*(delta_n)*(np.sqrt(p*(1-p)))*(x_n_initial - x_c_initial) + r*(delta_n)*(epsilon)
    return x_c_next, x_n_next

#initial conditions
p = 1/2
delta_c = 0
delta_n = 0.4
epsilon = 0.5
x_c = 0.2
x_n = 0.8

def check_convergence(p,delta_c,delta_n,epsilon,x_c,x_n,r):
    stable_generations = 0  # Initialize counter for stable generations
    change_threshold = 10**-15  # Set change threshold
    max_generations = 10**5  # Maximum number of generations to avoid infinite loop

    num_generations = 0
    difference = []
    while num_generations < max_generations:
        try:
            num_generations +=1
            x_old = np.array([x_c,x_n])
            x_c, x_n = calculate_next_generation(x_c, x_n, r, p,delta_n,delta_c,epsilon)
            x = np.array([x_c,x_n])

            difference.append(x_n-x_c)

            # Check if absolute change in all components of x is less than threshold
            if np.all(np.abs(x - x_old) < change_threshold):
                stable_generations += 1
            else:
                # Reset counter if change is above threshold
                stable_generations = 0

            if stable_generations >= 100:
                break
        except OverflowError:
            return np.inf, np.inf # Return infinity for both x_c and x_n if OverflowError is encountered

    if num_generations >= max_generations:
        print(f'OverflowError at r = {r}, num_generations = {num_generations}, x_c = {x_c}, x_n = {x_n}')
        return np.inf, np.inf # Return infinity if max_generations is reached
    return x_c, x_n



# Create a range of r values
r_values = np.linspace(0,1, 100)
#print(r_values)

#Create list of convergent values
final_x_c = []
final_x_n = []

var=0 #track number of runs

# Calculate the status of the function for every r value
for r in r_values:
    x_c, x_n = 0.2, 0.8  # Reset initial conditions
    x_c_final, x_n_final = check_convergence(p,delta_c,delta_n,epsilon,x_c,x_n,r)
    final_x_c.append(x_c_final)
    final_x_n.append(x_n_final)
    #print(str((100*var)/(len(r_values))) + "% complete")
    var +=1

print(x_c_final)
#Plotting the graphs
plt.figure(figsize=(10, 5))

plt.subplot(1,2,1)
plt.plot(r_values, final_x_c, label='x_c')
plt.ylim([0, 100])  # set the ylim to bottom, top
plt.xlabel('r')
plt.ylabel('Convergent x_c')
plt.title('Convergent x_c as a function of r')

plt.subplot(1,2,2)
plt.plot(r_values, final_x_n, label='x_n')
plt.ylim([0, 10])  # set the ylim to bottom, top
plt.xlabel('r')
plt.ylabel('Convergent x_n')
plt.title('Convergent x_n as a function of r')

plt.tight_layout()
plt.show()

