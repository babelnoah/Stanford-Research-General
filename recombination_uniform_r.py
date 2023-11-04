import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Fitness values
s_values = [0.05, 0.1, 0.3,0.5,0.7,0.9,1]
r_values = np.linspace(0, 0.5, 10001)

#pass in arr x: x[0] = AB, x[1] = Ab, x[2] = aB, x[3] = ab
def calculate_next_generation(x, r, delta, alpha, beta, gamma):
    D = x[0]*x[3] - x[1]*x[2] 
    w_bar = 1 - delta*(x[0]**2 + x[3]**2) - alpha*(x[1]**2 + x[2]**2) - 2*beta*(x[2]*x[3] + x[0]*x[1]) - 2*gamma*(x[0]*x[2] + x[1]*x[3])
    x_next = np.zeros(4)
    #normalize by dividing by fitness function
    x_next[0] = (x[0] - delta*x[0]**2 - beta*x[0]*x[1] - gamma*x[0]*x[2] - r*D)/w_bar
    x_next[1] = (x[1] - beta*x[0]*x[1] - alpha*x[1]**2 - gamma*x[1]*x[3] + r*D)/w_bar
    x_next[2] = (x[2] - gamma*x[0]*x[2] - alpha*x[2]**2 - beta*x[2]*x[3] + r*D)/w_bar
    x_next[3] = (x[3] - gamma*x[1]*x[3] - beta*x[2]*x[3] - delta*x[3]**2 - r*D)/ w_bar
    result = [x_next,D]
    return result

def iterate_generations(x, r, delta, alpha, beta, gamma):
    D_values = []
    while True:
        result = calculate_next_generation(x, r, delta, alpha, beta, gamma)
        x_next = result[0]  # the new genotype frequencies
        D = result[1]  # the new value of D
        D_values.append(D)
        if len(D_values) > 1:
            if abs(D - D_values[-2]) < 10**-7:
                break
        x = x_next  # update the genotype frequencies for the next iteration
    return D  # Return the final D value

# Create an empty DataFrame to store the results
df = pd.DataFrame(columns=['s', 'r', 'D'])

for i_s, s in enumerate(s_values):
    for i_r, r in enumerate(r_values):
        # Print progress every 100 r-values for each s-value
        if i_r % 100 == 0:
            print(f'Processing: s = {s}, r index = {i_r+1}/{len(r_values)}')

        # Calculate alpha, beta, gamma, delta based on s
        d = 1-(1-s)**2
        a = d
        g = s
        b = s
        # Run the simulation 100 times
        for _ in range(100):
            x_initial = np.random.rand(4)
            x_initial = x_initial / np.sum(x_initial)  # Normalize to sum to 1
            D_final = iterate_generations(x_initial, r, d, a, b, g)
            # Add the results to the DataFrame
            df_new = pd.DataFrame({'s': [s], 'r': [r], 'D': [D_final]})
            df = pd.concat([df, df_new], ignore_index=True)

# Initialize a figure with 3 subplots, one for each s value
fig, axs = plt.subplots(len(s_values), 1, figsize=(10, 15))

# Loop over the s values and create a plot for each
for i, s in enumerate(s_values):
    # Filter the DataFrame to include only the rows for this s value
    df_s = df[df['s'] == s]
    # Create a scatter plot of r vs D
    axs[i].scatter(df_s['r'], df_s['D'], alpha=0.5)
    # Alternatively line plot:
    # axs[i].plot(df_s['r'], df_s['D'], '.-')
    axs[i].set_title(f's = {s}')
    axs[i].set_xlabel('Recombination Rate (r)')
    axs[i].set_ylabel('Final Linkage Disequilibrium (D)')

# Adjust the layout so the plots don't overlap
fig.tight_layout()
plt.show()