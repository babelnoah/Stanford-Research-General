import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Fitness values
s_values = [0.05, 0.1, 0.3]
r_values = [0.01, 0.1, 0.3]

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


#Plot D over 1000 generations for all given s and r values
# Create an empty DataFrame to store the results
df = pd.DataFrame(columns=['s', 'r', 'AB', 'Ab', 'aB', 'ab'])

for s in s_values:
    for r in r_values:
        # Calculate alpha, beta, gamma, delta based on s
        d = 1-(1-s)**2
        a = d
        g = s
        b = s
        # Run the simulation 1000 times
        for _ in range(100):
            x_initial = np.random.rand(4)
            x_initial = x_initial / np.sum(x_initial)  # Normalize to sum to 1
            D_final = iterate_generations(x_initial, r, d, a, b, g)
            if D_final >0.1:
                print(f"For s={s}, r={r}, final D={D_final}")
            # Add the results to the DataFrame
            df_new = pd.DataFrame({'s': [s], 'r': [r], 'D': [D_final]})
            df = pd.concat([df, df_new], ignore_index=True)



# Create a violin plot of the final D values
plt.figure(figsize=(15, 10))
sns.violinplot(x='s', y='D', hue='r', data=df)
plt.xlabel('Selection Coefficient (s)')
plt.ylabel('Final Linkage Disequilibrium (D)')
plt.title('Distribution of Final D Values for Different s and r')
plt.show()