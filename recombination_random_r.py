import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def calculate_next_generation(x, r, delta, alpha, beta, gamma):
    D = x[0]*x[3] - x[1]*x[2] 
    w_bar = 1 - delta*(x[0]**2 + x[3]**2) - alpha*(x[1]**2 + x[2]**2) - 2*beta*(x[2]*x[3] + x[0]*x[1]) - 2*gamma*(x[0]*x[2] + x[1]*x[3])
    x_next = np.zeros(4)
    x_next[0] = (x[0] - delta*x[0]**2 - beta*x[0]*x[1] - gamma*x[0]*x[2] - r*D)/w_bar
    x_next[1] = (x[1] - beta*x[0]*x[1] - alpha*x[1]**2 - gamma*x[1]*x[3] + r*D)/w_bar
    x_next[2] = (x[2] - gamma*x[0]*x[2] - alpha*x[2]**2 - beta*x[2]*x[3] + r*D)/w_bar
    x_next[3] = (x[3] - gamma*x[1]*x[3] - beta*x[2]*x[3] - delta*x[3]**2 - r*D)/ w_bar
    D_next = x_next[0]*x_next[3] - x_next[1]*x_next[2]
    return x_next, D_next

def iterate_generations(x, delta, alpha, beta, gamma):
    while True:
        r = np.random.uniform(0,0.05)
        x, D = calculate_next_generation(x, r, delta, alpha, beta, gamma)
        if abs(D - calculate_next_generation(x, r, delta, alpha, beta, gamma)[1]) < 10**-7:
            break
    return D

# Create an empty DataFrame to store the results
df_D = pd.DataFrame(columns=['D'])

d = 0.5
a = 0.1
g = 0.9
b = g 

for i in range(1000):  # Run the simulation 1,000 times
    print(str(i/10) + "% complete")
    x_initial = np.random.rand(4)
    x_initial = x_initial / np.sum(x_initial)
    D_final = iterate_generations(x_initial, d, a, b, g)

    df_new_D = pd.DataFrame({'D': [D_final]})
    df_D = pd.concat([df_D, df_new_D], ignore_index=True)

# Create combined histogram and KDE plot with histplot
plt.figure(figsize=(10, 5))
sns.histplot(df_D['D'], bins=50, kde=True)
plt.xlabel('Final Linkage Disequilibrium (D)')
plt.ylabel('Density')
plt.tight_layout()
plt.show()