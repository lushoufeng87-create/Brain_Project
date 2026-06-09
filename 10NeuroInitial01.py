# initial version of hebbian learning
# Periods of widespread simultaneous firing are followed by quiet periods
# caused by recovery-variable fatigue, after which the network resumes firing.

import numpy as np
import matplotlib.pyplot as plt

# PARAMETERS

N = 10

a, b, c, d = 0.02, 0.2, -65, 8

time_steps = 2000

decay = 0.95

learning_rate = 0.001
max_weight = 10

# NEURON STATES

v = np.full(N, -65.0)
u = b * v

# CONNECTION MATRIX

W = np.random.normal(6, 1, (N, N))

# Remove self-connections
np.fill_diagonal(W, 0)

# Make last 2 neurons inhibitory
for i in [8, 9]:
    W[i, :] = -np.abs(W[i, :])

# SYNAPTIC CURRENTS

syn = np.zeros((N, N))

# RECORDING

spike_times = []
spike_neurons = []

avg_weight_trace = []

# SIMULATION

for t in range(time_steps):

    # Synaptic decay
    syn *= decay

    # Input currents
    I = np.zeros(N)

    # Sensory neurons
    I[0] = 10
    I[1] = 10

    # Synaptic input
    for target in range(N):
        I[target] += np.sum(syn[:, target])

    # Track spikes
    spikes = []

    # Update neurons

    for i in range(N):

        dt=0.5

        v[i] += dt * (0.04 * v[i]**2 + 5*v[i] + 140 - u[i] + I[i])
        u[i] += a * (b*v[i] - u[i])

        if v[i] >= 30:

            spikes.append(i)

            spike_times.append(t)
            spike_neurons.append(i)

            v[i] = c
            u[i] += d

    # Send spikes

    for source in spikes:

        for target in range(N):

            if W[source, target] != 0:

                syn[source, target] += W[source, target]

    # Hebbian Learning

    for source in spikes:

        for target in spikes:

            if source != target:

                W[source, target] += learning_rate

                W[source, target] = np.clip(
                    W[source, target],
                    -max_weight,
                    max_weight
                )

    avg_weight_trace.append(np.mean(np.abs(W)))

# VISUALIZATION
plt.figure(figsize=(12,8))

# Spike raster
plt.subplot(2,1,1)

plt.scatter(
    spike_times,
    spike_neurons,
    s=5
)

plt.ylabel("Neuron")
plt.title("Spike Raster Plot")

# Weight evolution
plt.subplot(2,1,2)

plt.plot(avg_weight_trace)

plt.ylabel("Average |Weight|")
plt.xlabel("Time Step")
plt.title("Hebbian Learning")

plt.tight_layout()
plt.show()

# FINAL WEIGHTS

print("\nFinal Weight Matrix:\n")
print(np.round(W, 2))