# RANDOMIZED Izhikevich network with hebbian Learning
# heterogeneity added
# transmission delays added

import numpy as np
import matplotlib.pyplot as plt

# SIMULATION PARAMETERS

N = 10
time_steps = 2000
dt = 0.5

decay = 0.97

learning_rate = 0.001
max_weight = 10

# HETEROGENEOUS NEURON PARAMETERS

a = np.random.normal(0.02, 0.002, N)
b = np.random.normal(0.20, 0.01, N)

c = np.full(N, -65.0)
d = np.full(N, 8.0)

v = c.copy()
u = b * v

# NETWORK WEIGHTS

W = np.zeros((N,N))

mask = np.random.random((N,N)) < 0.4

W[mask] = np.random.normal(8, 2, np.sum(mask))

# Remove self-connections
np.fill_diagonal(W, 0)

# Last 2 neurons inhibitory
for i in [8, 9]:
    W[i, :] = -0.5 * np.abs(W[i, :])

# TRANSMISSION DELAYS

delays = np.random.randint(1, 15, (N, N))

# EVENT QUEUE

event_queue = [[] for _ in range(time_steps + 100)]

# SYNAPTIC CURRENTS

syn_current = np.zeros(N)

# RECORDING

spike_times = []
spike_neurons = []

avg_weight_trace = []

spike_counts = np.zeros(N)

# MAIN LOOP

for t in range(time_steps):

    # Decay synaptic currents

    syn_current *= decay

    # Input currents

    I = syn_current.copy()

    # Sensory neurons
    I[0] += 10 + np.random.normal(0, 1)
    I[1] += 10 + np.random.normal(0, 1)

    # Small background noise
    I += np.random.normal(0, 0.3, N)

    # Deliver delayed spikes

    for target, current in event_queue[t]:
        syn_current[target] += current

    # Update neurons

    spikes = []

    for i in range(N):

        dv = (
            0.04 * v[i]**2
            + 5 * v[i]
            + 140
            - u[i]
            + I[i]
        )

        du = a[i] * (b[i] * v[i] - u[i])

        v[i] += dt * dv
        u[i] += dt * du

        # Safety clamp
        v[i] = np.clip(v[i], -100, 100)

        if v[i] >= 30:

            spikes.append(i)

            spike_times.append(t)
            spike_neurons.append(i)

            spike_counts[i] += 1

            v[i] = c[i]
            u[i] += d[i]

    # Schedule future spikes

    for source in spikes:

        for target in range(N):

            if source == target:
                continue

            arrival_time = t + delays[source, target]

            if arrival_time < len(event_queue):

                event_queue[arrival_time].append(
                    (
                        target,
                        W[source, target]
                    )
                )

    # Hebbian Learning

    recent_window = 20

    recent_neurons = []

    for tt, neuron in zip(spike_times, spike_neurons):

        if t - tt <= recent_window:
            recent_neurons.append(neuron)

    for source in spikes:

        for target in recent_neurons:

            if source == target:
                continue

            growth = (
                learning_rate
                * (1 - abs(W[source, target]) / max_weight)
            )

            W[source, target] += growth

            W[source, target] = np.clip(
                W[source, target],
                -max_weight,
                max_weight
            )

    avg_weight_trace.append(
        np.mean(np.abs(W))
    )

# DIAGNOSTICS

print("\nSpike Counts")
print("----------------")

for i in range(N):
    print(f"Neuron {i}: {int(spike_counts[i])}")

print("\nAverage Weight:")
print(np.mean(np.abs(W)))

print("\nMaximum Weight:")
print(np.max(np.abs(W)))

# RESULTS

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)

plt.scatter(
    spike_times,
    spike_neurons,
    s=4
)

plt.title("Spike Raster Plot")
plt.ylabel("Neuron")

plt.subplot(2, 1, 2)

plt.plot(avg_weight_trace)

plt.title("Hebbian Learning")
plt.ylabel("Average |Weight|")
plt.xlabel("Time Step")

plt.tight_layout()
plt.show()

