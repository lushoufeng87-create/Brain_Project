import numpy as np
import matplotlib.pyplot as plt

# PARAMETERS

N = 10

time_steps = 10000
dt = 0.5

decay = 0.97

# STDP parameters

A_plus = 0.02
A_minus = 0.025

tau_plus = 20
tau_minus = 20

max_weight = 10
min_weight = 0

# HETEROGENEOUS NEURONS

a = np.random.normal(0.02, 0.002, N)
b = np.random.normal(0.20, 0.01, N)

c = np.full(N, -65.0)
d = np.full(N, 8.0)

v = c.copy()
u = b * v

# STRUCTURED CONNECTIVITY

W = np.zeros((N, N))

# Feedforward

W[0,2] = 6
W[0,3] = 5

W[1,3] = 6
W[1,4] = 5

W[2,5] = 5
W[3,5] = 5
W[3,6] = 5
W[4,7] = 8

W[5,8] = 6
W[6,8] = 6
W[7,8] = 6

# Feedback

W[5,2] = 2
W[6,3] = 2
W[7,4] = 2

# Excitatory to inhibitory

W[5,9] = 4
W[6,9] = 4
W[7,9] = 4
W[8,9] = 4

# Inhibitory outputs

W[9,2] = -5
W[9,3] = -5
W[9,4] = -5

W[9,5] = -4
W[9,6] = -4
W[9,7] = -4

W[9,8] = -3

initial_W = W.copy()

# DELAYS, CHANGE TO DISTANCE-BASED DELAYS LATER

delays = np.ones((N,N), dtype=int)

delays[0,2] = 2
delays[0,3] = 3

delays[1,3] = 2
delays[1,4] = 3

delays[2,5] = 4
delays[3,5] = 4
delays[3,6] = 5
delays[4,7] = 4

delays[5,8] = 3
delays[6,8] = 3
delays[7,8] = 3

# EVENT QUEUE

event_queue = [[] for _ in range(time_steps + 100)]

syn_current = np.zeros(N)

# STDP STORAGE

last_spike = np.full(N, -10000)

# RECORDING

spike_times = []
spike_neurons = []

spike_counts = np.zeros(N)

avg_weight_trace = []

stdp_updates = 0

# MAIN SIMULATION

for t in range(time_steps):

    syn_current *= decay

    I = syn_current.copy()

    # sensory neurons

    I[0] += 9 + np.random.normal(0,1)
    I[1] += 9 + np.random.normal(0,1)

    # background noise

    I += np.random.normal(0,0.2,N)

    # delayed arrivals

    for target, current in event_queue[t]:
        syn_current[target] += current

    spikes = []

    # NEURON UPDATE

    for i in range(N):

        dv = (
            0.04*v[i]**2
            + 5*v[i]
            + 140
            - u[i]
            + I[i]
        )

        du = a[i]*(b[i]*v[i]-u[i])

        v[i] += dt*dv
        u[i] += dt*du

        if v[i] >= 30:

            spikes.append(i)

            spike_times.append(t)
            spike_neurons.append(i)

            spike_counts[i] += 1

            v[i] = c[i]
            u[i] += d[i]

    # STDP

    for neuron in spikes:

        # neuron just fired

        for source in range(N):

            if W[source, neuron] <= 0:
                continue

            dt_spike = t - last_spike[source]

            if 0 < dt_spike < 50:

                dW = (
                    A_plus
                    * np.exp(-dt_spike / tau_plus)
                )

                W[source, neuron] += dW

                W[source, neuron] = min(
                    W[source, neuron],
                    max_weight
                )

                stdp_updates += 1

        for target in range(N):

            if W[neuron, target] <= 0:
                continue

            dt_spike = t - last_spike[target]

            if 0 < dt_spike < 50:

                dW = (
                    A_minus
                    * np.exp(-dt_spike / tau_minus)
                )

                W[neuron, target] -= dW

                W[neuron, target] = max(
                    W[neuron, target],
                    min_weight
                )

                stdp_updates += 1

        last_spike[neuron] = t

    # PROPAGATE SPIKES

    for source in spikes:

        for target in range(N):

            if W[source,target] == 0:
                continue

            arrival = t + delays[source,target]

            if arrival < len(event_queue):

                event_queue[arrival].append(
                    (
                        target,
                        W[source,target]
                    )
                )

    # RECORDING

    avg_weight_trace.append(
        np.mean(np.abs(W[W != 0]))
    )

# DIAGNOSTICS

print("\nSpike Counts")

for i in range(N):
    print(f"Neuron {i}: {int(spike_counts[i])}")

print("\nSTDP Updates:")
print(stdp_updates)

print("\nLargest Weight Change:")
print(np.max(np.abs(W - initial_W)))

# PLOTS

plt.figure(figsize=(8, 8))

# Spike raster

plt.subplot(3,1,1)

plt.scatter(
    spike_times,
    spike_neurons,
    s=3
)

plt.ylabel("Neuron")
plt.title("Spike Raster")

# Average weight

plt.subplot(3,1,2)

plt.plot(avg_weight_trace)

plt.ylabel("Avg Weight")
plt.title("Average Synaptic Strength")

# Weight histogram

plt.subplot(3,1,3)

active_weights = W[W != 0]

plt.hist(
    active_weights,
    bins=10
)

plt.xlabel("Weight")
plt.ylabel("Count")
plt.title("Final Weight Distribution")

plt.tight_layout()
plt.show()