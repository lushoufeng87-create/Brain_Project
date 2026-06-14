import numpy as np
import matplotlib.pyplot as plt

# parameters

N = 5  # number of neurons

a, b, c, d = 0.02, 0.2, -65, 8

decay = 0.95
time_steps = 1000


# neuron States

v = np.full(N, -65.0)
u = b * v


# weight matrix
# W[i,j] means neuron i to neuron j. 0 means no connection,
# positive means excitatory, negative means inhibitory.

W = np.array([
    [0, 4, 3, 0, 0],
    [0, 0, 0, 4, 0],
    [0, 0, 0, 2, 0],
    [0, 0, 0, 0, 4],
    [0, 0, 0, 0, 0]
], dtype=float)

# synaptic Currents

syn = np.zeros((N, N))

# storing values

voltage_traces = [[] for _ in range(N)]

# simulation

for t in range(time_steps):

    # decay all synapses
    syn *= decay

    # compute input currents
    I = np.zeros(N)

    # external input to Neuron 0
    I[0] = 5

    # add synaptic inputs
    for target in range(N):
        I[target] += np.sum(syn[:, target])

    # track which neurons spike
    spikes = []

    # update all neurons
    for i in range(N):

        v[i] += 0.04 * v[i]**2 + 5*v[i] + 140 - u[i] + I[i]
        u[i] += a * (b*v[i] - u[i])

        if v[i] >= 30:

            spikes.append(i)

            voltage_traces[i].append(30)

            v[i] = c
            u[i] += d

        else:
            voltage_traces[i].append(v[i])

    # send signals through synapses
    for source in spikes:

        for target in range(N):

            if W[source, target] != 0:

                syn[source, target] += W[source, target]

# plot results

plt.figure(figsize=(12, 6))

for i in range(N):
    plt.plot(
        voltage_traces[i],
        label=f"Neuron {i}"
    )

plt.title("5-Neuron Network")
plt.xlabel("Time Step")
plt.ylabel("Membrane Potential (mV)")
plt.legend()
plt.show()