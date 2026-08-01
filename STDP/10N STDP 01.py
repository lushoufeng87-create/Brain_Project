# fixed network quiescence by increasing background noise
# changed to distance-based connectivity with stronger nearby connections
# added homeostatic plasticity to boost silent neurons
# neurons use exponential functions for STDP updates instead of linear decay, 
# they show that effects in the brain fade gradually. 

import numpy as np
import matplotlib.pyplot as plt

# PARAMETERS

N = 10

time_steps = 3000
dt = 0.5

decay = 0.97

# STDP

A_plus = 0.025 # initial weight increase value 
A_minus = 0.02 # initial weight decrease value

tau_plus = 20 # window of firing for which neurons can strengthen mutual connections
tau_minus = 20 # window of firing for which neurons can weaken mutual connections

max_weight = 10
min_weight = 0.3

# PARAMETERS

a = np.random.normal(0.02, 0.002, N)
b = np.random.normal(0.20, 0.01, N)

c = np.full(N, -65.0)
d = np.full(N, 8.0)

v = c.copy()
u = b * v

# DISTANCE-BASED CONNECTIVITY

W = np.zeros((N, N))

for i in range(N):
    for j in range(N): # checks all pairs of neurons

        if i == j:
            continue # neurons cannot connect to itself
        distance = abs(i - j)

        # probability decreases with distance
        p_connect = np.exp(-distance/2) # exp is e to the exponent function

        if np.random.random() < p_connect:

            # stronger nearby connections
            weight = ( 12 * np.exp(-distance / 3) + 
                      np.random.normal(0, 0.5) ) # random noise added to weight

            weight = max(weight, 0.5)

            W[i, j] = weight

# INHIBITORY NEURONS (20% OF NETWORK)

num_inhibitory = int(0.2 * N)

inhibitory_neurons = np.arange( N - num_inhibitory, N)

for neuron in inhibitory_neurons:
    W[neuron, :] = -np.abs(W[neuron, :])

# DELAYS

delays = np.ones((N, N), dtype=int)

for i in range(N):
    for j in range(N):
        if i == j:
            continue

        distance = abs(i - j)
        delays[i, j] = max( 1, distance )

# EVENT QUEUE

event_queue = [[] for _ in range(time_steps + 100)]

syn_current = np.zeros(N)

# STDP

last_spike = np.full(N, -10000) # placeholder value

# RECORDING

spike_times = []
spike_neurons = []

spike_counts = np.zeros(N)

avg_weight_trace = []

stdp_updates = 0

initial_W = W.copy()

# SIMULATION

for t in range(time_steps):

    syn_current *= decay

    I = syn_current.copy()

    # sensory input

    I[0] += 9 + np.random.normal(0, 1)
    I[1] += 9 + np.random.normal(0, 1)

    # cortical noise

    I += np.random.normal(0, 1.0, N)

    #homeostatic plasticity

    for i in range (N):
        silent_time = t - last_spike[i]

        if silent_time > 200:
            I[i] += 0.005 * (silent_time - 200)

    # delayed arrivals

    for target, current in event_queue[t]:
        syn_current[target] += current

    spikes = []

    # neuron update

    for i in range(N):

        dv = ( 0.04 * v[i]**2 + 5 * v[i] + 140 - u[i] + I[i] )
        du = a[i] * (b[i] * v[i] - u[i])

        v[i] += dt * dv
        u[i] += dt * du

        if v[i] >= 30:

            spikes.append(i)

            spike_times.append(t)
            spike_neurons.append(i)

            spike_counts[i] += 1

            v[i] = c[i]
            u[i] += d[i]

    # STDP

    for neuron in spikes:

        # strengthen causal inputs

        for source in range(N):

            if W[source, neuron] <= 0:
                continue

            dt_spike = t - last_spike[source]

            if 0 < dt_spike < 50:

                dW = ( A_plus * np.exp(-dt_spike / tau_plus) )

                W[source, neuron] += dW

                W[source, neuron] = min(
                    W[source, neuron],
                    max_weight
                )

                stdp_updates += 1

        # weaken anti-causal outputs

        for target in range(N):

            if W[neuron, target] <= 0:
                continue

            dt_spike = t - last_spike[target]

            if 0 < dt_spike < 50:

                dW = ( A_minus * np.exp(-dt_spike / tau_minus) )

                W[neuron, target] -= dW

                W[neuron, target] = max(
                    W[neuron, target], min_weight
                )

                stdp_updates += 1

        last_spike[neuron] = t

    # propagate spikes

    for source in spikes:

        for target in range(N):

            if W[source, target] == 0:
                continue

            arrival = t + delays[source, target]

            if arrival < len(event_queue):

                event_queue[arrival].append( ( target, W[source, target] ) )

    avg_weight_trace.append( np.mean(np.abs(W[W != 0])) )

# PLOTS - AI made

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)

plt.subplot(2,1,1)

plt.scatter(
    spike_times,
    np.array(spike_neurons) + 1,
    s=4
)

plt.yticks(range(1, N+1))

plt.ylabel("Neuron")
plt.title("Distance-Based STDP Network")

plt.subplot(2, 1, 2)

plt.plot(avg_weight_trace)

plt.ylabel("Average Weight")
plt.xlabel("Time")

plt.tight_layout()
plt.show()

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)

plt.imshow(
    initial_W,
    aspect='auto'
)

plt.colorbar()

plt.title("Initial Weights")

plt.xlabel("Target")
plt.ylabel("Source")

plt.subplot(1,2,2)

plt.imshow(
    W,
    aspect='auto'
)

plt.colorbar()

plt.title("Final Weights")

plt.xlabel("Target")
plt.ylabel("Source")

plt.tight_layout()
plt.show()

plt.figure(figsize=(6,6))

plt.imshow(
    W - initial_W,
    aspect='auto'
)

plt.colorbar(
    label='Weight Change'
)

plt.title("STDP Weight Changes")

plt.xlabel("Target")
plt.ylabel("Source")

plt.show()

# DIAGNOSTICS

print("\nSpike Counts")

for i in range(N):
    print(f"Neuron {i}: {int(spike_counts[i])}")

print("\nSTDP Updates:", stdp_updates)

print(
    "\nLargest Weight Change:",
    np.max(np.abs(W - initial_W))
)