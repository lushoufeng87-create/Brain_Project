# systemic manual weights are added, structured, predictable model
# organized layers for neurons, mimicking a tiny cortical circuit
# hebbian updates = 0. learning rule not triggered, because
# random synchronization was deliberately added, yet hebbian learning only works
# when neurons fire together. 

#new learning rule is required. 

import numpy as np
import matplotlib.pyplot as plt

# PARAMETERS

N = 10 # 10 neurons

time_steps = 3000
dt = 0.5 # splitting computation into half steps for better numerical stability

decay = 0.97

learning_rate = 0.001
max_weight = 10

# HETEROGENEOUS IZHIKEVICH NEURONS

a = np.random.normal(0.02, 0.002, N) # mean, std, size
# biologically, smaller a means slower recovery, larger means faster recovery.

b = np.random.normal(0.20, 0.01, N) # mean, std, size
# higher b means  stronger coupling between voltage and recovery

c = np.full(N, -65.0) # reset voltage after spike
d = np.full(N, 8.0) # increase in recovery variable after spike
# c, d are consistent for all 10 neurons just to control variability for now

v = c.copy() # independent array for voltage
u = b * v

# STRUCTURED CONNECTIVITY

W = np.zeros((N, N)) # initial weight matrix

# altering weight matrix manually
# Feedforward

W[0,2] = 6
W[0,3] = 5

W[1,3] = 6
W[1,4] = 5

W[2,5] = 5
W[3,5] = 5
W[3,6] = 5
W[4,7] = 10

W[5,8] = 6
W[6,8] = 6
W[7,8] = 6

# Feedback

W[5,2] = 2
W[6,3] = 2
W[7,4] = 2

# Interneuron input

W[5,9] = 4
W[6,9] = 4
W[7,9] = 4
W[8,9] = 4

# Interneuron inhibition, only 1 for now, usually 20% inhibitory

W[9,2] = -5
W[9,3] = -5
W[9,4] = -5

W[9,5] = -4
W[9,6] = -4
W[9,7] = -4

W[9,8] = -3
# more sophistication than a random connectivity network.

# DELAYS

delays = np.ones((N,N), dtype=int) 
# dtype=int forces integer values instead of floating point numbers '1.' in array

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
# next time will generate delay values based on distance between neurons

# most of these connections don't exist becaues their weights = 0, so
# their delay values are never used.

# EVENT QUEUE

event_queue = [[] for _ in range(time_steps + 100)]
# extra 100 time steps as buffer for delayed timesteps over 1000

syn_current = np.zeros(N)
# stores all incoming synaptic currents per time step

# RECORDING

spike_times = []
spike_neurons = []

spike_counts = np.zeros(N)

avg_weight_trace = []

weight_history = []

# SIMULATION

hebb_updates = 0

for t in range(time_steps):

    weight_history.append(W[0,2])

    syn_current *= decay

    I = syn_current.copy()

    # sensory input

    I[0] += 9 + np.random.normal(0,1)
    I[1] += 9 + np.random.normal(0,1)

    # background cortical noise

    I += np.random.normal(0,0.2,N)
    # real neurons never set in perfect silence. 
    # this noise term approximates the random fluctuations in input.

    # delayed arrivals

    for target, current in event_queue[t]:
        syn_current[target] += current

    spikes = []

    # neuron update

    for i in range(N):

        dv = (0.04*v[i]**2 + 5*v[i] + 140 - u[i] + I[i])

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

    # propagate spikes

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

    # Hebbian learning

    for source in spikes:

        for target in spikes:

            if source == target:
                continue

            if W[source,target] == 0:
                continue

            if W[source,target] > 0:

                W[source,target] += (
                    learning_rate
                    * (1 - W[source,target]/max_weight)
                )
                hebb_updates += 1
    avg_weight_trace.append(
        np.mean(np.abs(W[W != 0]))

    )

# DIAGNOSTICS

print("\nSpike Counts")

for i in range(N):
    print(f"Neuron {i}: {int(spike_counts[i])}")

print("\nFinal Weights")
print(np.round(W,2))

print(avg_weight_trace[0])
print(avg_weight_trace[-1])

print("Hebbian updates:", hebb_updates)

# PLOTS

plt.figure(figsize=(12,8))

plt.subplot(2,1,1)

plt.scatter(
    spike_times,
    spike_neurons,
    s=4
)

plt.ylabel("Neuron")
plt.title("Structured Cortical Network")

plt.subplot(2,1,2)

plt.plot(avg_weight_trace)

plt.ylabel("Average Active Weight")
plt.xlabel("Time")

plt.tight_layout()
plt.show()