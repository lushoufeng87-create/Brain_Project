# changed inhibitory neurons to a percentage population
# changed sensory input to a population of neurons
# enlarged event queue to avoid index errors

import numpy as np
import matplotlib.pyplot as plt

import os
from datetime import datetime

# PARAMETERS

N = 100

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
            weight = (12 * np.exp(-distance / 3) + # 3 is a characteristic distance for weight decay
                      np.random.normal(0, 0.5)) # random noise added to weight

            weight = max(weight, 0.5)

            W[i, j] = weight

# INHIBITORY NEURONS (20% OF NETWORK)

num_inhibitory = int(0.2 * N)

inhibitory_neurons = np.arange(N - num_inhibitory, N)

for neuron in inhibitory_neurons:
    W[neuron, :] = -np.abs(W[neuron, :])

# DELAYS

delays = np.ones((N, N), dtype=int)

for i in range(N):
    for j in range(N):
        if i == j:
            continue

        distance = abs(i - j)
        delays[i, j] = max(1, distance)

# EVENT QUEUE

event_queue = [[] for _ in range(time_steps + N + 50)]

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

# OUTPUT FOLDER

run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

output_folder = os.path.join("diagnostics_STDP", run_name)

os.makedirs(output_folder, exist_ok=True)

# SIMULATION

for t in range(time_steps):

    syn_current *= decay

    I = syn_current.copy()

    # sensory input (population)

    num_input = max(2, int(0.1 * N))
    input_neurons = np.arange(num_input)
    I[input_neurons] += (9 + np.random.normal(0, 1, num_input ))

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

        dv = (0.04 * v[i]**2 + 5 * v[i] + 140 - u[i] + I[i])
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

                dW = (A_plus * np.exp(-dt_spike / tau_plus))

                W[source, neuron] += dW

                W[source, neuron] = min(W[source, neuron], max_weight)

                stdp_updates += 1

        # weaken anti-causal outputs

        for target in range(N):

            if W[neuron, target] <= 0:
                continue

            dt_spike = t - last_spike[target]

            if 0 < dt_spike < 50:

                dW = (A_minus * np.exp(-dt_spike / tau_minus))

                W[neuron, target] -= dW

                W[neuron, target] = max(W[neuron, target], min_weight)

                stdp_updates += 1

        last_spike[neuron] = t

    # propagate spikes

    for source in spikes:

        for target in range(N):

            if W[source, target] == 0:
                continue

            arrival = t + delays[source, target]

            if arrival < len(event_queue):

                event_queue[arrival].append(( target, W[source, target] ))

    avg_weight_trace.append(np.mean(np.abs(W[W != 0])))


# Plots

# Weight matrices
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(initial_W,aspect='auto'); plt.title("Initial Weights"); plt.xlabel("Target"); plt.ylabel("Source"); plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(W,aspect='auto'); plt.title("Final Weights"); plt.xlabel("Target"); plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(W-initial_W,aspect='auto'); plt.title("STDP Change"); plt.xlabel("Target"); plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(output_folder,"weight_matrices.png"),dpi=300)

# Activity
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.scatter(spike_times,np.array(spike_neurons)+1,s=4)
plt.yticks(range(1,N+1))
plt.ylabel("Neuron")
plt.title("Spike Raster")
plt.subplot(2,1,2)
plt.plot(avg_weight_trace)
plt.ylabel("Average Weight")
plt.xlabel("Time")
plt.tight_layout()
plt.savefig(os.path.join(output_folder,"activity.png"),dpi=300)

# Network strength
incoming=np.sum(np.abs(W),axis=0)
outgoing=np.sum(np.abs(W),axis=1)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.bar(range(N),incoming)
plt.title("Incoming Strength")
plt.subplot(1,2,2)
plt.bar(range(N),outgoing)
plt.title("Outgoing Strength")
plt.tight_layout()
plt.savefig(os.path.join(output_folder,"strength.png"),dpi=300)
plt.show()

# Save data
np.savetxt(os.path.join(output_folder,"initial_weights.csv"),initial_W,delimiter=",",fmt="%.4f")
np.savetxt(os.path.join(output_folder,"final_weights.csv"),W,delimiter=",",fmt="%.4f")
np.savetxt(os.path.join(output_folder,"weight_change.csv"),W-initial_W,delimiter=",",fmt="%.4f")
np.savetxt(os.path.join(output_folder,"spike_counts.csv"),spike_counts,fmt="%d")
np.savetxt(os.path.join(output_folder,"incoming_strength.csv"),incoming,fmt="%.4f")
np.savetxt(os.path.join(output_folder,"outgoing_strength.csv"),outgoing,fmt="%.4f")

sim_sec=(time_steps*dt)/1000
with open(os.path.join(output_folder,"report.txt"),"w") as f:
    f.write("Simulation Report\n\n")
    f.write(f"STDP Updates: {stdp_updates}\n")
    f.write(f"Largest Weight Change: {float(np.max(np.abs(W-initial_W))):.4f}\n")
    f.write(f"Average Final Weight: {float(np.mean(np.abs(W[W!=0]))):.4f}\n\n")
    f.write("Spike Counts and Firing Rates\n")
    for i in range(N):
        f.write(f"Neuron {i}: {int(spike_counts[i])} spikes, {spike_counts[i]/sim_sec:.2f} Hz\n")
    f.write("\nTop 20 Strongest Connections\n")
    flat=np.argsort(np.abs(W).ravel())[::-1]
    c=0
    for idx in flat:
        s,t=np.unravel_index(idx,W.shape)
        if s==t: continue
        f.write(f"{s}->{t}: {W[s,t]:.3f}\n")
        c+=1
        if c==20: break

print(f"Simulation complete. Results saved to {output_folder}")
