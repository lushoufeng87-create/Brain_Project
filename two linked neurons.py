#2 connected neurons → small network → learning → sensory inputs → simple behaviors

import matplotlib.pyplot as plt

# Regular spiking neuron parameters, copy paste from web
a, b, c, d = 0.02, 0.2, -65, 8

# Neuron 1 state
v1 = -65
u1 = b * v1 # initial recovery variable, u = b*v at rest, brake for action potentials

# Neuron 2 state
v2 = -65
u2 = b * v2

# Connection strength from neuron 1 to neuron 2
w12 = 10

time_steps = 1000

#store values
v1_trace = [] 
v2_trace = []

for t in range(time_steps):

    #input to neuron 1 initial
    I1 = 10

    #no input to neuron 2, it only receives from neuron 1
    I2 = 0

    # Update neuron 1
    v1 += 0.04 * v1**2 + 5*v1 + 140 - u1 + I1
    u1 += a * (b*v1 - u1)

    spike1 = False

    if v1 >= 30:
        spike1 = True
        v1_trace.append(30)
        v1 = c
        u1 += d
    else:
        v1_trace.append(v1)

    # Synapse
    if spike1:
        I2 += w12

    # Update neuron 2
    v2 += 0.04 * v2**2 + 5*v2 + 140 - u2 + I2
    u2 += a * (b*v2 - u2)

    if v2 >= 30:
        v2_trace.append(30)
        v2 = c
        u2 += d
    else:
        v2_trace.append(v2)

# Plot both neurons
plt.figure(figsize=(10, 5))
plt.plot(v1_trace, label="Neuron 1")
plt.plot(v2_trace, label="Neuron 2")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.title("Two Interacting Izhikevich Neurons")
plt.legend()
plt.show()