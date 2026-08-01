# small network → learning → sensory inputs → simple behaviors
# 
#  3 neurons, 1 inhibitory
# Neuron 1 excites Neuron 2
# Neuron 3 inhibits Neuron 2
# Both Neuron 1 and Neuron 3 receive external input

import matplotlib.pyplot as plt

# parameters
a, b, c, d = 0.02, 0.2, -65, 8


# neuron states

# neuron 1 (excitatory)
v1 = -65
u1 = b * v1 # fatigue for neurons, increases when neuron spikes, mirrors biological behavior

# neuron 2 (target)
v2 = -65
u2 = b * v2

# neuron 3 (inhibitory)
v3 = -65
u3 = b * v3

# synapses

# N1 to N2 (excitatory)
w12 = 4 # each spike of neuron 1 produces 4 units of excitatory current in neuron 2
syn12 = 0 # stores the current state of the synapse

# N3 to N2 (inhibitory)
w32 = -6
syn32 = 0

decay = 0.95 #clearance of neutransmitters inside the synpatic cleft


# inputs

I1_const = 5
I3_const = 8


# record synapses

time_steps = 1000

v1_trace = []
v2_trace = []
v3_trace = []

syn12_trace = []
syn32_trace = []

# simulation

for t in range(time_steps):

    # synapses decay
    syn12 *= decay
    syn32 *= decay

    # external currents
    I1 = I1_const
    I3 = I3_const

    # target neuron receives input only from other neurons, no external input
    I2 = syn12 + syn32

    # neuron 1

    v1 += 0.04 * v1**2 + 5 * v1 + 140 - u1 + I1 # differential equation for Izhikevich neuron model
    u1 += a * (b * v1 - u1)

    spike1 = False

    if v1 >= 30:
        spike1 = True

        v1_trace.append(30)

        v1 = c
        u1 += d
    else:
        v1_trace.append(v1)

    if spike1:
        syn12 += w12

    # neuron 2

    v2 += 0.04 * v2**2 + 5 * v2 + 140 - u2 + I2
    u2 += a * (b * v2 - u2)

    if v2 >= 30:
        v2_trace.append(30)

        v2 = c
        u2 += d
    else:
        v2_trace.append(v2)


    # neuron 3

    v3 += 0.04 * v3**2 + 5 * v3 + 140 - u3 + I3 
    u3 += a * (b * v3 - u3)

    spike3 = False

    if v3 >= 30:
        spike3 = True

        v3_trace.append(30)

        v3 = c
        u3 += d
    else:
        v3_trace.append(v3)

    if spike3:
        syn32 += w32


    # Record synaptic currents
    syn12_trace.append(syn12)
    syn32_trace.append(syn32)

# plotting

plt.figure(figsize=(12, 8))

# neurons
plt.subplot(2, 1, 1)

plt.plot(v1_trace, label="Neuron 1 (Excitatory)")
plt.plot(v2_trace, label="Neuron 2 (Target)")
plt.plot(v3_trace, label="Neuron 3 (Inhibitory)")

plt.ylabel("Membrane Potential (mV)")
plt.title("Three-Neuron Network")
plt.legend()

# synapses
plt.subplot(2, 1, 2)

plt.plot(syn12_trace, label="Excitatory Current N1→N2")
plt.plot(syn32_trace, label="Inhibitory Current N3→N2")

plt.ylabel("Synaptic Current")
plt.xlabel("Time Step")
plt.legend()

plt.tight_layout()
plt.show()
