#synaptic decay in two neurons -> small network → learning → sensory inputs → simple behaviors

import matplotlib.pyplot as plt

# Regular spiking neuron parameters
a, b, c, d = 0.02, 0.2, -65, 8

#neuron 1 state
v1 = -65
u1 = b * v1

#neuron 2 state
v2 = -65
u2 = b * v2

#synapse strength transmitted (neuron 1 to neuron 2), changing this changes the strength of each transmission
w12 = 4

#synaptic current
syn12 = 0

#how quickly neurotransmitter effect fades, changing this changes how long the effect of a spike lasts on the second neuron
decay = 0.98

time_steps = 2000

v1_trace = []
v2_trace = []
syn_trace = []

for t in range(time_steps):

    #external input to Neuron 1, changing this changes the rate at which the first neuron is stimulated
    I1 = 5

    syn12 *= decay

    #neuron 2 receives current from synapse
    I2 = syn12

    #update Neuron 1
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

    #if Neuron 1 spikes, release neurotransmitter
    if spike1:
        syn12 += w12

    #update Neuron 2
    v2 += 0.04 * v2**2 + 5*v2 + 140 - u2 + I2
    u2 += a * (b*v2 - u2)

    if v2 >= 30:
        v2_trace.append(30)

        v2 = c
        u2 += d
    else:
        v2_trace.append(v2)

    syn_trace.append(syn12)

#plot neuron voltages, AI helped
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(v1_trace, label="Neuron 1")
plt.plot(v2_trace, label="Neuron 2")
plt.ylabel("Membrane Potential (mV)")
plt.legend()

#horizontal line for reference of decay current
plt.subplot(2,1,2)
plt.plot(syn_trace)
plt.axhline(w12, linestyle='--')

plt.subplot(2, 1, 2)
plt.plot(syn_trace)
plt.ylabel("Synaptic Current")
plt.xlabel("Time Step")

plt.tight_layout()
plt.show()

#initial two fire currents show a very simple version of neurons attempting to reach firing equilibrium
#the second current has not decayed to 0 fully yet, so the second neuron reaches a higher current