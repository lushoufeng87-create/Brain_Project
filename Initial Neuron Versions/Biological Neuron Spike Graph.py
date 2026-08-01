import matplotlib.pyplot as plt

# Izhikevich Parameters for regular spiking (RS) neuron
a = 0.02; b = 0.2; c = -65; d = 8
v = -65  # Membrane potential
u = b * v # Membrane recovery variable
I = 10   # Input current

time_steps = 1000
voltage_trace = []

for t in range(time_steps):
    # differential equations
    v += 0.04 * v**2 + 5 * v + 140 - u + I
    u += a * (b * v - u)
    
    # If a spike occurs
    if v >= 30:
        voltage_trace.append(30)
        v = c
        u += d
    else:
        voltage_trace.append(v)

plt.plot(voltage_trace)
plt.title("Simulated Neuron Action Potential")
plt.ylabel("Membrane Potential (mV)")
plt.xlabel("Time (ms)")
plt.show()
