import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from patterns import make_patterns, apply_pattern, random_training_schedule

# storage folder
def make_output_folder(root="diagnostics_STDP"): # creates/checks for folder to save results
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") #strftime converts current date into string
    output_folder = os.path.join(root, run_name) # os.path.join joins proj names together
    os.makedirs(output_folder, exist_ok=True) # creates new folder, exist_ok=True means if folder already exists, don't throw default error
    return output_folder # returns path to created folder as a string. 
# ensures no files are overwritten and each run is saved in a unique folder.

# neuron parameters
def init_neuron_parameters(N):
    a = np.random.normal(0.02, 0.002, N)
    b = np.random.normal(0.20, 0.01, N)
    c = np.full(N, -65.0)
    d = np.full(N, 8.0)
    v = c.copy()
    u = b * v
    return a, b, c, d, v, u

# distance based connectivity
def init_connectivity(N, inhibitory_fraction=0.2): # specified parameters
    W = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i == j:
                continue # skips self-connections
            distance = abs(i - j) # abs = absolute value
            p_connect = np.exp(-distance / 2) # nearby neurons are more likely to connect

            if np.random.random() < p_connect: # generates biased random connectivity based on distance
                weight = (12 * np.exp(-distance / 3) + np.random.normal(0, 0.5)) # weight decreases with distance, plus some noise
                W[i, j] = max(weight, 0.5) # ensures minimum weight of 0.5 so connections dont die down

    num_inhibitory = int(inhibitory_fraction * N) # converts fraction to number of inhibitory neurons
    inhibitory_neurons = np.arange(N - num_inhibitory, N) # .arange creates integers for inhibitory neurons
    for neuron in inhibitory_neurons:
        W[neuron, :] = -np.abs(W[neuron, :]) # makes inhibitory neurons have negative weights to all targets
        #[neuron, :] selects the entire column, and -np.abs(...) ensures all weights going out from inhibitory neurons are negative.

    return W # returns full network weight matrix

# synaptic delays
def init_delays(N):
    delays = np.ones((N, N), dtype=int) # datatype integers instead of default floating points 1.0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue # no self-connections, skip them 
            delays[i, j] = max(1, abs(i - j)) # minimum delay of 1 timestep, increases with distance between neurons
    return delays

# recording variables
def init_recording(N, time_steps):
    event_queue = [[] for _ in range(time_steps + N + 100)] # list comprehension, creates list of empty lists
    syn_current = np.zeros(N) # initializes synaptic current for each neuron to zero
    last_spike = np.full(N, -696969) # initializes last spike times to a large negative value, ensuring no neuron has spiked yet
    spike_times = []
    spike_neurons = []
    spike_counts = np.zeros(N)
    avg_weight_trace = []
    stdp_updates = 0
    pattern_trace = []
    return event_queue, syn_current, last_spike, spike_times, spike_neurons, spike_counts, avg_weight_trace, stdp_updates, pattern_trace


def apply_sensory_input(I, pattern, amplitude=9.0, noise_std=1.0):
    I = apply_pattern(I, pattern, amplitude=amplitude, noise_std=noise_std)
    return I


def apply_homeostatic_plasticity(I, last_spike, t):
    for i in range(len(I)):
        silent_time = t - last_spike[i]
        if silent_time > 200:
            I[i] += 0.005 * (silent_time - 200)
    return I


def update_neurons(N, v, u, a, b, c, d, I, dt, t, spike_times, spike_neurons, spike_counts):
    spikes = []
    for i in range(N):
        dv = 0.04 * v[i] ** 2 + 5 * v[i] + 140 - u[i] + I[i]
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

    return spikes


def apply_stdp(spikes, W, last_spike, t, A_plus, A_minus, tau_plus, tau_minus, max_weight, min_weight):
    stdp_updates = 0

    for neuron in spikes:
        for source in range(W.shape[0]):
            if W[source, neuron] <= 0:
                continue
            dt_spike = t - last_spike[source]
            if 0 < dt_spike < 50:
                dW = A_plus * np.exp(-dt_spike / tau_plus)
                W[source, neuron] += dW
                W[source, neuron] = min(W[source, neuron], max_weight)
                stdp_updates += 1

        for target in range(W.shape[1]):
            if W[neuron, target] <= 0:
                continue
            dt_spike = t - last_spike[target]
            if 0 < dt_spike < 50:
                dW = A_minus * np.exp(-dt_spike / tau_minus)
                W[neuron, target] -= dW
                W[neuron, target] = max(W[neuron, target], min_weight)
                stdp_updates += 1

        last_spike[neuron] = t

    return stdp_updates


def propagate_spikes(spikes, W, delays, event_queue, t):
    for source in spikes:
        for target in range(W.shape[1]):
            if W[source, target] == 0:
                continue
            arrival = t + delays[source, target]
            if arrival < len(event_queue):
                event_queue[arrival].append((target, W[source, target]))


def simulate(params, training_patterns):
    N = params["N"]
    time_steps = params["time_steps"]
    dt = params["dt"]
    decay = params["decay"]
    A_plus = params["A_plus"]
    A_minus = params["A_minus"]
    tau_plus = params["tau_plus"]
    tau_minus = params["tau_minus"]
    max_weight = params["max_weight"]
    min_weight = params["min_weight"]

    a, b, c, d, v, u = init_neuron_parameters(N)
    W = init_connectivity(N, params.get("inhibitory_fraction", 0.2))
    delays = init_delays(N)
    event_queue, syn_current, last_spike, spike_times, spike_neurons, spike_counts, avg_weight_trace, stdp_updates, pattern_trace = init_recording(N, time_steps)
    initial_W = W.copy()

    pattern_names = ["A", "B", "C", "D"]
    on_duration = 500
    off_duration = 100
    cycle_duration = (on_duration + off_duration) * len(pattern_names)  # 2400 timesteps

# main() 
    for t in range(time_steps):
        syn_current *= decay
        I = syn_current.copy()
        
        # Determine current pattern and whether it's active
        cycle_position = t % cycle_duration
        pattern_index = cycle_position // (on_duration + off_duration)
        position_in_pattern = cycle_position % (on_duration + off_duration)
        
        pattern_name = pattern_names[pattern_index]
        pattern_trace.append(pattern_name)
        
        # Apply sensory input only if pattern is in "on" state
        if position_in_pattern < on_duration:
            I = apply_sensory_input(I, training_patterns[pattern_name])
        
        I += np.random.normal(0, 0.3, N)
        I = apply_homeostatic_plasticity(I, last_spike, t)

        for target, current in event_queue[t]:
            syn_current[target] += current

        spikes = update_neurons(N, v, u, a, b, c, d, I, dt, t, spike_times, spike_neurons, spike_counts)
        stdp_updates += apply_stdp(spikes, W, last_spike, t, A_plus, A_minus, tau_plus, tau_minus, max_weight, min_weight)
        propagate_spikes(spikes, W, delays, event_queue, t)
        avg_weight_trace.append(np.mean(np.abs(W[W != 0])))

    return {
        "params": params,
        "W": W,
        "initial_W": initial_W,
        "spike_times": spike_times,
        "spike_neurons": spike_neurons,
        "spike_counts": spike_counts,
        "avg_weight_trace": avg_weight_trace,
        "stdp_updates": stdp_updates,
        "pattern_trace": pattern_trace,
    }


def plot_results(results, output_folder):
    W = results["W"]
    initial_W = results["initial_W"]
    spike_times = results["spike_times"]
    spike_neurons = results["spike_neurons"]
    avg_weight_trace = results["avg_weight_trace"]
    N = results["params"]["N"]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(initial_W, aspect="auto")
    plt.title("Initial Weights")
    plt.xlabel("Target")
    plt.ylabel("Source")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(W, aspect="auto")
    plt.title("Final Weights")
    plt.xlabel("Target")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(W - initial_W, aspect="auto")
    plt.title("STDP Change")
    plt.xlabel("Target")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "weight_matrices.png"), dpi=300)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.scatter(spike_times, np.array(spike_neurons) + 1, s=4)
    plt.yticks(range(1, N + 1))
    plt.ylabel("Neuron")
    plt.title("Spike Raster")

    plt.subplot(2, 1, 2)
    plt.plot(avg_weight_trace)
    plt.ylabel("Average Weight")
    plt.xlabel("Time")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "activity.png"), dpi=300)

    incoming = np.sum(np.abs(W), axis=0)
    outgoing = np.sum(np.abs(W), axis=1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.bar(range(N), incoming)
    plt.title("Incoming Strength")
    plt.subplot(1, 2, 2)
    plt.bar(range(N), outgoing)
    plt.title("Outgoing Strength")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "strength.png"), dpi=300)
    plt.close("all")


def save_results(results, output_folder):
    W = results["W"]
    initial_W = results["initial_W"]
    spike_counts = results["spike_counts"]
    params = results["params"]

    incoming = np.sum(np.abs(W), axis=0)
    outgoing = np.sum(np.abs(W), axis=1)

    np.savetxt(os.path.join(output_folder, "initial_weights.csv"), initial_W, delimiter=",", fmt="%.4f")
    np.savetxt(os.path.join(output_folder, "final_weights.csv"), W, delimiter=",", fmt="%.4f")
    np.savetxt(os.path.join(output_folder, "weight_change.csv"), W - initial_W, delimiter=",", fmt="%.4f")
    np.savetxt(os.path.join(output_folder, "spike_counts.csv"), spike_counts, fmt="%d")
    np.savetxt(os.path.join(output_folder, "incoming_strength.csv"), incoming, fmt="%.4f")
    np.savetxt(os.path.join(output_folder, "outgoing_strength.csv"), outgoing, fmt="%.4f")

    sim_sec = (params["time_steps"] * params["dt"]) / 1000
    with open(os.path.join(output_folder, "report.txt"), "w") as f:
        f.write("Simulation Report\n\n")
        f.write(f"STDP Updates: {results['stdp_updates']}\n")
        f.write(f"Largest Weight Change: {float(np.max(np.abs(W - initial_W))):.4f}\n")
        f.write(f"Average Final Weight: {float(np.mean(np.abs(W[W != 0]))):.4f}\n\n")
        f.write("Spike Counts and Firing Rates\n")
        for i in range(params["N"]):
            f.write(f"Neuron {i}: {int(spike_counts[i])} spikes, {spike_counts[i] / sim_sec:.2f} Hz\n")
        f.write("\nTop 20 Strongest Connections\n")

        flat = np.argsort(np.abs(W).ravel())[::-1]
        c = 0
        for idx in flat:
            s, t = np.unravel_index(idx, W.shape)
            if s == t:
                continue
            f.write(f"{s}->{t}: {W[s,t]:.3f}\n")
            c += 1
            if c == 20:
                break


def main():
    params = {
        "N": 100,
        "time_steps": 4800,
        "dt": 0.5,
        "decay": 0.97,
        "A_plus": 0.025,
        "A_minus": 0.02,
        "tau_plus": 20,
        "tau_minus": 20,
        "max_weight": 10,
        "min_weight": 0.3,
        "inhibitory_fraction": 0.2,
    }

    # Initialize pattern-based sensory input
    N = params["N"]
    training_patterns = make_patterns(N, input_fraction=0.1)

    output_folder = make_output_folder()
    results = simulate(params, training_patterns)
    plot_results(results, output_folder)
    save_results(results, output_folder)
    print(f"Simulation complete. Results saved to {output_folder}")


if __name__ == "__main__":
    main()

