import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import patterns


def make_output_folder(root="diagnostics_STDP"):
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join(root, run_name)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def init_neuron_parameters(N):
    a = np.random.normal(0.02, 0.002, N)
    b = np.random.normal(0.20, 0.01, N)
    c = np.full(N, -65.0)
    d = np.full(N, 8.0)
    v = c.copy()
    u = b * v
    return a, b, c, d, v, u


def init_connectivity(N, inhibitory_fraction=0.2):
    W = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            distance = abs(i - j)
            p_connect = np.exp(-distance / 2)

            if np.random.random() < p_connect:
                weight = (12 * np.exp(-distance / 3) + np.random.normal(0, 0.5))
                W[i, j] = max(weight, 0.5)

    num_inhibitory = int(inhibitory_fraction * N)
    inhibitory_neurons = np.arange(N - num_inhibitory, N)
    for neuron in inhibitory_neurons:
        W[neuron, :] = -np.abs(W[neuron, :])

    return W


def init_delays(N):
    delays = np.ones((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            delays[i, j] = max(1, abs(i - j))
    return delays


def init_recording(N, time_steps):
    event_queue = [[] for _ in range(time_steps + N + 50)]
    syn_current = np.zeros(N)
    last_spike = np.full(N, -10000)
    spike_times = []
    spike_neurons = []
    spike_counts = np.zeros(N)
    avg_weight_trace = []
    stdp_updates = 0
    return event_queue, syn_current, last_spike, spike_times, spike_neurons, spike_counts, avg_weight_trace, stdp_updates


def apply_sensory_input(I, N):
    num_input = max(2, int(0.1 * N))
    input_neurons = np.arange(num_input)
    I[input_neurons] += 9 + np.random.normal(0, 1, num_input)
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


def simulate(params):
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
    event_queue, syn_current, last_spike, spike_times, spike_neurons, spike_counts, avg_weight_trace, stdp_updates = init_recording(N, time_steps)
    initial_W = W.copy()

    for t in range(time_steps):
        syn_current *= decay
        I = syn_current.copy()
        I = apply_sensory_input(I, N)
        I += np.random.normal(0, 1.0, N)
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
        "time_steps": 3000,
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

    output_folder = make_output_folder()
    results = simulate(params)
    plot_results(results, output_folder)
    save_results(results, output_folder)
    print(f"Simulation complete. Results saved to {output_folder}")


if __name__ == "__main__":
    main()

