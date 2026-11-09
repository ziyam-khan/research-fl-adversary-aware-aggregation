import numpy as np
import torch


def fedavg(global_model, client_state_dicts, client_weights):
    global_dict = global_model.state_dict()
    total_weight = sum(client_weights)

    for key in global_dict:
        global_dict[key] = sum(
            client_state_dicts[i][key].float() * (client_weights[i] / total_weight)
            for i in range(len(client_state_dicts))
        )

    global_model.load_state_dict(global_dict)
    return global_model


def coordinate_median(global_model, client_state_dicts, client_weights):
    global_dict = global_model.state_dict()

    for key in global_dict:
        stacked = torch.stack([sd[key].float() for sd in client_state_dicts])
        global_dict[key] = torch.median(stacked, dim=0).values

    global_model.load_state_dict(global_dict)
    return global_model


def trimmed_mean(global_model, client_state_dicts, client_weights, beta=0.1):
    global_dict = global_model.state_dict()
    n = len(client_state_dicts)
    trim = max(1, int(n * beta))
    if n - 2 * trim < 1:
        trim = max(0, (n - 1) // 2)

    for key in global_dict:
        stacked = torch.stack([sd[key].float() for sd in client_state_dicts])
        sorted_vals, _ = torch.sort(stacked, dim=0)
        trimmed = sorted_vals[trim:n - trim]
        global_dict[key] = trimmed.mean(dim=0)

    global_model.load_state_dict(global_dict)
    return global_model


def aawa(global_model, client_state_dicts, client_weights, client_ids,
         participation_history, grad_norm_history, class_distributions,
         window_size=10):
    global_dict = global_model.state_dict()
    n_total_clients = len(participation_history)
    n_participating = len(client_state_dicts)

    avg_grads = np.zeros(n_total_clients)
    for cid in range(n_total_clients):
        gnorms = grad_norm_history[cid][-window_size:]
        if len(gnorms) > 0:
            avg_grads[cid] = np.mean(gnorms)
    max_grad = avg_grads.max() + 1e-8

    reliability_scores = np.ones(n_total_clients)
    for cid in range(n_total_clients):
        history = participation_history[cid][-window_size:]
        if len(history) == 0:
            continue
        participation_rate = sum(history) / len(history)
        normalized_grad = avg_grads[cid] / max_grad

        if (1 - participation_rate) > 0 and normalized_grad > 0:
            suspicion = (1 - participation_rate) * normalized_grad
        else:
            suspicion = 0.0

        reliability_scores[cid] = max(0.1, 1.0 - suspicion)

    absent_ids = set(range(n_total_clients)) - set(client_ids)
    suppressed_dist = np.zeros(class_distributions.shape[1])
    n_suppressed = 0
    for aid in absent_ids:
        if reliability_scores[aid] < 0.5:
            suppressed_dist += class_distributions[aid]
            n_suppressed += 1

    adjusted_weights = np.array(client_weights, dtype=float)
    if n_suppressed > 0:
        suppressed_dist /= n_suppressed
        for i, cid in enumerate(client_ids):
            similarity = 1.0 - 0.5 * np.sum(np.abs(
                class_distributions[cid] - suppressed_dist
            ))
            compensation = 1.0 + similarity * (n_suppressed / n_total_clients)
            adjusted_weights[i] *= compensation

    for i, cid in enumerate(client_ids):
        adjusted_weights[i] *= reliability_scores[cid]

    total = adjusted_weights.sum()
    if total > 0:
        adjusted_weights /= total

    for key in global_dict:
        global_dict[key] = sum(
            client_state_dicts[i][key].float() * adjusted_weights[i]
            for i in range(n_participating)
        )

    global_model.load_state_dict(global_dict)
    return global_model
