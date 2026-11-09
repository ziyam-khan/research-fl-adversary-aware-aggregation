import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn

from data import get_cifar10, dirichlet_split, make_client_loaders, get_client_class_distribution
from model import CIFAR10Net
from attacks import NoAttack, RandomDropout, StrategicGradient, StrategicDistribution, CombinedStrategic
from aggregators import fedavg, coordinate_median, trimmed_mean, aawa

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLIENTS = 20
ALPHA = 0.1
ROUNDS = 100
LOCAL_EPOCHS = 3
LR = 0.01
BATCH_SIZE = 64
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def train_client(model, loader, epochs, lr):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

    grad_norm = sum(
        p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None
    ) ** 0.5

    return model, total_loss / max(total_samples, 1), grad_norm


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def run_fl(aggregator_name, aggregator_fn, attack, train_set, test_loader,
           client_indices, class_dists, num_rounds=None):
    if num_rounds is None:
        num_rounds = ROUNDS
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    global_model = CIFAR10Net().to(DEVICE)
    client_loaders = make_client_loaders(train_set, client_indices, BATCH_SIZE)
    n_clients = len(client_indices)

    participation_history = {i: [] for i in range(n_clients)}
    grad_norm_history = {i: [] for i in range(n_clients)}
    grad_norms_current = np.zeros(n_clients)

    accuracies = []
    losses = []

    for rnd in range(num_rounds):
        client_info = {
            "grad_norms": grad_norms_current.copy(),
            "class_distributions": class_dists,
        }
        victims = attack.select_victims(rnd, n_clients, client_info)
        participants = [i for i in range(n_clients) if i not in victims]

        for i in range(n_clients):
            participation_history[i].append(i in participants)

        client_state_dicts = []
        client_weights = []
        client_ids = []

        local_model = CIFAR10Net().to(DEVICE)
        for cid in participants:
            local_model.load_state_dict(global_model.state_dict())
            local_model.zero_grad(set_to_none=True)
            local_model, loss, gnorm = train_client(
                local_model, client_loaders[cid], LOCAL_EPOCHS, LR
            )
            client_state_dicts.append({k: v.clone() for k, v in local_model.state_dict().items()})
            client_weights.append(len(client_indices[cid]))
            client_ids.append(cid)
            grad_norms_current[cid] = gnorm
            grad_norm_history[cid].append(gnorm)

        if len(client_state_dicts) == 0:
            acc = evaluate(global_model, test_loader)
            accuracies.append(acc)
            losses.append(0.0)
            continue

        if aggregator_name == "aawa":
            global_model = aggregator_fn(
                global_model, client_state_dicts, client_weights, client_ids,
                participation_history, grad_norm_history, class_dists
            )
        else:
            global_model = aggregator_fn(
                global_model, client_state_dicts, client_weights
            )

        acc = evaluate(global_model, test_loader)
        accuracies.append(acc)

        if (rnd + 1) % 10 == 0:
            print(f"  [{aggregator_name}] Round {rnd+1}/{num_rounds} — Acc: {acc:.4f}")

    return accuracies


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading CIFAR-10...")
    train_set, test_set = get_cifar10(
        os.path.join(os.path.dirname(__file__), "..", "data")
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)

    print(f"Splitting data: {NUM_CLIENTS} clients, Dirichlet α={ALPHA}")
    client_indices = dirichlet_split(train_set, NUM_CLIENTS, ALPHA, SEED)
    class_dists = get_client_class_distribution(train_set, client_indices)

    attacks = {
        "no_attack": NoAttack(),
        "random_30": RandomDropout(fraction=0.3, seed=SEED),
        "combined_strategic": CombinedStrategic(fraction=0.3),
    }

    aggregators = {
        "fedavg": fedavg,
        "median": coordinate_median,
        "trimmed_mean": trimmed_mean,
        "aawa": aawa,
    }

    results_path = os.path.join(RESULTS_DIR, "results.json")

    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"Loaded existing results: {[(a, list(aggs.keys())) for a, aggs in all_results.items()]}")
    else:
        all_results = {}

    for atk_name, attack in attacks.items():
        if atk_name not in all_results:
            all_results[atk_name] = {}

        for agg_name, agg_fn in aggregators.items():
            if agg_name in all_results[atk_name]:
                print(f"  [SKIP] {atk_name} × {agg_name} — already done")
                continue

            print(f"\n{'='*60}")
            print(f"Attack: {atk_name} | Aggregator: {agg_name}")
            print(f"{'='*60}")
            accs = run_fl(
                agg_name, agg_fn, attack, train_set, test_loader,
                client_indices, class_dists
            )
            all_results[atk_name][agg_name] = accs

            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  >> Saved ({atk_name} × {agg_name}) to {results_path}")

    print(f"\nAll main experiments complete. Starting sweep...")

    # --- Suppression fraction sweep (combined strategic attack) ---
    SWEEP_ROUNDS = 50
    sweep_fractions = [0.1, 0.3, 0.5]
    sweep_path = os.path.join(RESULTS_DIR, "sweep_results.json")

    if os.path.exists(sweep_path):
        with open(sweep_path) as f:
            sweep_results = json.load(f)
        print(f"Loaded existing sweep results")
    else:
        sweep_results = {}

    for agg_name, agg_fn in aggregators.items():
        if agg_name not in sweep_results:
            sweep_results[agg_name] = {}

        for frac in sweep_fractions:
            frac_key = str(frac)
            if frac_key in sweep_results[agg_name]:
                print(f"  [SKIP] {agg_name} × {frac_key} — already done")
                continue

            print(f"\n--- {agg_name} | suppression={frac:.0%} ---")
            attack = CombinedStrategic(fraction=frac)
            accs = run_fl(
                agg_name, agg_fn, attack, train_set, test_loader,
                client_indices, class_dists, num_rounds=SWEEP_ROUNDS
            )
            sweep_results[agg_name][frac_key] = accs[-1]

            with open(sweep_path, "w") as f:
                json.dump(sweep_results, f, indent=2)
            print(f"  >> Saved ({agg_name} × {frac_key}) to {sweep_path}")

    print(f"\nAll experiments complete.")


if __name__ == "__main__":
    main()
