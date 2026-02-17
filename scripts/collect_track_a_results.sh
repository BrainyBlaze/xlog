#!/bin/bash
# Collect Track A results after all seeds complete.
# Usage: bash scripts/collect_track_a_results.sh
set -euo pipefail

RESULTS_BASE="examples/neural/results/track_a/20260216T145409Z_track_a_dev"

echo "Checking Track A 01_minimal results..."
echo ""

all_done=true
for seed in 7 42 123; do
    dir="$RESULTS_BASE/01_minimal/seed_$seed"
    if [ -f "$dir/exit_code.txt" ] && [ -s "$dir/exit_code.txt" ]; then
        code=$(cat "$dir/exit_code.txt")
        if [ "$code" = "0" ]; then
            echo "  seed $seed: COMPLETE (exit 0)"
            # Extract final loss from stdout.log
            final_loss=$(grep -oP 'Epoch 5/5: avg_loss=\K[0-9.]+' "$dir/stdout.log" 2>/dev/null || echo "N/A")
            elapsed=$(grep -oP 'ELAPSED_SEC=\K[0-9.]+' "$dir/time.txt" 2>/dev/null || echo "N/A")
            echo "    final_loss=$final_loss  elapsed=${elapsed}s"
        else
            echo "  seed $seed: FAILED (exit $code)"
            all_done=false
        fi
    else
        echo "  seed $seed: PENDING"
        all_done=false
    fi
done

echo ""

if [ "$all_done" = false ]; then
    echo "Not all seeds complete. Check 'ps aux | grep train.py' for running processes."
    exit 1
fi

echo "All seeds complete! Creating metrics..."

# Create per-seed metrics.json files
for seed in 42 123; do
    dir="$RESULTS_BASE/01_minimal/seed_$seed"
    if [ -f "$dir/metrics.json" ]; then
        echo "  seed $seed: metrics.json already exists, skipping"
        continue
    fi

    # Extract data
    elapsed=$(grep -oP 'ELAPSED_SEC=\K[0-9.]+' "$dir/time.txt")

    # Extract all epoch losses
    losses=$(grep -oP 'avg_loss=\K[0-9.]+' "$dir/stdout.log" | paste -sd',' -)
    initial=$(grep -oP 'avg_loss=\K[0-9.]+' "$dir/stdout.log" | head -1)
    final=$(grep -oP 'avg_loss=\K[0-9.]+' "$dir/stdout.log" | tail -1)

    python3 -c "
import json, sys
losses = [$losses]
data = {
    'seed': $seed,
    'example': '01_minimal',
    'engine': 'xlog',
    'training_function': 'train_model_tensor',
    'train_limit': 512,
    'num_pairs': 256,
    'epochs': 5,
    'batch_size': 64,
    'learning_rate': 0.001,
    'device': 'cuda',
    'exit_code': 0,
    'elapsed_sec': $elapsed,
    'epoch_losses': losses,
    'initial_loss': losses[0],
    'final_loss': losses[-1],
    'loss_improvement_pct': round((1 - losses[-1]/losses[0]) * 100, 1),
    'model_saved': 'mnist_net.pt'
}
print(json.dumps(data, indent=2))
" > "$dir/metrics.json"
    echo "  seed $seed: metrics.json created"
done

echo ""
echo "Updating mnist_vs_deepproblog.json with n=3 stats..."

python3 -c "
import json, math

base = '$RESULTS_BASE'
seeds = [7, 42, 123]
final_losses = []
times = []

for s in seeds:
    with open(f'{base}/01_minimal/seed_{s}/metrics.json') as f:
        m = json.load(f)
    final_losses.append(m['final_loss'])
    times.append(m['elapsed_sec'])

mean_loss = sum(final_losses) / len(final_losses)
std_loss = math.sqrt(sum((l - mean_loss)**2 for l in final_losses) / (len(final_losses) - 1))
mean_time = sum(times) / len(times)
std_time = math.sqrt(sum((t - mean_time)**2 for t in times) / (len(times) - 1))

with open(f'{base}/comparisons/mnist_vs_deepproblog.json') as f:
    data = json.load(f)

data['status'] = 'complete'
data['note'] = 'All 3 seeds complete.'
data['xlog_track_a']['n'] = 3
data['xlog_track_a']['seeds_complete'] = seeds
data['xlog_track_a']['seeds_pending'] = []
data['xlog_track_a']['final_loss_mean'] = round(mean_loss, 6)
data['xlog_track_a']['final_loss_std'] = round(std_loss, 6)
data['xlog_track_a']['training_time_sec_mean'] = round(mean_time, 2)
data['xlog_track_a']['training_time_sec_std'] = round(std_time, 2)

for s in seeds:
    with open(f'{base}/01_minimal/seed_{s}/metrics.json') as f:
        m = json.load(f)
    data['xlog_track_a'][f'seed_{s}'] = {
        'final_loss': m['final_loss'],
        'initial_loss': m['initial_loss'],
        'epoch_losses': m['epoch_losses'],
        'training_time_sec': m['elapsed_sec']
    }

with open(f'{base}/comparisons/mnist_vs_deepproblog.json', 'w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')

print(f'  final_loss: {mean_loss:.6f} +/- {std_loss:.6f}')
print(f'  training_time: {mean_time:.1f}s +/- {std_time:.1f}s')
print('  Written to mnist_vs_deepproblog.json')
"

echo ""
echo "Done! Commit with:"
echo "  git add examples/neural/results/"
echo "  git commit -m 'evidence: Track A all seeds complete (01_minimal)'"
