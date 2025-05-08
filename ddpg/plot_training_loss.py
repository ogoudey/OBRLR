"""
plot_training_loss.py

Reads TensorBoard event files and plots a chosen training‐loss scalar
versus the number of training steps.
"""

import argparse
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

def plot_loss_curve(logdir: str, tag: str):
    # Load all scalar events
    ea = event_accumulator.EventAccumulator(
        logdir,
        size_guidance={event_accumulator.SCALARS: 0},  # load all scalars
    )
    ea.Reload()

    # Check that the desired scalar tag exists
    scalar_tags = ea.Tags().get("scalars", [])
    if tag not in scalar_tags:
        raise ValueError(f"Tag '{tag}' not found. Available tags: {scalar_tags}")

    # Extract steps and values
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]

    # Plot
    plt.figure()
    plt.plot(steps, values)
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title(f"{tag} vs. Training Steps")
    plt.tight_layout()
    plt.savefig(f"{tag.replace('/', '_')}_curve.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Plot a training loss curve from TensorBoard logs"
    )
    parser.add_argument(
        "logdir",
        help="Path to directory containing event files (e.g., tensorboard/)"
    )
    parser.add_argument(
        "--tag",
        default="train/actor_loss",
        help="Scalar tag to plot (e.g., 'train/actor_loss')"
    )
    args = parser.parse_args()
    plot_loss_curve(args.logdir, args.tag)

if __name__ == "__main__":
    main()
