import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

PLOT_COLORS = {
    "BRIE": "blue",
    "BRIE+SUM": "green",
    "RANDOM": "orange",
    "CNT": "red",
}


def percentile_figure(data: dict):
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax2 = ax1.twinx()

    plt.rcParams.update({"font.size": 17})

    # LEFT AXIS (percentile curves)
    for metrics in data["metrics"]:
        ax1.plot(
            metrics["min_photos"],
            metrics["median_percentile"],
            linewidth=3.0,
            label=metrics["model_name"],
            alpha=0.8,
            color=PLOT_COLORS.get(metrics["model_name"], None),
        )

    ax1.set_xlabel("Users with ≥x train images")
    ax1.set_ylabel("Median percentile of author's image")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 1)
    ax1.grid(True, linestyle="--", color="lightgray")

    # RIGHT AXIS (test cases)
    for metrics in data["metrics"]:
        ax2.plot(
            metrics["min_photos"],
            metrics["num_test_cases"],
            linewidth=2.5,
            color="black",
            alpha=0.3,
        )

    ax2.set_ylabel("Test cases")
    ax2.set_yscale("log")

    ax1.set_title(data["city"])

    ax1.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(
        f'figures/{data["city"]}/percentile_{data["city"]}.pdf',
        bbox_inches="tight"
    )
    plt.close()


def retrieval_figure(data: dict, metric_name: str):
    rcParams["figure.figsize"] = 4, 4

    plt.title(metric_name)

    for metrics in data["metrics"]:
        plt.plot(
            metrics["k"],
            metrics[metric_name],
            linewidth=2.0,
            label=metrics["model_name"],
        )

    plt.xlabel("Position k of ranking")
    plt.ylabel(f"{metric_name} at k")

    plt.xlim(1, 10)

    # Make the lim a bit larger than the max value
    plt.ylim(0, max([max(metrics[metric_name]) for metrics in data["metrics"]]) * 1.1)

    plt.xticks(range(1, 11, 1))

    plt.grid(True, linestyle="--", color="lightgray")
    plt.legend(loc="upper left")
    plt.tight_layout()

    # Output
    plt.savefig(f'docs/{data["city"]}/{metric_name}.pdf', bbox_inches="tight")
    plt.show()

def bleu_figure(data: dict):
    plt.figure(figsize=(9, 5))

    xmax = 0

    for metrics in data["metrics"]:
        x = np.array(metrics["min_photos"])
        y = np.array(metrics["mean_bleu"])

        mask = ~np.isnan(y)

        x_valid = x[mask]
        y_valid = y[mask]

        if len(x_valid) == 0:
            continue

        plt.plot(
            x_valid,
            y_valid,
            label=metrics["model_name"],
            linewidth=3
        )

        xmax = max(xmax, x_valid.max())

    plt.xlabel("Users with ≥x train images")
    plt.ylabel("Mean BLEU")

    if xmax > 0:
        plt.xlim(0, xmax + 2)
    else:
        plt.xlim(0, 100)

    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.title(data["city"])
    plt.tight_layout()

    plt.savefig(f'figures/{data["city"]}/bleu_{data["city"]}.pdf')
    plt.close()

def rouge_figure(data: dict):
    plt.figure(figsize=(9, 5))

    for metrics in data["metrics"]:
        x = np.array(metrics["min_photos"])
        y = np.array(metrics["mean_rouge"])

        mask = ~np.isnan(y)

        plt.plot(
            x[mask],
            y[mask],
            label=metrics["model_name"],
            linewidth=3
        )

    plt.xlabel("Users with ≥x train images")
    plt.ylabel("Mean ROUGE")
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.title(data["city"])
    plt.tight_layout()

    plt.savefig(f'figures/{data["city"]}/rouge_{data["city"]}.pdf')
    plt.close()

def generic_metric_figure(data: dict, metric_key: str, ylabel: str, filename: str):
    plt.figure(figsize=(9, 5))

    for metrics in data["metrics"]:
        x = np.array(metrics["min_photos"])
        y = np.array(metrics[metric_key], dtype=np.float32)

        mask = ~np.isnan(y.astype(np.float32))

        plt.plot(
            x[mask],
            y[mask],
            label=metrics["model_name"],
            linewidth=3
        )

    plt.xlabel("Users with ≥x train images")
    plt.ylabel(ylabel)
    ymax = max([
        np.nanmax(metrics[metric_key])
        for metrics in data["metrics"]
        if len(metrics[metric_key]) > 0
    ])

    plt.ylim(0, ymax * 1.1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.title(data["city"])
    plt.tight_layout()

    plt.savefig(f'figures/{data["city"]}/{filename}.pdf')
    plt.close()