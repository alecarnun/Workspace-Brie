import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

    ax1.set_xlabel("Users with ≥x train reviews")
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
    fig, ax1 = plt.subplots(figsize=(9, 5))

    plt.rcParams.update({'font.size': 14})

    ax2 = ax1.twinx()

    xmax = 0

    # LEFT AXIS (BLEU curves)
    for metrics in data["metrics"]:
        x = np.array(metrics["min_photos"])
        y = np.array(metrics["mean_bleu"])

        mask = ~np.isnan(y)

        x_valid = x[mask]
        y_valid = y[mask]

        if len(x_valid) == 0:
            continue

        ax1.plot(
            x_valid,
            y_valid,
            label=metrics["model_name"],
            linewidth=3,
            color=PLOT_COLORS.get(metrics["model_name"], None),
            alpha=0.8
        )

        xmax = max(xmax, x_valid.max())

    ax1.set_xlabel("Users with ≥x train reviews", fontsize=14)
    ax1.set_ylabel("Mean BLEU", fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=12)

    if xmax > 0:
        ax1.set_xlim(0, xmax + 2)
    else:
        ax1.set_xlim(0, 100)

    ax1.set_ylim(0, 1)
    ax1.grid(True, linestyle="--", alpha=0.5)
    legend1 = ax1.legend(loc="upper left", fontsize=12)
    ax1.add_artist(legend1)

    testcase_line = Line2D(
        [0], [0],
        color="gray",
        linewidth=3,
        alpha=0.7,
        linestyle="-",
        label="Test cases"
    )

    ax1.legend(
        handles=[testcase_line],
        loc="upper right",
        fontsize=12
    )

    # RIGHT AXIS (Test cases)
    for metrics in data["metrics"]:
        x = np.array(metrics["min_photos"])
        num_cases = np.array(metrics["num_cases"])

        mask = ~np.isnan(np.array(metrics["mean_bleu"]))

        ax2.plot(
            x[mask],
            num_cases[mask],
            linewidth=2.5,
            color="gray",
            alpha=0.4
        )

    ax2.set_ylabel("Test cases", fontsize=14, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray', labelsize=12)
    ax2.set_yscale("log")

    ax1.set_title(data["city"], fontsize=16)
    plt.tight_layout()

    plt.savefig(f'figures/{data["city"]}/bleu_{data["city"]}.pdf', bbox_inches='tight')
    plt.close()

def rouge_figure(data: dict):
    fig, ax1 = plt.subplots(figsize=(9, 5))

    # Aumentar tamaño de fuente
    plt.rcParams.update({'font.size': 14})

    ax2 = ax1.twinx()

    # LEFT AXIS (ROUGE curves)
    for metrics in data["metrics"]:
        x = np.array(metrics["min_photos"])
        y = np.array(metrics["mean_rouge"])

        mask = ~np.isnan(y)

        ax1.plot(
            x[mask],
            y[mask],
            label=metrics["model_name"],
            linewidth=3,
            color=PLOT_COLORS.get(metrics["model_name"], None),
            alpha=0.8
        )

    ax1.set_xlabel("Users with ≥x train reviews", fontsize=14)
    ax1.set_ylabel("Mean ROUGE", fontsize=14)
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    legend1 = ax1.legend(loc="upper left", fontsize=12)
    ax1.add_artist(legend1)

    testcase_line = Line2D(
        [0], [0],
        color="gray",
        linewidth=3,
        alpha=0.7,
        linestyle="-",
        label="Test cases"
    )

    ax1.legend(
        handles=[testcase_line],
        loc="upper right",
        fontsize=12
    )

    # RIGHT AXIS (Test cases)
    for metrics in data["metrics"]:
        x = np.array(metrics["min_photos"])
        num_cases = np.array(metrics["num_cases"])

        mask = ~np.isnan(np.array(metrics["mean_rouge"]))

        ax2.plot(
            x[mask],
            num_cases[mask],
            linewidth=2.5,
            color="gray",
            alpha=0.4
        )

    ax2.set_ylabel("Test cases", fontsize=14, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray', labelsize=12)
    ax2.set_yscale("log")

    ax1.set_title(data["city"], fontsize=16)
    plt.tight_layout()

    plt.savefig(f'figures/{data["city"]}/rouge_{data["city"]}.pdf', bbox_inches='tight')
    plt.close()

def generic_metric_figure(data: dict, metric_key: str, ylabel: str, filename: str):
    fig, ax1 = plt.subplots(figsize=(9, 5))

    # Aumentar tamaño de fuente
    plt.rcParams.update({'font.size': 14})

    ax2 = ax1.twinx()

    # LEFT AXIS (Metric curves)
    for metrics in data["metrics"]:
        x = np.array(metrics["min_photos"])
        y = np.array(metrics[metric_key], dtype=np.float32)

        mask = ~np.isnan(y.astype(np.float32))

        ax1.plot(
            x[mask],
            y[mask],
            label=metrics["model_name"],
            linewidth=3,
            color=PLOT_COLORS.get(metrics["model_name"], None),
            alpha=0.8
        )

    ax1.set_xlabel("Users with ≥x train reviews", fontsize=14)
    ax1.set_ylabel(ylabel, fontsize=14)

    ymax = max([
        np.nanmax(metrics[metric_key])
        for metrics in data["metrics"]
        if len(metrics[metric_key]) > 0
    ])

    if metric_key == "mean_dist2":
        ax1.set_ylim(0.97, 1)
    else:
        ax1.set_ylim(0, ymax * 1.1)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    legend1 = ax1.legend(loc="upper left", fontsize=12)
    ax1.add_artist(legend1)

    testcase_line = Line2D(
        [0], [0],
        color="gray",
        linewidth=3,
        alpha=0.7,
        linestyle="-",
        label="Test cases"
    )

    ax1.legend(
        handles=[testcase_line],
        loc="upper right",
        fontsize=12
    )

    # RIGHT AXIS (Test cases)
    for metrics in data["metrics"]:
        x = np.array(metrics["min_photos"])
        num_cases = np.array(metrics["num_cases"])

        y_metric = np.array(metrics[metric_key], dtype=np.float32)
        mask = ~np.isnan(y_metric.astype(np.float32))

        ax2.plot(
            x[mask],
            num_cases[mask],
            linewidth=2.5,
            color="gray",
            alpha=0.4
        )

    ax2.set_ylabel("Test cases", fontsize=14, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray', labelsize=12)
    ax2.set_yscale("log")

    ax1.set_title(data["city"], fontsize=16)
    plt.tight_layout()

    plt.savefig(f'figures/{data["city"]}/{filename}.pdf', bbox_inches='tight')
    plt.close()