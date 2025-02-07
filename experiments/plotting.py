import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from avtrust.plotting import get_quad_trust_axes, plot_trust_on_quad


font_family = "serif"


def make_trust_metric_meta_plot(
    all_meta_metrics,
    x_axis_var: str,
    x_axis_label: str,
    fig_title: str,
    fig_dir: str,
):
    """Make plots of median trust accuracy vs number of active agents"""
    df_meta = pd.DataFrame(all_meta_metrics)

    # apply some functions to get columns for plotting
    df_meta["last-agent-trust-metric"] = df_meta["metrics"].apply(
        lambda x: x[-1]["trust-agents"].mean_metric
    )
    df_meta["median-agent-trust-metric"] = df_meta["metrics"].apply(
        lambda x: np.nanmedian([xi["trust-agents"].mean_metric for xi in x])
    )
    df_meta["last-track-trust-metric"] = df_meta["metrics"].apply(
        lambda x: x[-1]["trust-tracks"].mean_metric
    )
    df_meta["median-track-trust-metric"] = df_meta["metrics"].apply(
        lambda x: np.nanmedian([xi["trust-tracks"].mean_metric for xi in x])
    )

    # label lines
    line_ys_legs = [
        ("median-agent-trust-metric", "Agent Trust Metric"),
        ("median-track-trust-metric", "Track Trust Metric"),
    ]

    # Create a line plot for each specified metric
    plt.figure(figsize=(8, 5))

    for y, label in line_ys_legs:
        # plot the median as a function of number of active agents
        sns.lineplot(
            x=x_axis_var,
            y=y,
            data=df_meta,
            label=label,
            linewidth=3,
        )

    # Add labels, title, and grid
    plt.ylim([0.5, 1])
    plt.xlabel(x_axis_label, family=font_family)
    plt.ylabel("Accuracy", family=font_family)
    plt.title("Trust Estimation Accuracy", family=font_family)
    plt.legend(prop={"family": font_family})
    plt.grid(True)
    plt.tight_layout()

    # save figure
    plt.savefig(os.path.join(fig_dir, fig_title + ".pdf"))
    plt.savefig(os.path.join(fig_dir, fig_title + ".png"))
    plt.show()


def make_assignment_metric_meta_plot(
    all_meta_metrics,
    x_axis_var: str,
    x_axis_label: str,
    fig_title: str,
    fig_dir: str,
):
    """Make plots of median trust accuracy vs number of active agents"""
    df_meta = pd.DataFrame(all_meta_metrics)

    # apply some functions to get columns for plotting
    df_meta["median-precision"] = df_meta["metrics"].apply(
        lambda x: np.nanmedian([xi["assignment-fused"].precision for xi in x])
    )
    df_meta["median-recall"] = df_meta["metrics"].apply(
        lambda x: np.nanmedian([xi["assignment-fused"].recall for xi in x])
    )
    df_meta["median-precision-filtered"] = df_meta["metrics"].apply(
        lambda x: np.nanmedian([xi["assignment-fused-filtered"].precision for xi in x])
    )
    df_meta["median-recall-filtered"] = df_meta["metrics"].apply(
        lambda x: np.nanmedian([xi["assignment-fused-filtered"].recall for xi in x])
    )

    # label lines
    line_ys_legs = [
        ("median-recall", "Recall, No Trust"),
        ("median-precision", "Precision, No Trust"),
        ("median-recall-filtered", "Recall, Trust"),
        ("median-precision-filtered", "Precision, Trust"),
    ]

    # Create a line plot for each specified metric
    plt.figure(figsize=(8, 5))

    for y, label in line_ys_legs:
        # plot the median as a function of number of active agents
        sns.lineplot(
            x=x_axis_var,
            y=y,
            data=df_meta,
            label=label,
            linewidth=3,
        )

    # Add labels, title, and grid
    plt.ylim([0, 1])
    plt.xlabel(x_axis_label, family=font_family)
    plt.ylabel("Metric", family=font_family)
    plt.title("Fusion Performance", family=font_family)
    plt.legend(prop={"family": font_family})
    plt.grid(True)
    plt.tight_layout()

    # save figure
    plt.savefig(os.path.join(fig_dir, fig_title + ".pdf"))
    plt.savefig(os.path.join(fig_dir, fig_title + ".png"))
    plt.show()


def make_plot_for_case_assignment(metrics, fig_title: str, fig_dir: str):
    df_metrics = pd.DataFrame(metrics)

    # apply some functions to derive plotable metrics

    # -- self
    df_metrics["tracking-precision-self"] = df_metrics["assignment-self"].apply(
        lambda x: x.precision
    )
    df_metrics["tracking-recall-self"] = df_metrics["assignment-self"].apply(
        lambda x: x.recall
    )

    # -- fused before filtering
    df_metrics["tracking-precision-fused"] = df_metrics["assignment-fused"].apply(
        lambda x: x.precision
    )
    df_metrics["tracking-recall-fused"] = df_metrics["assignment-fused"].apply(
        lambda x: x.recall
    )

    # -- fused after filtering
    df_metrics["tracking-precision-fused-filtered"] = df_metrics[
        "assignment-fused-filtered"
    ].apply(lambda x: x.precision)
    df_metrics["tracking-recall-fused-filtered"] = df_metrics[
        "assignment-fused-filtered"
    ].apply(lambda x: x.recall)

    # Define the line labels for readability
    line_ys_legs = [
        # ("tracking-recall-self", "Ego Recall"),
        # ("tracking-precision-self", "Ego Precision"),
        ("tracking-recall-fused", "Recall, No Trust"),
        ("tracking-precision-fused", "Precision, No Trust"),
        ("tracking-recall-fused-filtered", "Recall, Trust"),
        ("tracking-precision-fused-filtered", "Precision, Trust"),
    ]

    # Create a line plot for each specified metric
    plt.figure(figsize=(8, 5))

    for y, label in line_ys_legs:
        sns.lineplot(
            x="timestamp",
            y=y,
            data=df_metrics,
            label=label,
        )

    # Add labels, title, and grid
    plt.ylim([0.5, 1])
    plt.xlabel("Timestamp", family=font_family)
    plt.ylabel("Metrics Value", family=font_family)
    plt.title("Tracking Metrics Over Time", family=font_family)
    plt.legend(prop={"family": font_family})
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, fig_title + ".pdf"))
    plt.savefig(os.path.join(fig_dir, fig_title + ".png"))

    plt.show()


def make_plot_for_case_trust_metric(metrics, fig_title: str, fig_dir: str):
    df_metrics = pd.DataFrame(metrics)
    df_metrics["trust-agents-mean-metric"] = df_metrics["trust-agents"].apply(
        lambda x: x.mean_metric
    )
    df_metrics["trust-tracks-mean-metric"] = df_metrics["trust-tracks"].apply(
        lambda x: x.mean_metric
    )

    # Define the line labels for readability
    line_ys_legs = [
        ("trust-agents-mean-metric", "Agent Trust Acc."),
        ("trust-tracks-mean-metric", "Track Trust Acc."),
    ]

    # Create a line plot for each specified metric
    plt.figure(figsize=(8, 5))

    for y, label in line_ys_legs:
        sns.lineplot(
            x="timestamp",
            y=y,
            data=df_metrics,
            label=label,
            linewidth=3,
        )

    # Add labels, title, and grid
    plt.ylim([0, 1])
    plt.xlabel("Timestamp", family=font_family)
    plt.ylabel("Accuracy", family=font_family)
    plt.title("Trust Performance Metrics", family=font_family)
    plt.legend(prop={"family": font_family})
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, fig_title + ".pdf"))
    plt.savefig(os.path.join(fig_dir, fig_title + ".png"))

    plt.show()


def plot_last_trust_distributions(all_diag, fig_title: str, fig_dir: str):
    df_diag = pd.DataFrame(all_diag)
    df_trust_last_frame = df_diag.loc[df_diag["frame"] == df_diag["frame"].max()][
        ["agent", "trust-agents", "trust-tracks"]
    ]
    for idx_agent, trust_agents, trust_tracks in df_trust_last_frame.itertuples(
        index=False
    ):
        axs = get_quad_trust_axes(font_family=font_family)
        plot_trust_on_quad(
            axs=axs,
            trust_agents=trust_agents,
            trust_tracks=trust_tracks,
            font_family=font_family,
        )

        # save figure
        plt.tight_layout()
        fig_title_agent = fig_title + f"agent_{idx_agent}"
        plt.savefig(os.path.join(fig_dir, fig_title_agent + ".pdf"))
        plt.savefig(os.path.join(fig_dir, fig_title_agent + ".png"))

        plt.show()
