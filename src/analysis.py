import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from config import domains

class Analysis:
    def __init__(self, order_tag: str, seeds: list, frac: float) -> None:
        """order_tag is the short name of the domain order, seeds are the runs
        to average over and frac is the fraction of the stream used in the runs"""
        self.order_tag = order_tag
        self.seeds = seeds
        self.frac = frac

        self.cwd = Path.cwd()
        self.output_dir = Path(self.cwd, "outputs")
        self.run_dir = Path(self.output_dir, "runs")
        self.table_dir = Path(self.output_dir, "tables")
        self.fig_dir = Path(self.output_dir, "figures")
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        self.runs = self._load_runs()

    def _load_runs(self) -> dict:
        """it loads all saved runs of this order and frac"""
        runs = {}
        for path in sorted(self.run_dir.glob(f"*_{self.order_tag}_seed*_frac{self.frac}.json")):
            with open(path, mode="r", encoding="utf-8") as f:
                run = json.load(f)
            if run["seed"] in self.seeds:
                runs.setdefault(run["name"], {})[run["seed"]] = run
        print(f"Loaded runs: { {name: sorted(seeds) for name, seeds in runs.items()} }")

        return runs

    @staticmethod
    def online_accuracy(run: dict) -> dict:
        """accuracy of the predictions made while the stream arrived, for the whole stream and for every segment"""
        correct = np.array(run["correct"])
        acc = {"all": correct.mean()}
        for seg in run["segments"]:
            acc[f"{seg['domain']}_{seg['visit']}"] = correct[seg["start"]:seg["end"]].mean()

        return acc

    @staticmethod
    def earlier_domain_accuracy(run: dict) -> float:
        """average heldout accuracy on the domains seen before the current segment, measured at the end of every segment.
        the current domain is left out because it shows adaptation, not memory"""
        accs = []
        for h in run.get("heldout", []):
            current = run["segments"][h["segment"]]["domain"]
            earlier = {seg["domain"] for seg in run["segments"][:h["segment"]]} - {current}
            accs += [h["acc"][d] for d in earlier]

        return np.mean(accs) if accs else np.nan

    @staticmethod
    def forgetting(run: dict) -> float:
        """for every domain: best heldout accuracy after the domain was first seen minus the final heldout accuracy.
        it returns the average over domains, a larger value means more forgetting"""
        heldout = run.get("heldout", [])
        if not heldout:
            return np.nan

        values = []
        for domain in heldout[-1]["acc"]:
            first = min(i for i, seg in enumerate(run["segments"]) if seg["domain"] == domain)
            best = max(h["acc"][domain] for h in heldout if h["segment"] >= first)
            values.append(best - heldout[-1]["acc"][domain])

        return np.mean(values)

    @staticmethod
    def return_accuracy(run: dict, k: int) -> float:
        """accuracy on the first k images of every domain's second visit, averaged over domains.
        compare it with the zero-shot row: a higher value means the method remembers the domain when it comes back"""
        correct = np.array(run["correct"])
        accs = [correct[seg["start"]:seg["start"] + k].mean() for seg in run["segments"] if seg["visit"] == "visit2"]

        return np.mean(accs)

    def summary(self, k: int = 100) -> pd.DataFrame:
        """one row per method with the mean and std over seeds of every metric"""
        rows = []
        for name, runs in self.runs.items():
            for seed, run in runs.items():
                online = self.online_accuracy(run)
                rows.append({
                    "method": name,
                    "online": online["all"],
                    "clipart": np.mean([v for key, v in online.items() if key.startswith("Clipart")]),
                    "earlier": self.earlier_domain_accuracy(run),
                    "forgetting": self.forgetting(run),
                    "return": self.return_accuracy(run, k),
                    "memory": run["memory_used"]
                })

        return pd.DataFrame(rows).groupby("method").agg(["mean", "std"])

    def zero_shot_table(self, tau: float = 0.3) -> None:
        """it saves table 1: zero-shot accuracy, admission rate and label accuracy of admitted images for every domain"""
        with open(Path(self.output_dir, "zero_shot_stats.json"), mode="r", encoding="utf-8") as f:
            stats = json.load(f)

        rows = []
        for domain, s in stats.items():
            rows.append({
                "domain": domain,
                "zero_shot_acc": s["zero_shot_acc"] * 100,
                "admitted": s[f"tau{tau}"]["admitted"] * 100,
                "label_acc": s[f"tau{tau}"]["label_acc"] * 100
            })
        df = pd.DataFrame(rows).round(2)

        csv_path = Path(self.table_dir, f"zero_shot_tau{tau}.csv")
        df.to_csv(csv_path, index=False)
        print(df.to_string(index=False))
        print(f"Table saved at {csv_path}")

    def main_table(self, methods: list) -> None:
        """it saves the main results as a csv file: one row per method, values in percent as mean and std over seeds"""
        summary = self.summary()
        summary = summary.loc[[m for m in methods if m in summary.index]]
        # flatten the two level column names, e.g. (online, mean) -> online_mean
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary = summary.drop(columns="memory_std")
        for col in summary.columns:
            if col != "memory_mean":
                summary[col] = summary[col] * 100
        summary = summary.round(2)

        csv_path = Path(self.table_dir, f"main_{self.order_tag}_frac{self.frac}.csv")
        summary.to_csv(csv_path)
        print(summary.to_string())
        print(f"Table saved at {csv_path}")

    def plot_online_accuracy(self, methods: list, seed: int, window: int = 100) -> None:
        """it plots the online accuracy over the stream, averaged over a sliding window, with a line at every domain change"""
        methods = [m for m in methods if m in self.runs]
        segments = self.runs[methods[0]][seed]["segments"]

        fig, ax = plt.subplots(figsize=(12, 4))
        for name in methods:
            correct = np.array(self.runs[name][seed]["correct"], dtype=float)
            smooth = np.convolve(correct, np.ones(window) / window, mode="valid")
            ax.plot(np.arange(len(smooth)) + window, smooth * 100, label=name, linewidth=1)

        for seg in segments:
            ax.axvline(seg["start"], color="grey", linestyle="--", linewidth=0.5)
            # the label is placed just above the plot, at the start of the segment
            ax.text(seg["start"], 1.01, f"{seg['domain'][:4]}{seg['visit'][-1]}", transform=ax.get_xaxis_transform(), fontsize=8)

        ax.set_xlabel("Stream position")
        ax.set_ylabel(f"Accuracy (%), window {window}")
        ax.legend(fontsize=8)
        plt.tight_layout()

        fig_path = Path(self.fig_dir, f"online_{self.order_tag}_seed{seed}_frac{self.frac}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"Figure saved at {fig_path}")

    def plot_cache_composition(self, methods: list, seed: int) -> None:
        """for every method it plots how many cached images come from each domain over the stream"""
        methods = [m for m in methods if m in self.runs and self.runs[m][seed].get("composition")]

        fig, axs = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4), squeeze=False, sharey=True)
        for ax, name in zip(axs[0], methods):
            comp = self.runs[name][seed]["composition"]
            steps = [c["step"] for c in comp]
            counts = [[c["counts"].get(d, 0) for c in comp] for d in domains]
            ax.stackplot(steps, counts, labels=domains)
            ax.set_title(name)
            ax.set_xlabel("Stream position")
            ax.set_ylabel("Cached images")
        axs[0][0].legend(loc="upper left", fontsize=8)
        plt.tight_layout()

        fig_path = Path(self.fig_dir, f"composition_{self.order_tag}_seed{seed}_frac{self.frac}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"Figure saved at {fig_path}")

def main(order_tag: str, seeds: list, frac: float) -> None:
    ana = Analysis(order_tag=order_tag, seeds=seeds, frac=frac)
    ana.zero_shot_table(tau=0.3)
    ana.main_table(["zero_shot", "unbounded", "fifo", "entropy", "oracle", "cmr", "true_labels"])
    ana.plot_online_accuracy(["zero_shot", "entropy", "cmr", "unbounded"], seed=seeds[0])
    ana.plot_cache_composition(["entropy", "oracle", "cmr"], seed=seeds[0])

if __name__ == "__main__":
    main("ACPR", [0, 1, 2], 0.1)
    main("RPCA", [0, 1, 2], 0.1)
    main("CRAP", [0, 1, 2], 0.1)