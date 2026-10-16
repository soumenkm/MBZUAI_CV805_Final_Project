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

def main(order_tag: str, seeds: list, frac: float) -> None:
    ana = Analysis(order_tag=order_tag, seeds=seeds, frac=frac)
    ana.zero_shot_table(tau=0.3)

if __name__ == "__main__":
    main("ACPR", [0], 0.1)