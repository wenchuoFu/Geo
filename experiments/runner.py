"""
Experiment runner for geo_seq_inference experiments E1–E7.

Reads YAML configs specifying parameter grids, runs Monte Carlo simulations,
and outputs parquet result files. Designed for single-machine execution;
total runtime target < 12 hours for all experiments.

Usage:
    python experiments/runner.py --config experiments/configs/e1.yaml --output-dir results/e1
    python experiments/runner.py --all --output-dir results/
"""

import argparse
import itertools
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


@dataclass
class ExperimentResult:
    """Single row of experiment output."""
    config_name: str
    params: Dict[str, Any]
    rejection_rate: float
    ci_coverage: float
    avg_ci_width: float
    avg_runtime_sec: float
    mc_replications: int
    seed: int


class ExperimentRunner:
    """Orchestrates parameter-grid experiments."""

    def __init__(self, config_path: str, output_dir: str):
        self.config = self._load_yaml(config_path)
        self.config_name = self.config.get("name", Path(config_path).stem)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_yaml(path: str) -> dict:
        try:
            import yaml
            with open(path) as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("pyyaml required for experiment configs. Install with: pip install pyyaml")

    def run(self) -> List[ExperimentResult]:
        param_grid = self._build_param_grid()
        fixed = self.config.get("fixed", {})
        n_jobs = fixed.get("n_jobs", 1)

        print(f"Experiment: {self.config_name}")
        print(f"  Parameter combinations: {len(param_grid)}")
        print(f"  Jobs: {n_jobs}")

        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(self._run_one_combination)(combo, seed)
            for seed, combo in enumerate(param_grid)
        )

        self._save_results(results)
        return results

    def _build_param_grid(self) -> List[Dict[str, Any]]:
        params = self.config.get("parameters", {})
        if not params:
            return [{}]

        keys = list(params.keys())
        values = [params[k] for k in keys]
        return [
            dict(zip(keys, combo))
            for combo in itertools.product(*values)
        ]

    def _run_one_combination(self, params: dict, seed: int) -> ExperimentResult:
        from gsi.dgp import make_paths, realize
        from gsi.bands import uniform_band_test
        from gsi.inversion import confidence_set
        from gsi.omega_design import design_omega

        fixed = self.config.get("fixed", {})
        N_MC = fixed.get("N_MC", 500)
        B = fixed.get("B", 999)
        alpha = fixed.get("alpha", 0.05)

        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()

        n_pairs = params.get("n_pairs", 30)
        T_max = params.get("T_max", 60)
        tail = params.get("tail", "gaussian")
        rho = params.get("rho", 0.3)
        theta_true = params.get("theta_true", 0.5)
        stat_name = params.get("stat", "sgn")
        hetero_scale = fixed.get("hetero_scale", 0.1)
        seasonality = fixed.get("seasonality", False)
        method = params.get("method", "gsi_fixed")

        # Omega
        omega_shape = params.get("omega_shape", "pocock")
        target_window = fixed.get("target_window", None)
        delta_star = fixed.get("delta_star", 0.5)
        omega = design_omega(None, omega_shape, target_window, delta_star, T_max)

        reject_count = 0
        coverage_count = 0
        ci_widths = []

        for mc in range(N_MC):
            mc_rng = np.random.default_rng(seed * 10000 + mc * 100 + 1)

            panel = make_paths(
                n_pairs=n_pairs, T_max=T_max, theta=theta_true,
                tail=tail, rho=rho, seasonality=seasonality,
                hetero_scale=hetero_scale, entry_days=None, rng=mc_rng,
            )
            Z = mc_rng.choice(np.array([-1, 1], dtype=np.int8), size=n_pairs)
            obs = panel.realize(Z, mc_rng)

            # Run test at theta0 = theta_true (null) or theta0 = 0 (alternative)
            # For size: theta0 == theta_true; for power: theta0 == 0
            if theta_true == 0.0:
                test_theta = 0.0  # under null
            else:
                test_theta = 0.0  # test null for power calculation

            try:
                if method == "gsi_fixed":
                    result = uniform_band_test(
                        obs=obs, theta0=test_theta, stat=stat_name,
                        omega=omega, B=B, alpha=alpha, rng=mc_rng,
                    )
                    if result.reject:
                        reject_count += 1

                elif method == "gsi_sequential":
                    result = uniform_band_test(
                        obs=obs, theta0=test_theta, stat=stat_name,
                        omega=omega, B=B, alpha=alpha, rng=mc_rng,
                    )
                    if result.reject:
                        reject_count += 1

                elif method in ("tm", "wald", "catoni", "hoeffding"):
                    # Baseline methods — skip in basic runner, handled by specialized scripts
                    pass

                # Coverage check: CI contains true_theta?
                if hasattr(panel, 'theta') and panel.theta[0] is not None:
                    cs = confidence_set(
                        obs=obs, stat=stat_name, omega=omega if method == "gsi_sequential" else None,
                        B=min(B, 499), alpha=alpha, rng=mc_rng,
                    )
                    if (cs.lower is None or cs.lower <= theta_true) and \
                       (cs.upper is None or cs.upper >= theta_true):
                        coverage_count += 1
                    if cs.lower is not None and cs.upper is not None:
                        ci_widths.append(cs.upper - cs.lower)

            except Exception as e:
                warnings.warn(f"MC replication {mc} failed: {e}")

        runtime = time.perf_counter() - t0

        N_eff = max(N_MC, 1)
        return ExperimentResult(
            config_name=self.config_name,
            params=params,
            rejection_rate=reject_count / N_eff,
            ci_coverage=coverage_count / N_eff,
            avg_ci_width=np.mean(ci_widths) if ci_widths else float('nan'),
            avg_runtime_sec=runtime,
            mc_replications=N_MC,
            seed=seed,
        )

    def _save_results(self, results: List[ExperimentResult]):
        rows = []
        for r in results:
            row = {"config_name": r.config_name, "seed": r.seed}
            row.update(r.params)
            row.update({
                "rejection_rate": r.rejection_rate,
                "ci_coverage": r.ci_coverage,
                "avg_ci_width": r.avg_ci_width,
                "avg_runtime_sec": r.avg_runtime_sec,
                "mc_replications": r.mc_replications,
            })
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = self.output_dir / f"{self.config_name}.csv"
        parquet_path = self.output_dir / f"{self.config_name}.parquet"
        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)

        print(f"  Results saved to: {csv_path} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Run geo_seq_inference experiments")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--all", action="store_true", help="Run all experiments E1-E7")
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Output directory for results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.all:
        config_dir = Path(__file__).parent / "configs"
        configs = sorted(config_dir.glob("e*.yaml"))
        print(f"Running {len(configs)} experiments: {[c.stem for c in configs]}")
        for cfg_path in configs:
            exp_output = output_dir / cfg_path.stem
            runner = ExperimentRunner(str(cfg_path), str(exp_output))
            runner.run()
    elif args.config:
        runner = ExperimentRunner(args.config, args.output_dir)
        runner.run()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
