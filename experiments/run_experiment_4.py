import argparse
import os
import pickle

from algorithms import run_experiment


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="outputs")
    parser.add_argument(
        "--data_dir", type=str, default="../data/multi-agent-aerial-dense"
    )
    parser.add_argument("--n_agents", type=int, default=25)
    parser.add_argument("--n_frames", type=int, default=50)
    parser.add_argument("--pct_fp_attacked", type=float, default=0.50)
    parser.add_argument("--pct_fn_attacked", type=float, default=0.0)
    parser.add_argument("--scene_idx", type=int, default=0)
    args = parser.parse_args()

    all_metrics = {}
    all_diag = {}

    # -- case 1: no prior information
    all_metrics["no_prior"], all_diag["no_prior"] = run_experiment(
        data_dir=args.data_dir,
        n_agents=args.n_agents,
        n_frames=args.n_frames,
        scene_index=args.scene_idx,
        pct_fp_attacked=args.pct_fp_attacked,
        pct_fn_attacked=args.pct_fn_attacked,
        strong_prior_unattacked=False,
        n_frames_trust_burnin=5,
    )

    # -- case 2: strong prior information
    all_metrics["prior"], all_diag["prior"] = run_experiment(
        data_dir=args.data_dir,
        n_agents=args.n_agents,
        n_frames=args.n_frames,
        scene_index=args.scene_idx,
        pct_fp_attacked=args.pct_fp_attacked,
        pct_fn_attacked=args.pct_fn_attacked,
        strong_prior_unattacked=True,
        n_frames_trust_burnin=5,
    )

    # save the results
    save_dir = os.path.join(args.exp_dir, "experiment_4")
    os.makedirs(save_dir, exist_ok=True)
    for extension, data in zip(["metrics", "diag"], [all_metrics, all_diag]):
        save_file = os.path.join(save_dir, f"{extension}.p")
        with open(save_file, "wb") as f:
            pickle.dump(data, f)
