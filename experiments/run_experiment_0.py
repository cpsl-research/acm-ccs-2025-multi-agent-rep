import argparse
import os
import pickle

from algorithms import run_experiment


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="outputs")
    parser.add_argument(
        "--data_dir", type=str, default="../data/multi-agent-aerial-dense/raw"
    )
    parser.add_argument("--n_agents", type=int, default=25)
    parser.add_argument("--n_frames", type=int, default=50)
    parser.add_argument("--scene_idx", type=int, default=0)
    parser.add_argument("--with_prior", action="store_true")
    args = parser.parse_args()

    # run the experiment
    all_metrics, all_diag = run_experiment(
        data_dir=args.data_dir,
        n_agents=args.n_agents,
        n_frames=args.n_frames,
        scene_index=args.scene_idx,
        pct_fp_attacked=0.20,
        pct_fn_attacked=0.0,
        strong_prior_unattacked=args.with_prior,
        n_frames_trust_burnin=5,
    )

    # save the results
    save_dir = os.path.join(args.exp_dir, "experiment_0")
    os.makedirs(save_dir, exist_ok=True)
    for extension, data in zip(["metrics", "diag"], [all_metrics, all_diag]):
        save_file = os.path.join(save_dir, f"{extension}.p")
        with open(save_file, "wb") as f:
            pickle.dump(data, f)
