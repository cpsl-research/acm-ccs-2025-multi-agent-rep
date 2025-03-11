import argparse
import os
import pickle

import numpy as np
from algorithms import run_experiment


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="outputs")
    parser.add_argument(
        "--data_dir", type=str, default="../data/multi-agent-aerial-dense/raw"
    )
    parser.add_argument("--min_pct_attacked", type=float, default=0.1)
    parser.add_argument("--max_pct_attacked", type=int, default=0.8)
    parser.add_argument("--num_pct_attacked", type=int, default=12)
    parser.add_argument("--n_agents", type=int, default=25)
    parser.add_argument("--n_scenes", type=int, default=1)
    parser.add_argument("--n_random_trials", type=int, default=3)
    parser.add_argument("--n_frames", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()

    # set up directories
    save_dir = os.path.join(args.exp_dir, "experiment_3")
    save_dir_sub = os.path.join(save_dir, "data")
    os.makedirs(save_dir_sub, exist_ok=True)

    # run experiments
    all_meta_metrics_capability = []
    np.random.seed(args.seed)
    pcts_attacked = np.linspace(
        args.min_pct_attacked, args.max_pct_attacked, args.num_pct_attacked
    )
    for scene_idx in range(args.n_scenes):
        for pct_fp_attacked in pcts_attacked:
            for i_trial in range(args.n_random_trials):
                # run the experiment
                print(
                    f"Evaluating Scene {scene_idx} with {args.n_agents} Agents"
                    f" for Trial {i_trial} and {pct_fp_attacked} pct attacked"
                )
                all_metrics, all_diag = run_experiment(
                    data_dir=args.data_dir,
                    n_agents=args.n_agents,
                    n_frames=args.n_frames,
                    scene_index=scene_idx,
                    pct_fp_attacked=pct_fp_attacked,
                    pct_fn_attacked=0,
                    n_frames_trust_burnin=5,
                )

                # store the metrics
                all_meta_metrics_capability.append(
                    {
                        "n_agents": args.n_agents,
                        "pct_fp_attacked": pct_fp_attacked,
                        "trial_index": i_trial,
                        "scene_index": scene_idx,
                        "metrics": all_metrics,
                        "attacked": True,
                    }
                )

                # save the metrics from this scene
                save_file = os.path.join(
                    save_dir_sub,
                    f"attacker_capability_scene{scene_idx}_agents"
                    f"{args.n_agents}_trial{i_trial}_pctatt{pct_fp_attacked}.p",
                )
                with open(save_file, "wb") as f:
                    pickle.dump(all_metrics, f)

    # save meta results
    save_file = os.path.join(save_dir, "metrics.p")
    with open(save_file, "wb") as f:
        pickle.dump(all_meta_metrics_capability, f)
