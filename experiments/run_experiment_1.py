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
    parser.add_argument("--min_agents", type=int, default=6)
    parser.add_argument("--max_agents", type=int, default=51)
    parser.add_argument("--stride_agents", type=int, default=2)
    parser.add_argument("--n_scenes", type=int, default=1)
    parser.add_argument("--n_random_trials", type=int, default=3)
    parser.add_argument("--n_frames", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()

    # set up directories
    save_dir = os.path.join(args.exp_dir, "experiment_1")
    save_dir_sub = os.path.join(save_dir, "data")
    os.makedirs(save_dir_sub, exist_ok=True)

    # run experiments
    all_meta_metrics_benign = []
    np.random.seed(args.seed)
    for scene_idx in range(args.n_scenes):
        for n_agents in range(args.min_agents, args.max_agents, args.stride_agents):
            for i_trial in range(args.n_random_trials):
                # run the experiment
                print(
                    f"Evaluating Scene {scene_idx} with {n_agents} Agents for Trial {i_trial}"
                )
                all_metrics, all_diag = run_experiment(
                    data_dir=args.data_dir,
                    n_agents=n_agents,
                    n_frames=args.n_frames,
                    scene_index=scene_idx,
                    pct_fp_attacked=0.0,
                    pct_fn_attacked=0.0,
                    n_frames_trust_burnin=5,
                )

                # store the metrics
                all_meta_metrics_benign.append(
                    {
                        "n_agents": n_agents,
                        "trial_index": i_trial,
                        "scene_index": scene_idx,
                        "metrics": all_metrics,
                        "attacked": False,
                    }
                )

                # save the metrics from this scene
                save_file = os.path.join(
                    save_dir_sub,
                    f"agent_density_benign_scene{scene_idx}_agents{n_agents}_trial{i_trial}.p",
                )
                with open(save_file, "wb") as f:
                    pickle.dump(all_metrics, f)

    # save meta results
    save_file = os.path.join(save_dir, "metrics.p")
    with open(save_file, "wb") as f:
        pickle.dump(all_meta_metrics_benign, f)
