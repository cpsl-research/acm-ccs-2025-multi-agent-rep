# Trust-Based Ground-Vehicle Autonomy

Experiments for Trust-Based Assured Sensor Fusion in Distributed Ground-Vehicle Autonomy to appear in CCS '25

This repository provides instructions to repeat the results of the [Multi-Agent Trust CCS paper][paper]. We provide a docker container that uses [`poetry`][poetry] to simplify reproducing our work.

## Summary of Instructions

NOTE: you do not need the full repository if using the docker image, however, it can be useful to clone the repository if you want to use the data download and install utilities. If cloning the repository and you want access to the submodules, ensure that you recurse the submodules when cloning with:

```
git clone --recurse-submodules https://github.com/cpsl-research/acm-ccs-2025-multi-agent-rep.git
```

## Requirements

Repeating these examples requires a few things:

#### GPU/Memory

- GPU with >= 5 GB memory for perception models
- Lots of NVME/SSD/HDD memory somewhere to save datasets. The amount of memory needed depends on how many scenes the user wants to download. If downloading all scenes, it will take around 200 GB. This is necessary to run all of the experiments and retrieve the results from the paper. Simply exploring data only requires a scene or two. Changing the number of downloaded scenes is a configuration parameter that can be set in `./run_install.sh`.

#### Datasets

The evaluations in the [paper][paper] generated a dataset from the [CARLA simulator][carla]. The datasets are publicly available but very large. The data is hosted on Google Drive in [this folder][data-folder]. We provide a script to download the data, `download_data_gdown.sh`. It worked for us to run this with `bash`. Some users have had issues with using the script, so instead, you can download the data manually from the drive folder. Put the data in a place that is consistent with `run_docker.sh`, e.g., `/data/test-ccs/data` (the default in the docker start script). 

#### Docker

To use the provided docker container, you must have the [NVIDIA Container Toolkit][nvidia-container] installed. Additionally, the docker container is beefy - it bases on the nvidia container and then installs [`avstack`][avstack-core] and [`avapi`][avstack-api] as development submodules. It is about 16 GB in size, so prepare for a lengthy pull process the first time. **NOTE:** The pull process happens inside `run_docker.sh`. 

## Installation

Running `./run_install.sh` has worked on multiple machines. This *should* be the only thing necessary to get a working installation. There is the possibility that your system's permissions do not allow you to make a `/data/` folder. If that is the case, change `DATAD` and `MODELD` to something else. It doesn't matter what they are set to. As we have configured the setup now, it will take around 1 hour to download the datasets and perception model weights.

A directory structure that worked for us is:
```
/data/test-ccs/
    data
        multi-agent-intersection
            run-2024-11-12_20:41:57
            etc.....
    models
        mmdet
            work_dirs
                carla
                    faster_rcnn_r50_fpn_1x_carla_joint.pth
                    faster_rcnn_r50_fpn_1x_carla_joint.py
```

### Getting into docker

The `run_install.sh` will start a docker container which will automatically trigger the running of a jupyter notebook. The data and model folders will automatically bind-mount into the docker container, and paths inside the docker container should be managed automatically without issue. **IMPORTANT:** The docker container maps port 8888 in the docker container (jupyter default) to an exposed port 8888 on the host machine. If this port is for some reason already occupied on your host machine, you must change the mapping inside `run_docker.sh` (NOTE: `./run_install.sh` calls `run_docker.sh` at the end).

If the docker container is successfully started, you will see a jupyter-notebook-type output in the console. You can verify in the `run_docker.sh` script that, upon starting the container, we immediately start a jupyter notebook. The console text will tell you of a URL to visit where the notebook is running (basically: `localhost:port:token`). It will also have token text appended to the URL. Copy and paste all of this into your browser. You may need this token to start the notebook because jupyter will not recognize the docker container as a trusted agent or something of that nature.

The experiments are run inside the docker container while the results are visualized in the notebooks within jupyter that should have appeared in the previous step. 

## Running Experiments

To run our experiments, enter the docker container (e.g., if already running from `run_docker.sh`, then just execute `docker exec run -it CONTAINER_ID_HERE bash`) and execute the python files in the `experiments` folder within a poetry shell. I.e., run
```
poetry shell
cd experiments
python 
```


 To run them all straight, you can execute `./run_all_experiments.sh`. Some of the experiments are large and take a while to run. `experiment_0.py` is quick, so if you just want a short verification, we suggest running that one. 

To visualize the results of the experiments, run the jupyter notebooks (`plot_experiment_results.ipynb`) from the notebook that opened. 


## Additional Details

N/A


[poetry]: https://github.com/python-poetry/poetry
[paper]: https://arxiv.org/pdf/2503.04954
[data-folder]: https://drive.google.com/drive/folders/1uLNB7F8bTOwtGkjJYmRaAzH-gZ613zkB
[avstack-core]: https://github.com/avstack-lab/avstack-core
[avstack-api]: https://github.com/avstack-lab/avstack-api
[avstack-lab]: https://github.com/avstack-lab
[carla-sandbox]: https://github.com/avstack-lab/carla-sandbox
[carla]: https://github.com/carla-simulator/carla
[nvidia-container]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
[generate-carla-dataset]: https://github/com/avstack-lab/carla-sandbox/docs/how-to-guides/generate-collaborative-dataset.md