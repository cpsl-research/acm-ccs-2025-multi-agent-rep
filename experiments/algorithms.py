from functools import partial
from typing import Dict

import numpy as np
from avapi.carla import CarlaScenesManager
from avsec import (
    AdversaryModel,
    FalseNegativeManifest,
    FalsePositiveManifest,
    StaticPropagator,
)
from avstack.calibration import CameraCalibration
from avstack.datastructs import DataContainer
from avstack.environment import ObjectState
from avstack.geometry import GlobalOrigin3D, Polygon, ReferenceFrame, q_mult_vec
from avstack.geometry.fov import box_in_fov
from avstack.maskfilters import box_in_fov as box_in_fov_calib
from avstack.metrics import get_instantaneous_metrics
from avstack.modules.clustering import SampledAssignmentClusterer
from avstack.modules.fusion import CovarianceIntersectionFusionToBox
from avstack.modules.perception.detections import BoxDetection
from avstack.modules.perception.object2dfv import MMDetObjectDetector2D
from avstack.modules.tracking import BasicBoxTracker3D
from avtrust import CentralizedTrustEstimator, TrustArray, TrustUpdater, ViewBasedPsm
from avtrust.metrics import get_trust_agents_metrics, get_trust_tracks_metrics
from tqdm import tqdm


def get_ground_fov_from_camera_fov(
    calib: CameraCalibration,
    frame: int,
    timestamp: float,
    square_pixels: bool = False,
    in_global_frame: bool = True,
) -> Polygon:
    """Gets the viewable area on the ground plane in a ground-projected reference"""

    # get the spread of viewing angle
    d_az = np.arctan(calib.img_shape[1] / (2 * calib.P[0, 0]))
    if square_pixels:
        d_el = np.arctan(calib.img_shape[0] / (2 * calib.P[1, 1]))
    else:
        d_el = d_az
    ref_int = calib.reference.integrate(start_at=GlobalOrigin3D)

    # go from angles to coordinates
    dx = d_az * ref_int.x[2]
    dy = d_el * ref_int.x[2]

    # store the vertices of the polygon
    vertices = np.array(
        [
            [-dx, -dy],
            [-dx, dy],
            [dx, dy],
            [dx, -dy],
        ]
    )

    # adjust to the global frame
    if in_global_frame:
        # apply the rotation adjustment
        vert_homog = np.append(vertices, np.zeros((len(vertices), 1)), 1)
        vert_homog = q_mult_vec(ref_int.q, vert_homog)
        vertices = vert_homog[:, :2]

        # apply the position adjustment
        vertices += ref_int.x[:2]
        reference = GlobalOrigin3D
    else:
        reference = calib.reference.get_ground_projected_reference()

    # make fov as a polygon
    fov = Polygon(
        boundary=vertices,
        reference=reference,
        frame=frame,
        timestamp=timestamp,
    )

    return fov


def convert_2d_to_3d_detections(dets_2d: DataContainer, reference: ReferenceFrame):
    """Uses constraint that objects are on the ground"""
    ref_to_global = reference.integrate(start_at=GlobalOrigin3D)

    # each corner of the box has an angle off boresight
    dets_3d = DataContainer(
        frame=dets_2d.frame,
        timestamp=dets_2d.timestamp,
        data=[],
        source_identifier=dets_2d.source_identifier,
    )
    z = ref_to_global.x[0]  # assume x is direction to the ground
    for det in dets_2d:
        box_3d = det.box.upscale_to_3d(z_to_box=z, height=2.0)
        dets_3d.append(
            BoxDetection(
                data=box_3d,
                noise=np.array([1, 1, 1, 1, 1, 1]) ** 2,
                source_identifier=det.source_identifier,
                reference=det.reference,
            )
        )
    return dets_3d


def communicate(
    agent_receive: ObjectState,
    agent_send: ObjectState,
    d_max: float = 30.0,
    model: str = "quadratic",
) -> bool:
    """Probabilistic, distance-based communication model"""
    d = agent_receive.position.distance(agent_send.position)

    # run the communications model
    if model == "quadratic":
        # p = 1 - 1/d_max*2 * d**2
        raise NotImplementedError
        comm = random.rand <= p
    elif model == "absolute":
        comm = d <= d_max
    elif model == "always":
        comm = True
    else:
        raise NotImplementedError(model)

    return comm


def fusion(
    id_self: int,
    tracks_self: DataContainer,
    tracks_received: Dict[int, DataContainer],
    trust_agents: TrustArray,
    trust_tracks: TrustArray,
    clustering: str = "assignment",
    assign_radius: float = 2.0,
    fusion: str = "ci",
) -> DataContainer:
    """Perform fusion of own tracks with neighbors"""

    # perform assignment/clustering
    if clustering == "assignment":
        clusters = SampledAssignmentClusterer.cluster(
            objects={id_self: tracks_self, **tracks_received},
            frame=tracks_self.frame,
            timestamp=tracks_self.timestamp,
            assign_radius=assign_radius,
            check_reference=True,
        )
    else:
        raise NotImplementedError

    # get the trust means
    if trust_agents is not None:
        trust_means = {ID: trust.mean for ID, trust in trust_agents.items()}
    else:
        trust_means = None

    # perform fusion on output clusters
    if fusion == "ci":
        tracks_out = []
        for cluster in clusters:

            # get weights for fusion
            if trust_means is not None:
                weights = [trust_means[ID] for ID in cluster.agent_IDs]
            else:
                weights = "uniform"

            # fuse tracks and add to result
            fused_tracks = CovarianceIntersectionFusionToBox.fuse(
                tracks=cluster,
                weights=weights,
                force_ID=True,
            )
            tracks_out.append(fused_tracks)
    else:
        raise NotImplementedError

    return DataContainer(
        data=tracks_out,
        frame=tracks_self.frame,
        timestamp=tracks_self.timestamp,
        source_identifier=tracks_self.source_identifier,
    )


def track_in_fov(fov, calib, track):
    return box_in_fov_calib(box_3d=track.box, calib=calib)
    # return box_in_fov(track.box, fov)


def run_experiment(
    n_agents: int,
    n_frames: int,
    scene_index: int,
    pct_fp_attacked: float,
    pct_fn_attacked: float,
    strong_prior_unattacked: bool = False,
    n_frames_trust_burnin: int = 5,
    d_comms: int = 40,
    data_dir="/data/shared/CARLA/multi-agent-aerial-dense/raw",
    with_diagnostics: bool = False,
):
    """Run the trust experiments"""

    # set up the datasets
    CSM = CarlaScenesManager(data_dir=data_dir)
    CDM = CSM.get_scene_dataset_by_index(scene_index)

    # set up the sensor/agents
    sensor = "camera-0"
    agents = CDM.get_agents(frame=None)
    agents = np.random.choice(agents, size=n_agents, replace=False)

    # determine which agents are compromised
    agent_IDs = [agent.ID for agent in agents]
    n_fp_att = int(pct_fp_attacked * len(agents))
    n_fn_att = int(pct_fn_attacked * len(agents))
    idx_fp_att = np.random.choice(agent_IDs, size=n_fp_att, replace=False)
    idx_fn_att = np.random.choice(agent_IDs, size=n_fn_att, replace=False)
    print(f"Running with FPs on {idx_fp_att} and FNs on {idx_fn_att}")

    # set up the algorithms
    if strong_prior_unattacked:
        prior_agents = {
            agent.ID: {"type": "trusted", "strength": 0.9}
            if agent.ID not in idx_fp_att
            else {"type": "untrusted", "strength": 1.0}
            for agent in agents
        }
        print(prior_agents)
    else:
        prior_agents = {}
    perception = MMDetObjectDetector2D(model="fasterrcnn", dataset="carla-joint")
    trackers = {agent.ID: BasicBoxTracker3D() for agent in agents}
    trust_est = {
        agent.ID: CentralizedTrustEstimator(
            measurement=ViewBasedPsm(),
            updater=TrustUpdater(
                agent_negativity_bias=3,
                track_negativity_bias=1.0,
                prior_agents=prior_agents,
            ),
        )
        for agent in agents
    }

    # set up the adversary hooks
    fp_adv = FalsePositiveManifest(exact_select=4, x_sigma=15, x_bias=0)
    fn_adv = FalseNegativeManifest(n_select_poisson=3, max_range=40)
    adv_hooks = {
        agent.ID: AdversaryModel(
            propagator=StaticPropagator(),
            manifest_fp=fp_adv if agent.ID in idx_fp_att else None,
            manifest_fn=fn_adv if agent.ID in idx_fn_att else None,
            manifest_tr=None,
            dt_init=1.0,
            dt_reset=10.0,
            enabled=True,
        )
        for agent in agents
    }

    # set up data structures
    all_metrics = []
    all_diag = []
    truths_3d = {agent.ID: None for agent in agents}
    fov_agents = {agent.ID: None for agent in agents}
    tracks_3d = {agent.ID: None for agent in agents}
    fused_3d = {agent.ID: None for agent in agents}
    trust_agents = {agent.ID: None for agent in agents}
    trust_tracks = {agent.ID: None for agent in agents}
    last_imgs = {agent.ID: None for agent in agents}

    # flags
    det_type = "3d"  # "2d_perception", "2d_conversion", "3d"

    # loop over frames and replay some data
    for frame in tqdm(CDM.get_frames(sensor=sensor, agent=0)[3:n_frames]):

        # get global information
        timestamp = CDM.get_timestamp(frame=frame, sensor=sensor, agent=0)
        truths_global = CDM.get_objects_global(frame=frame)
        agent_positions = {agent.ID: agent.position for agent in agents}

        # loop over agents at this frame for local processing
        for agent_local in agents:

            # get data
            img = CDM.get_image(frame=frame, sensor=sensor, agent=agent_local.ID)
            truths_3d[agent_local.ID] = CDM.get_objects(
                frame=frame, sensor=sensor, agent=agent_local.ID
            )
            last_imgs[agent_local.ID] = img

            # run perception
            if det_type == "2d_perception":
                objs_det = perception(img)
            elif det_type in ["2d_conversion", "3d"]:

                # get bounding boxes
                objs_det = truths_3d[agent_local.ID].apply_and_return(
                    "getattr", "box3d"
                )
                if det_type == "2d_conversion":
                    objs_det = objs_det.apply_and_return(
                        "project_to_2d_bbox", img.calibration
                    )
                    noise = np.array([5, 5, 5, 5]) ** 2
                elif det_type == "3d":
                    noise = np.array([1, 1, 1, 1, 1, 1]) ** 2

                # build box detections
                objs_det = objs_det.apply_and_return(
                    BoxDetection,
                    noise=noise,
                    source_identifier=0,
                    reference=img.calibration.reference,
                )

            else:
                raise NotImplementedError(det_type)

            # convert 2d detections to 3d bounding boxes
            if det_type in ["2d_perception", "2d_conversion"]:
                objs_det_3d = convert_2d_to_3d_detections(objs_det, img.reference)
            else:
                objs_det_3d = objs_det

            # get FOV model of the agent -- assume just camera angles
            fov_agents[agent_local.ID] = get_ground_fov_from_camera_fov(
                calib=img.calibration,
                frame=img.frame,
                timestamp=img.timestamp,
            )

            # apply adversary model to the detections
            objs_det_3d, _, _ = adv_hooks[agent_local.ID](
                objects=objs_det_3d,
                fov=fov_agents[agent_local.ID],
                reference=img.reference,
                fn_dist_threshold=30,
                threshold_obj_dist=70,
            )

            # convert to global reference frame
            objs_det_3d.apply("change_reference", GlobalOrigin3D, inplace=True)

            # run tracking locally and save the history
            tracks_3d[agent_local.ID] = trackers[agent_local.ID](
                detections=objs_det_3d,
                check_reference=False,
                platform=GlobalOrigin3D,
            )

        # loop over receiving agents
        for agent_receive in agents:
            receive_data = {}

            # loop over sending agents
            for agent_send in agents:

                # get info from all agents nearby
                if agent_send.ID == agent_receive.ID:
                    continue
                else:
                    # run communications model
                    if communicate(
                        agent_receive, agent_send, model="absolute", d_max=d_comms
                    ):
                        receive_data[agent_send.ID] = tracks_3d[agent_send.ID]
                    else:
                        pass

            # perform fusion on all the received data - weight by agent trustedness
            fused_3d[agent_receive.ID] = fusion(
                id_self=agent_receive.ID,
                tracks_self=tracks_3d[agent_receive.ID],
                tracks_received=receive_data,
                trust_agents=trust_agents[agent_receive.ID],
                trust_tracks=trust_tracks[agent_receive.ID],
                clustering="assignment",
                assign_radius=2.0,
                fusion="ci",
            )

            # perform distributed trust estimation
            if fused_3d[agent_receive.ID] is not None:
                agent_fusion_ids = list(receive_data.keys()) + [agent_receive.ID]
                if frame > n_frames_trust_burnin:
                    (
                        trust_agents[agent_receive.ID],
                        trust_tracks[agent_receive.ID],
                        psms_agents,
                        psms_tracks,
                    ) = trust_est[agent_receive.ID](
                        position_agents={
                            ID: agent_positions[ID] for ID in agent_fusion_ids
                        },
                        fov_agents={ID: fov_agents[ID] for ID in agent_fusion_ids},
                        tracks_agents={ID: tracks_3d[ID] for ID in agent_fusion_ids},
                        tracks_cc=fused_3d[agent_receive.ID],
                    )

        # -- objects viewable by any agent
        truths_global_visible = []
        for obj in truths_global:
            for agent_view in agents:
                if box_in_fov(obj.box, fov_agents[agent_view.ID]):
                    truths_global_visible.append(obj)
                    break

        # compute metrics for each agent
        for agent_metrics in agents:

            # -- objects viewable by only this agent
            agent_tracks_self = tracks_3d[agent_metrics.ID]
            agent_tracks_fused = fused_3d[agent_metrics.ID].filter(
                partial(
                    track_in_fov,
                    fov_agents[agent_metrics.ID],
                    last_imgs[agent_metrics.ID].calibration,
                )
            )
            # for track in agent_tracks_fused:
            #     print(trust_tracks[agent_metrics.ID][track.ID].mean)
            agent_tracks_fused_filter = agent_tracks_fused.filter(
                lambda x: trust_tracks[agent_metrics.ID][x.ID].mean >= 0.5,
            )
            # print(len(agent_tracks_fused) - len(agent_tracks_fused_filter))

            # preallocate metrics datastructure
            metrics = {
                "frame": frame,
                "timestamp": timestamp,
                "agent": agent_metrics.ID,
                "assignment-self": None,
                "assignment-fused": None,
                "assignment-fused-filtered": None,
                "trust-agents": None,
                "trust-tracks-local": None,
                "trust-tracks-global": None,
            }

            # preallocate diagnostics datastructure
            diag = {
                "frame": frame,
                "timestamp": timestamp,
                "agent": agent_metrics.ID,
                "tracks-self": agent_tracks_self,
                "tracks-fused": agent_tracks_fused,
                "trust-agents": trust_agents[agent_metrics.ID],
                "trust-tracks": trust_tracks[agent_metrics.ID],
            }

            # -------------------------------------------------
            # filter by FOV to just measure local information
            # -------------------------------------------------

            # convert truths to the global reference
            tracks_truth_global = truths_3d[agent_metrics.ID].apply_and_return(
                "change_reference",
                GlobalOrigin3D,
                inplace=False,
            )

            # -------------------------------------------------
            # assignment metrics
            # -------------------------------------------------

            # -- without trust filtering
            # metrics on only what the ego agent can see
            metrics_assign_self = get_instantaneous_metrics(
                tracks=agent_tracks_self,
                truths=tracks_truth_global,
                assign_radius=3,
                timestamp=timestamp,
                transform_to_global=True,
            )
            metrics["assignment-self"] = metrics_assign_self

            # metrics on the global operating picture
            metrics_assign_fused = get_instantaneous_metrics(
                tracks=agent_tracks_fused,
                truths=tracks_truth_global,
                assign_radius=3,
                timestamp=timestamp,
                transform_to_global=True,
            )
            metrics["assignment-fused"] = metrics_assign_fused

            # -- with trust filtering
            metrics_assign_fused_filtering = get_instantaneous_metrics(
                tracks=agent_tracks_fused_filter,
                truths=tracks_truth_global,
                assign_radius=3,
                timestamp=timestamp,
                transform_to_global=True,
            )
            metrics["assignment-fused-filtered"] = metrics_assign_fused_filtering

            # -------------------------------------------------
            # trust metrics
            # -------------------------------------------------

            if trust_agents[agent_metrics.ID] is not None:
                # metrics on the agent trust estimation (local info)
                agent_trust_metrics = get_trust_agents_metrics(
                    truths_agents=tracks_truth_global,
                    tracks_agents=agent_tracks_self,
                    trust_agents=trust_agents[agent_metrics.ID],
                    attacked_agents=set(),
                    assign_radius=2.0,
                    use_f1_threshold=False,
                )
                metrics["trust-agents"] = agent_trust_metrics

                # -- global
                track_trust_metrics = get_trust_tracks_metrics(
                    truths=tracks_truth_global,
                    tracks_cc=agent_tracks_fused,
                    trust_tracks=trust_tracks[agent_metrics.ID],
                    assign_radius=2.0,
                )
                metrics["trust-tracks"] = track_trust_metrics

                # add metrics
                all_metrics.append(metrics)
                all_diag.append(diag)

    return all_metrics, all_diag
