#!/usr/bin/env python3
"""M1 step 2/3 — render the candidate-view RGB for each sampled decision point.

This is *not* an evaluation run. No harness, no navigation loop, no LLM, no config
touched: it opens habitat_sim directly, teleports the agent to a logged pose with
set_agent_state, renders, and writes PNGs. Scenes are visited once each (points are
grouped by scene) because scene load dominates the cost.

Camera rig is a byte-level replica of the online one, so the image handed to the
image arm is the same pixels SpatialBot described:
  * 12 RGB sensors, yaw k*30 deg for k=0..11  (vlnce_baselines/utils.py:get_camera_orientations,
    base_angle_rad = pi/6), pitch fixed 0.0
  * 224x224, HFOV 90                          (habitat_extensions/config/vlnce_task.yaml)
  * sensor position [0, 1.25, 0], agent height 1.5, radius 0.1  (habitat defaults)
  * candidate id k <-> camera k: construct_image_dicts() bins angle_deg into
    (30(k-1), 30k] and hands that bin image_dict[k], i.e. the camera at the bin's
    UPPER edge. Reproduced exactly, including that up-to-30-deg offset.

Pose convention (derived, then asserted at runtime): the trace's `heading` is
computed as atan2 of R^-1 [0,0,-1], which for a yaw-only rotation equals the yaw
itself, so agent rotation = quat_from_angle_axis(heading, [0,1,0]). The renderer
re-derives the heading from the state it just set and aborts if it does not match
the logged value -- a silent direction rotation is the failure mode this whole file
has to be paranoid about (see the load-bearing note in current_task.md §五).

Usage:
  python dump_candidate_views.py                    # all points in m1_points.json
  python dump_candidate_views.py --limit 4 --debug  # smoke: 4 points + a 12-view contact sheet
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N_VIEWS = 12
BASE_YAW = np.pi / 6  # 30 deg, exactly get_camera_orientations(12)'s base_angle_rad


def build_sim(scene_glb, width, height, hfov, gpu_id):
    import habitat_sim
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = scene_glb
    backend.gpu_device_id = gpu_id
    backend.allow_sliding = True
    specs = []
    for k in range(N_VIEWS):
        s = habitat_sim.SensorSpec()
        s.uuid = f'rgb_{k}'
        s.sensor_type = habitat_sim.SensorType.COLOR
        s.resolution = [height, width]
        s.position = [0.0, 1.25, 0.0]
        s.orientation = [0.0, BASE_YAW * k, 0.0]
        s.parameters['hfov'] = str(hfov)
        specs.append(s)
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = specs
    agent_cfg.height = 1.5
    agent_cfg.radius = 0.1
    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))


# Verified 20260727 by pixel identity: rendering camera k at heading h is the same
# image as camera 0 at heading h + k*30deg (mean |diff| <= 0.008/255, vs 54-74/255 for
# the opposite sign), i.e. sensor ORIENTATION [0, +yaw, 0] composes as a LEFT turn and
# shares sign and scale with the candidates' angle_rad. That is the assumption the whole
# candidate<->camera mapping rests on; see verify_view_mapping() below to re-run it.


def set_pose(sim, pos, heading):
    """Teleport; return the heading habitat reports back, recomputed with the exact
    formula the harness uses (habitat_extensions/nav.py:129)."""
    import habitat_sim
    from habitat_sim.utils.common import quat_from_angle_axis, quat_rotate_vector
    st = habitat_sim.AgentState()
    st.position = np.array(pos, dtype=np.float32)
    st.rotation = quat_from_angle_axis(float(heading), np.array([0.0, 1.0, 0.0]))
    sim.get_agent(0).set_state(st)
    got = sim.get_agent(0).get_state()
    hv = quat_rotate_vector(got.rotation.inverse(), np.array([0.0, 0.0, -1.0]))
    return float(np.arctan2(hv[0], -hv[2])), got.position


def verify_view_mapping(pt, width, height, hfov, gpu, tol=0.05):
    """Pixel-identity test for the camera<->candidate mapping: camera k at heading h
    must render exactly what camera 0 renders at heading h + k*30deg. The opposite
    sign is also scored, so the test is shown to discriminate rather than to pass
    vacuously. Returns True if every k matches."""
    sim = build_sim(pt['scene_glb'], width, height, hfov, gpu)
    try:
        set_pose(sim, pt['pos'], pt['head'])
        rig = {k: np.asarray(sim.get_sensor_observations()[f'rgb_{k}'])[:, :, :3].copy()
               for k in range(N_VIEWS)}
        ok = True
        for k in range(N_VIEWS):
            d = []
            for sign in (+1, -1):
                set_pose(sim, pt['pos'], pt['head'] + sign * BASE_YAW * k)
                ref = np.asarray(sim.get_sensor_observations()['rgb_0'])[:, :, :3]
                d.append(float(np.abs(rig[k].astype(int) - ref.astype(int)).mean()))
            ok &= d[0] <= tol
            print(f'  k={k:2d}  |cam_k - cam0@(h+k*30)| = {d[0]:7.4f}   '
                  f'|cam_k - cam0@(h-k*30)| = {d[1]:7.4f}')
        return ok
    finally:
        sim.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--points', default=os.path.join(HERE, 'm1_points.json'))
    ap.add_argument('--outdir', default=os.path.join(HERE, 'views'))
    ap.add_argument('--width', type=int, default=224)
    ap.add_argument('--height', type=int, default=224)
    ap.add_argument('--hfov', type=int, default=90)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0, help='only first N points (smoke)')
    ap.add_argument('--debug', action='store_true',
                    help='also write a 12-view contact sheet per point for eyeballing')
    ap.add_argument('--tol-deg', type=float, default=0.5,
                    help='abort if the re-derived heading drifts more than this')
    ap.add_argument('--verify-mapping', action='store_true',
                    help='run the camera<->candidate pixel-identity test and exit')
    args = ap.parse_args()

    from PIL import Image
    blob = json.load(open(args.points))
    pts = blob['points'][:args.limit] if args.limit else blob['points']

    if args.verify_mapping:
        print('camera<->candidate mapping check (camera k @ h  ==  camera 0 @ h+k*30deg):')
        ok = verify_view_mapping(pts[0], args.width, args.height, args.hfov, args.gpu)
        print('\n' + ('PASS: 相机 yaw = +k*30deg，与候选 angle_rad 同号同尺度'
                      if ok else 'FAIL: 方向映射不成立，勿继续'))
        sys.exit(0 if ok else 1)
    os.makedirs(args.outdir, exist_ok=True)

    by_scene = {}
    for p in pts:
        by_scene.setdefault(p['scene_glb'], []).append(p)
    print(f'{len(pts)} 决策点 / {len(by_scene)} 场景 -> {args.outdir}')

    manifest, n_img, worst_h, worst_p = {}, 0, 0.0, 0.0
    for si, (glb, group) in enumerate(sorted(by_scene.items()), 1):
        if not os.path.exists(glb):
            sys.exit(f'scene missing: {glb}')
        print(f'[{si}/{len(by_scene)}] {os.path.basename(glb)}  ({len(group)} 点)', flush=True)
        sim = build_sim(glb, args.width, args.height, args.hfov, args.gpu)
        try:
            for p in group:
                got_h, got_pos = set_pose(sim, p['pos'], p['head'])
                dh = abs((got_h - p['head'] + np.pi) % (2 * np.pi) - np.pi)
                dp = float(np.linalg.norm(np.array(got_pos) - np.array(p['pos'])))
                worst_h, worst_p = max(worst_h, np.degrees(dh)), max(worst_p, dp)
                if np.degrees(dh) > args.tol_deg:
                    sys.exit(f'heading mismatch ep{p["ep"]} step{p["step"]}: '
                             f'set {p["head"]:.4f} got {got_h:.4f} ({np.degrees(dh):.2f} deg)')
                obs = sim.get_sensor_observations()
                key = f'{p["ep"]}_{p["step"]}'
                files = {}
                for c in p['candidates']:
                    k = int(c['cid'])
                    if not 0 <= k < N_VIEWS:
                        sys.exit(f'candidate id out of camera range: {k}')
                    rgb = np.asarray(obs[f'rgb_{k}'])[:, :, :3]
                    fn = f'{key}_cand{k}.png'
                    Image.fromarray(rgb).save(os.path.join(args.outdir, fn))
                    files[c['cid']] = fn
                    n_img += 1
                if args.debug:
                    sheet = np.concatenate(
                        [np.concatenate([np.asarray(obs[f'rgb_{r*6+c}'])[:, :, :3]
                                         for c in range(6)], 1) for r in range(2)], 0)
                    Image.fromarray(sheet).save(os.path.join(args.outdir, f'{key}_ALL12.png'))
                manifest[key] = dict(ep=p['ep'], step=p['step'], scene=p['scene'],
                                     files=files, pose_err_m=dp, head_err_deg=float(np.degrees(dh)))
        finally:
            sim.close()

    with open(os.path.join(args.outdir, 'manifest.json'), 'w') as f:
        json.dump(dict(width=args.width, height=args.height, hfov=args.hfov,
                       n_points=len(pts), n_images=n_img, items=manifest), f, indent=1)
    print(f'\n{n_img} 张图 / {len(manifest)} 点  最大位姿误差 {worst_p:.4f}m  '
          f'最大朝向误差 {worst_h:.3f}deg')
    print(f'manifest -> {os.path.join(args.outdir, "manifest.json")}')


if __name__ == '__main__':
    main()
