# Copyright (c) 2020 Carnegie Mellon University, Wenshan Wang <wenshanw@andrew.cmu.edu>
# For License information please see the LICENSE file in the root directory.

import numpy as np
from contextlib import suppress
from evaluator_base import ATEEvaluator, RPEEvaluator, KittiEvaluator, quats2SEs, transform_trajs

# from trajectory_transform import timestamp_associate

class TartanAirEvaluator:
    @staticmethod
    def evaluate_one_trajectory(gt_traj, est_traj, scale=False, kittitype=True):
        """
        scale = True: calculate a global scale
        """
        # load trajectories
        with suppress(TypeError):
            gt_traj = np.loadtxt(gt_traj)
            est_traj = np.loadtxt(est_traj)

        if gt_traj.shape[0] != est_traj.shape[0]:
            raise Exception("POSEFILE_LENGTH_ILLEGAL")
        if gt_traj.shape[1] != 7 or est_traj.shape[1] != 7:
            raise Exception("POSEFILE_FORMAT_ILLEGAL")

        # transform and scale
        gt_traj_trans, est_traj_trans, s = transform_trajs(gt_traj, est_traj, scale)
        # print("  Scale, {}".format(s))

        gt_SEs, est_SEs = quats2SEs(gt_traj_trans, est_traj_trans)

        (
            ate_score,
            gt_ate_aligned,
            est_ate_aligned,
            ate_rot,
            ate_trans,
            ate_scale,
            ate_T,
        ) = ATEEvaluator.evaluate(gt_traj, est_traj, scale)
        rpe_score = RPEEvaluator.evaluate(gt_SEs, est_SEs)
        kitti_score = KittiEvaluator.evaluate(gt_SEs, est_SEs, kittitype=kittitype)

        return {
            "ate_score": ate_score,
            "rpe_score": rpe_score,
            "kitti_score": kitti_score,
            "gt_aligned": gt_ate_aligned,
            "est_aligned": est_ate_aligned,
            "scale": s,
            "ate_scale": ate_scale,
            "ate_rot": ate_rot,
            "ate_trans": ate_trans,
            "ate_T": ate_T,
        }


if __name__ == "__main__":
    # scale = True for monocular track, scale = False for stereo track
    result = TartanAirEvaluator.evaluate_one_trajectory("pose_gt.txt", "pose_est.txt", scale=True)
    print(result)
