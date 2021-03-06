#!/usr/bin/env python3

from typing import Dict, List, Tuple, Union

import numpy as np

from equilib.grid_sample import numpy_func

from ..base import BaseEqui2Equi
from .utils import create_rotation_matrix

__all__ = ["Equi2Equi"]


class Equi2Equi(BaseEqui2Equi):
    def __init__(self, **kwargs) -> None:
        r"""Equi2Equi Numpy"""
        super().__init__(**kwargs)

    def create_coordinate(self, h_out: int, w_out: int) -> np.ndarray:
        r"""Create mesh coordinate grid with height and width

        return:
            coordinate: numpy.ndarray
        """
        xs = np.linspace(1, w_out, w_out)
        #theta = xs * 2 * np.pi / w_out - np.pi
        theta = (xs - w_out / 2 - 0.5) / w_out * np.pi * 2

        ys = np.linspace(1, h_out, h_out)
        #phi = ys * np.pi / h_out - np.pi / 2
        phi = (ys - h_out / 2 - 0.5) / h_out * np.pi

        theta, phi = np.meshgrid(theta, phi)
        coord = np.stack((theta, -phi), axis=-1)
        return coord

    def rotation_matrix(
        self,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> np.ndarray:
        r"""Create Rotation Matrix

        params:
            roll: x-axis rotation float
            pitch: y-axis rotation float
            yaw: z-axis rotation float

        return:
            rotation matrix: numpy.ndarray

        Global coordinates -> x-axis points forward, z-axis points upward
        """
        R = create_rotation_matrix(x=roll, y=pitch, z=yaw)
        return R

    @staticmethod
    def _get_img_size(img: np.ndarray) -> Tuple[int]:
        r"""Return height and width"""
        return img.shape[-2:]

    def _run_single(
        self,
        src: np.ndarray,
        rot: Dict[str, float],
        sampling_method: str,
        mode: str,
        matrix: np.ndarray
    ) -> np.ndarray:
        # define variables
        h_equi, w_equi = self._get_img_size(src)
        if self.h_out is None and self.w_out is None:
            self.h_out = h_equi
            self.w_out = w_equi

        a = self.create_coordinate(self.h_out, self.w_out)  # (theta, phi)s
        norm_A = 1
        x = norm_A * np.cos(a[:, :, 1]) * np.sin(a[:, :, 0])
        y = norm_A * np.cos(a[:, :, 1]) * np.cos(a[:, :, 0])
        z = norm_A * np.sin(a[:, :, 1])
        A = np.stack((x, y, z), axis=-1)

        #R = self.rotation_matrix(**rot)
        R = matrix

        # conversion:
        # B = R @ A
        #A = A[:, :, :, np.newaxis]
        #B = R @ A
        #B = B.squeeze(3)

        xyzNew = A.reshape(-1, 3)
        B = np.linalg.solve(R, xyzNew.T).T
        #B = xyzNew @ R
        B = B.reshape(h_equi, w_equi, 3)

        # calculate rotations per perspective coordinates
        phi = np.arcsin(B[:, :, 2] / np.linalg.norm(B, axis=-1))
        #theta = np.arctan2(B[:, :, 1], B[:, :, 0])

        normXY = np.sqrt(B[:, :, 1] ** 2 +  B[:, :, 0] ** 2)
        normXY[normXY < 0.000001] = 0.000001
        theta = np.arcsin( B[:, :, 0] / normXY)

        valid = (B[:, :, 1] < 0) & (theta >= 0)
        theta[valid] = np.pi - theta[valid]

        valid = (B[:, :, 1] < 0) & (theta <= 0)
        theta[valid] = -np.pi - theta[valid]

        ui = (theta + np.pi) / (2 * np.pi) * w_equi + 0.5
        uj = (-phi + np.pi / 2) / np.pi * h_equi + 0.5

        # center the image and convert to pixel location
        #ui = (theta - np.pi) * w_equi / (2 * np.pi)
        #uj = (phi - np.pi / 2) * h_equi / np.pi
        # out-of-bounds calculations
        ui = np.where(ui < 0, ui + w_equi, ui)
        ui = np.where(ui >= w_equi, ui - w_equi, ui)
        uj = np.where(uj < 0, uj + h_equi, uj)
        uj = np.where(uj >= h_equi, uj - h_equi, uj)
        grid = np.stack((uj, ui), axis=0)

        # grid sample
        grid_sample = getattr(numpy_func, sampling_method, "faster")
        sampled = grid_sample(src, grid, mode=mode)
        return sampled

    def run(
        self,
        src: Union[np.ndarray, List[np.ndarray]],
        rot: Union[Dict[str, float], List[Dict[str, float]]],
        sampling_method: str = "faster",
        mode: str = "bilinear",
        matrix: np.ndarray=None
    ) -> np.ndarray:
        r"""Run Equi2Pers

        params:
            src: equirectangular image np.ndarray[C, H, W]
            rot: Dict[str, float]
            sampling_method: str (default="faster")
            mode: str (default="bilinear")

        returns:
            pers: perspective image np.ndarray[C, H, W]

        NOTE: input can be batched [B, C, H, W] or List[np.ndarray]
        NOTE: when using batches, the output types match
        """
        _return_type = type(src)
        _original_shape_len = len(src.shape)
        if _return_type == np.ndarray:
            assert (
                _original_shape_len >= 3
            ), "ERR: got {} for input equi".format(_original_shape_len)
            if _original_shape_len == 3:
                src = src[np.newaxis, :, :, :]
                rot = [rot]

        assert len(src) == len(
            rot
        ), "ERR: length of src and rot differs {} vs {}".format(
            len(src), len(rot)
        )
        samples = []
        for s, r in zip(src, rot):
            # iterate through batches
            # TODO: batch implementation
            sample = self._run_single(
                src=s,
                rot=r,
                sampling_method=sampling_method,
                mode=mode,
                matrix=matrix
            )
            samples.append(sample)

        if _return_type == np.ndarray:
            samples = np.stack(samples, axis=0)
            if _original_shape_len == 3:
                samples = np.squeeze(samples, axis=0)

        return samples
