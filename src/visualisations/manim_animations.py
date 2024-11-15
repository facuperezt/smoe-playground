#%%
import jupyter_manim
from manim import *
from numpy import typing as npt
from typing import Annotated, Any, Literal, Optional, Tuple, TypeVar

DType = TypeVar('DType', bound=np.generic)
Array2 = Annotated[npt.NDArray[DType], Literal[2]]
Array2x2 = Annotated[npt.NDArray[DType], Literal[2, 2]]
ArrayNx2 = Annotated[npt.NDArray[DType], Literal["N", 2]]
ArrayNx2x2 = Annotated[npt.NDArray[DType], Literal["N", 2, 2]]
ArrayNx1 = Annotated[npt.NDArray[DType], Literal["N", 1]]
ArrayNxM = Annotated[npt.NDArray[DType], Literal["N", "M"]]
ArrayMxN = Annotated[npt.NDArray[DType], Literal["M", "N"]]

def smoe_kernel(X: ArrayNx2[np.float32], mu: Array2[np.float32], A: Array2x2[np.float32]) -> ArrayNx1[np.float32]:
    centered = X - mu  # N x 2
    sigma = A@A.T  # 2 x 2
    exp = ((centered@sigma)*centered)  # N x 2
    return np.exp(-0.5*exp.sum(axis=1))  # N x 1

def smoe_gating(X: ArrayNx2[np.float32], mus: ArrayNx2[np.float32], As: ArrayNx2x2[np.float32]) -> ArrayNxM:
    kernels = np.stack([smoe_kernel(X, mu, A) for mu, A in zip(mus, As)], axis=1)
    kernels = kernels/kernels.sum(axis=1, keepdims=True)
    return kernels

def make_plotting_function_kernel(mu: Array2[np.float32], A: Array2x2[np.float32], nu: float):
    def _smoe_kernel(u: ArrayNx1, v: ArrayNx1) -> ArrayNx1:
        z = smoe_kernel(np.stack([u, v], axis=0).reshape(-1, 2), mu, A)
        return np.array([u, v, nu*z[0]])
    return _smoe_kernel

def get_values_points_kernels(mu: Array2[np.float32], A: Array2x2[np.float32], nu: float, u_lims: Tuple[int, int], v_lims: Tuple[int, int], resolution: Tuple[int, int] = (32, 32)):
    X, Y = np.meshgrid(np.linspace(u_lims[0], u_lims[1], resolution[0]), np.linspace(v_lims[0], v_lims[1], resolution[1]))
    XX = np.stack([X.flatten(), Y.flatten()], axis=1)
    Z = smoe_kernel(XX, mu, A)
    return XX[:, 0], XX[:, 1], Z

def make_plotting_function_gating(mus: ArrayNx2[np.float32], As: ArrayNx2x2[np.float32], nus: ArrayNx1[np.float32], ind: int):
    def _smoe_gating(u: ArrayNx1, v: ArrayNx1) -> ArrayNxM:
        z = smoe_gating(np.stack([u, v], axis=0).reshape(-1, 2), mus, As)
        return np.array([u, v, (nus*z)[0][ind]])
    return _smoe_gating

def get_values_points_gates(mus: ArrayNx2[np.float32], As: ArrayNx2x2[np.float32], nus: ArrayNx1[np.float32], u_lims: Tuple[int, int], v_lims: Tuple[int, int], resolution: Tuple[int, int] = (32, 32),  ind: Optional[int] = None):
    X, Y = np.meshgrid(np.linspace(u_lims[0], u_lims[1], resolution[0]), np.linspace(v_lims[0], v_lims[1], resolution[1]))
    XX = np.stack([X.flatten(), Y.flatten()], axis=1)
    Z = smoe_gating(XX, mus, As)*nus.reshape(-1, mus.shape[0]) 
    if ind is not None:
        return XX[:, 0], XX[:, 1], Z[:, ind]
    return XX[:, 0], XX[:, 1], Z

def get_u_v_for_func_above_threshold(u: ArrayNx1[np.float32], v: ArrayNx1[np.float32], func_values: ArrayNx1[np.float32], threshold: float = 0.05) -> Tuple[ArrayNx1[np.float32], ArrayNx1[np.float32]]:
    mask = func_values > threshold*np.abs(func_values).max()
    return u[mask], v[mask]
#%%
block_size= 8
sigmas = np.stack([np.array([[a, 0], [1, a]]) for a in [4, 3, 2, 1]])
mus = np.stack([[a*block_size, b*block_size] for a,b in [[0.3, 0.6], [0.9, 0.1], [0.6, 0.3], [0.1, 0.9]]])
nus = np.array([0.8, 0.6, 0.4, 0.2])
# gates_lims = lambda ind: get_u_v_for_func_above_threshold(*get_values_points_gates(mus, sigmas, nus, (-1, block_size+1), (-1, block_size+1), (32, 32), ind))
# kernel_lims = lambda ind: get_u_v_for_func_above_threshold(*get_values_points_kernels(mus[ind], sigmas[ind], nus[ind], (-1, block_size+1), (-1, block_size+1), (32, 32)))
#%%
%%manim -ql ThreeDSurfacePlot
class HackedSurface(Surface):
    def set_u_v_values_func(self, func):
        _get_u_values_and_v_values = func
        return self

class ThreeDSurfacePlot(ThreeDScene):
    def construct(self):
        resolution_fa = 20
        self.set_camera_orientation(phi=75 * DEGREES, theta=-10 * DEGREES, zoom=0.5, frame_center=(0, 0, -1))
        axes = ThreeDAxes()
        axes.animate()
        colors = color_gradient([RED, GREEN, BLUE], len(mus))

        kernel_lims = lambda ind: get_u_v_for_func_above_threshold(*get_values_points_kernels(mus[ind], sigmas[ind], 1, (-1, block_size+1), (-1, block_size+1), (32, 32)))
        smoe_kernels_no_nu = [HackedSurface(
            lambda u, v: axes.c2p(*make_plotting_function_kernel(mu, sig, nu)(u, v)),
            resolution=(resolution_fa, resolution_fa),
            u_range=[mu[0] - 3*(1/sig[0][0]), mu[0] + 3*(1/sig[0][0])],
            v_range=[mu[1] - 3*(1/sig[1][1]), mu[1] + 3*(1/sig[1][1])],
            fill_color=c,
            fill_opacity=0.3,
            stroke_color=c, 
            checkerboard_colors=None
        ).set_u_v_values_func(lambda: kernel_lims(i)) for i, (sig, mu, nu, c) in enumerate(zip(sigmas, mus, np.ones(len(mus)), colors))]
        
        kernel_lims = lambda ind: get_u_v_for_func_above_threshold(*get_values_points_kernels(mus[ind], sigmas[ind], nus[ind], (-1, block_size+1), (-1, block_size+1), (32, 32)))
        smoe_kernels_with_nu = [HackedSurface(
            lambda u, v: axes.c2p(*make_plotting_function_kernel(mu, sig, nu)(u, v)),
            resolution=(resolution_fa, resolution_fa),
            u_range=[mu[0] - 3*(1/sig[0][0]), mu[0] + 3*(1/sig[0][0])],
            v_range=[mu[1] - 3*(1/sig[1][1]), mu[1] + 3*(1/sig[1][1])],
            fill_color=c,
            fill_opacity=0.3,
            stroke_color=c,
            checkerboard_colors=None
        ).set_u_v_values_func(lambda: kernel_lims(i)) for i, (sig, mu, nu, c) in enumerate(zip(sigmas, mus, nus, colors))]
        
        gates_lims = lambda ind: get_u_v_for_func_above_threshold(*get_values_points_gates(mus, sigmas, np.ones(len(mus)), (-1, block_size+1), (-1, block_size+1), (32, 32), ind))
        smoe_gates_no_nus = [HackedSurface(
            lambda u, v: axes.c2p(*make_plotting_function_gating(mus, sigmas, np.ones(len(mus)), i)(u, v)),
            resolution=(resolution_fa, resolution_fa),
            # u_range=[mus[i][0] - 3*(1/sigmas[i][0][0]), mus[i][0] + 3*(1/sigmas[i][0][0])],
            # v_range=[mus[i][1] - 3*(1/sigmas[i][1][1]), mus[i][1] + 3*(1/sigmas[i][1][1])],
            u_range=[-1, block_size+1],
            v_range=[-1, block_size+1],
            fill_color=c,
            fill_opacity=0.3,
            stroke_color=c,
            checkerboard_colors=None
        ).set_u_v_values_func(lambda: gates_lims(i)) for i, c in zip(range(len(mus)), colors)]
        
        gates_lims = lambda ind: get_u_v_for_func_above_threshold(*get_values_points_gates(mus, sigmas, nus, (-1, block_size+1), (-1, block_size+1), (32, 32), ind))
        smoe_gates_with_nus = [HackedSurface(
            lambda u, v: axes.c2p(*make_plotting_function_gating(mus, sigmas, nus, i)(u, v)),
            resolution=(resolution_fa, resolution_fa),
            # u_range=[mus[i][0] - 3*(1/sigmas[i][0][0]), mus[i][0] + 3*(1/sigmas[i][0][0])],
            # v_range=[mus[i][1] - 3*(1/sigmas[i][1][1]), mus[i][1] + 3*(1/sigmas[i][1][1])],
            u_range=[-1, block_size+1],
            v_range=[-1, block_size+1],
            fill_color=c,
            fill_opacity=0.3,
            stroke_color=c,
            checkerboard_colors=None
        ).set_u_v_values_func(lambda: gates_lims(i)) for i, c in zip(range(len(mus)), colors)]
        
        self.add(axes)
        self.begin_ambient_camera_rotation(rate=0.075)
        kernel_equation = MathTex(
            r"\mathcal{K} (\mathbf{x}, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) = \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\right)",
            substrings_to_isolate=[r"\mathcal{K}", r"\boldsymbol{\mu}", r"\boldsymbol{\Sigma}"]
        ).align_on_border(UL)
        self.add_fixed_in_frame_mobjects(kernel_equation)
        self.play(
            LaggedStart(
                *(FadeIn(kernel) for kernel in smoe_kernels_no_nu),
                lag_ratio=0.25,
            ),
            run_time=2,
        )
        self.wait(1)
        # gate_equation = MathTex(
        #     r"g_k(\mathbf{x}) = \frac{\mathcal{K} (\mathbf{x}, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{k=1}^{K} \mathcal{K} (\mathbf{x}, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}",
        #     substrings_to_isolate=[r"\mathcal{K}", r"\boldsymbol{\mu}", r"\boldsymbol{\Sigma}"]
        # ) 
        # self.play(FadeOut(kernel_equation))
        # self.add_fixed_in_frame_mobjects(gate_equation)
        # self.play(Transform(kernel_equation, gate_equation))
        self.play(
            LaggedStart(
                *(Transform(kernel, gate) for kernel, gate in zip(smoe_kernels_no_nu, smoe_gates_no_nus)),
                lag_ratio=0.25,
            ),
            run_time=3,
            )
        self.wait(1)
        # smoe_equation = MathTex(
        #     r"m_k(\mathbf{x}) \cdot g_k(\mathbf{x})",
        #     substrings_to_isolate=[r"\mathcal{K}", r"\boldsymbol{\mu}", r"\boldsymbol{\Sigma}"]
        # )
        # self.play(Transform(kernel_equation, smoe_equation))
        self.play(
            LaggedStart(
                *(Transform(no_nu, yes_nu) for no_nu, yes_nu in zip(smoe_kernels_no_nu, smoe_gates_with_nus)),
                lag_ratio=0.25,
            ),
            run_time=3,
        )
        self.wait(1) 
        # self.play(
        #     AnimationGroup(
        #         Transform(no_nu, gate)
        #         for no_nu, gate in zip(smoe_kernels_no_nu, smoe_gates)
        #     ),
        #     run_time=1, lag_ratio=5,
        # )
        # fade_in_curves = AnimationGroup(*[FadeIn(plane) for plane in smoe_kernels])
        # self.play(fade_in_curves, run_time=2)
        # self.wait(1)
        self.stop_ambient_camera_rotation()
        self.interactive_embed()
# %%
