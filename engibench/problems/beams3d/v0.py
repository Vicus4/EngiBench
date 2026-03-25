"""Structural 3D Beams Problem."""

import dataclasses
from dataclasses import dataclass, field
from typing import Annotated, Any

from gymnasium import spaces
import napari
import numpy as np
import numpy.typing as npt

from engibench.constraint import bounded, constraint, IMPL, THEORY
from engibench.core import ObjectiveDirection, OptiStep, Problem
from engibench.problems.beams3d.model import fem_model
from engibench.problems.beams3d.model.fem_model import FeaModel3D

NELX = NELY = NELZ = 64

# Default fixed elements (e.g., clamping the 4 bottom corners)
FIXED_ELEMENTS = np.zeros((NELX + 1, NELY + 1, NELZ + 1), dtype=int)
FIXED_ELEMENTS[0, 0, 0] = 1
FIXED_ELEMENTS[-1, 0, 0] = 1
FIXED_ELEMENTS[0, -1, 0] = 1
FIXED_ELEMENTS[-1, -1, 0] = 1

# Default force (Center of the top face)
FORCE_ELEMENTS_Z = np.zeros((NELX + 1, NELY + 1, NELZ + 1), dtype=int)
FORCE_ELEMENTS_Z[NELX // 2, NELY // 2, -1] = 1


class Beams3D(Problem[npt.NDArray]):
    """3D topology optimization problem for structural compliance only."""

    version = 1
    objectives: tuple[tuple[str, ObjectiveDirection], ...] = (
        ("structural_compliance", ObjectiveDirection.MINIMIZE),
        ("volume_fraction", ObjectiveDirection.MINIMIZE),
    )

    @dataclass
    class Conditions:
        """Conditions."""
        fixed_elements: Annotated[npt.NDArray[np.int64], bounded(lower=0.0, upper=1.0).category(THEORY)] = field(
            default_factory=lambda: FIXED_ELEMENTS
        )
        """Binary NxNxN array of the structurally fixed elements in the domain"""
        
        force_elements_z: Annotated[npt.NDArray[np.int64], bounded(lower=0.0, upper=1.0).category(THEORY)] = field(
            default_factory=lambda: FORCE_ELEMENTS_Z
        )
        """Binary NxNxN array specifying elements that have a vertical structural load (z-direction)"""
        
        volfrac: Annotated[float, bounded(lower=0.0, upper=1.0).category(THEORY)] = 0.3
        """Target volume fraction for the volume fraction constraint"""
        
        rmin: Annotated[
            float, bounded(lower=1.0).category(THEORY), bounded(lower=0.0, upper=3.0).warning().category(IMPL)
        ] = 1.5
        """Filter size used in the optimization routine"""
        
        penal: Annotated[
            float, bounded(lower=1.0).category(THEORY), bounded(lower=0.0, upper=10.0).warning().category(IMPL)
        ] = 3.0

    conditions = Conditions()
    design_space = spaces.Box(low=0.0, high=1.0, shape=(NELX, NELY, NELZ), dtype=np.float32)
    dataset_id = "IDEALLab/beams_3d_v0"
    container_id = None

    @dataclass
    class Config(Conditions):
        """Structured representation of configuration parameters for a numerical computation."""
        nelx: Annotated[int, bounded(lower=1).category(THEORY)] = NELX
        nely: Annotated[int, bounded(lower=1).category(THEORY)] = NELY
        nelz: Annotated[int, bounded(lower=1).category(THEORY)] = NELZ
        max_iter: int = fem_model.MAX_ITERATIONS

        @constraint
        @staticmethod
        def rmin_bound(rmin: float, nelx: int, nely: int, nelz: int) -> None:
            assert 0.0 < rmin <= max(nelx, nely, nelz), f"Params.rmin: {rmin} ∉ (0, max(nelx, nely, nelz)]"

        @constraint
        @staticmethod
        def bc_check(
            nelx: int,
            nely: int,
            nelz: int,
            fixed_elements: npt.NDArray[np.int64],
            force_elements_z: npt.NDArray[np.int64],
        ) -> None:
            assert fixed_elements.shape == (nelx + 1, nely + 1, nelz + 1), "Invalid shape for fixed_elements."
            assert force_elements_z.shape == (nelx + 1, nely + 1, nelz + 1), "Invalid shape for force_elements_z."

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)

    def _prepare_fem_dict(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Helper to merge config and inject dummy thermal variables for FeaModel3D compatibility."""
        boundary_dict = dataclasses.asdict(self.conditions)
        for key, value in (config or {}).items():
            if key in boundary_dict:
                boundary_dict[key] = np.array(value) if isinstance(value, list) else value
        
        # Inject dummy parameters to prevent the original FEM solver from crashing
        nelx = config.get("nelx", NELX) if config else NELX
        nely = config.get("nely", NELY) if config else NELY
        nelz = config.get("nelz", NELZ) if config else NELZ
        
        boundary_dict["force_elements_x"] = np.zeros((nelx + 1, nely + 1, nelz + 1), dtype=int)
        boundary_dict["force_elements_y"] = np.zeros((nelx + 1, nely + 1, nelz + 1), dtype=int)
        boundary_dict["heatsink_elements"] = np.zeros((nelx + 1, nely + 1, nelz + 1), dtype=int)
        boundary_dict["weight"] = 1.0  # 100% structural
        
        return boundary_dict

    def simulate(self, design: npt.NDArray, config: dict[str, Any] | None = None) -> npt.NDArray:
        boundary_dict = self._prepare_fem_dict(config)
        results = FeaModel3D(plot=False, eval_only=True).run(boundary_dict, x_init=design)
        return np.array([results["structural_compliance"], results["volume_fraction"]])

    def optimize(
        self, starting_point: npt.NDArray, config: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, list[OptiStep]]:
        boundary_dict = self._prepare_fem_dict(config)
        max_iter = (config or {}).get("max_iter", self.Config.max_iter)
        results = FeaModel3D(plot=False, eval_only=False, max_iter=max_iter).run(boundary_dict, x_init=starting_point)
        
        design = np.array(results["design"]).astype(np.float32)
        return design, results["opti_steps"]

    def render(self, design: np.ndarray, *, open_window: bool = False) -> np.ndarray:
        design = np.array(design)
        design = np.transpose(design, (2, 0, 1))

        viewer = napari.Viewer()
        viewer.add_image(design, name="rho", rendering="attenuated_mip")
        viewer.dims.ndisplay = 3
        if open_window:
            napari.run()
        return viewer.export_figure(flash=False)