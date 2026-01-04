# Built-in libraries
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional

# Third-party libraries
import numpy as np
import numpy.typing as npt
from scipy.sparse import csc_array
from scipy.sparse.linalg import spsolve

# Local libraries
from .model import Structure
from .formulation import global_analysis
from .stability import buckling_analysis
from .visualization import AnalysisType, SolverVisualizer
from SimuFrame.utils.helpers import (
    get_local_displacements,
    check_convergence,
)


@dataclass
class NewtonRaphsonParams:
    """Newton-Raphson parameters."""

    initial_step: int = 15
    max_iter: int = 100
    iter_ideal: int = 5
    increase_factor: float = 1.20
    cut_back_slow_convergence: float = 0.75
    cut_back_divergence: float = 0.5
    max_cut_back: int = 8
    abs_tol: float = 1e-8
    rel_tol: float = 1e-8
    close_delay_sec: float = 0.5


@dataclass
class ArcLengthParams:
    """Arc-length parameters."""

    lmbda0: float = 0.1
    max_iter: int = 25
    allow_lambda_exceed: bool = False
    max_lambda: float = 2.0
    psi: float = 0.0
    max_reducoes: int = 8
    abs_tol: float = 1e-8
    rel_tol: float = 1e-8
    close_delay_sec: float = 0.5


@dataclass
class IncrementState:
    """State of an increment."""

    d: np.ndarray
    Fint: np.ndarray
    Kt: csc_array
    λ: float
    Δλ: float
    Ra: np.ndarray
    fe: np.ndarray
    de: np.ndarray


@dataclass
class ConvergenceData:
    """Data for convergence analysis."""

    increments: List[Dict] = field(default_factory=list)
    iteration_details: List[Dict] = field(default_factory=list)
    lambda_history: List[float] = field(default_factory=lambda: [0.0])
    max_displ_history: List[float] = field(default_factory=lambda: [0.0])
    rejected_points: Dict[str, List] = field(
        default_factory=lambda: {"displ": [], "lambda": []}
    )
    total_increments: int = 0
    accepted_increments: int = 0
    rejected_increments: int = 0
    final_lambda: float = 0.0
    max_displacement: float = 0.0
    converged: bool = False

class ArcLengthSolver:
    """Arc-length solver."""

    def __init__(
        self,
        global_force_vector: npt.NDArray[np.float64],
        global_elastic_stiffness: csc_array,
        structure: Structure,
        properties: Dict[str, Any],
        num_dofs: int,
        free_dofs_mask: npt.NDArray[np.bool_],
        element_dofs: npt.NDArray[np.integer],
        transformation_matrices: npt.NDArray[np.float64],
        nonlinear: bool,
        params: ArcLengthParams,
    ):
        self.F = global_force_vector
        self.structure = structure
        self.props = properties
        self.num_dofs = num_dofs
        self.free_dofs = free_dofs_mask
        self.el_dofs = element_dofs
        self.T = transformation_matrices
        self.nonlinear = nonlinear
        self.params = params

        # Extrapolation parameters (n-1 and n)
        self.d_n = np.zeros_like(global_force_vector)  # d at step n (last converged)
        self.d_n_1 = np.zeros_like(global_force_vector)  # d at step n-1 (second last converged)
        self.λ_n = 0.0  # λ at step n
        self.λ_n_1 = 0.0  # λ at step n-1

        # FF = F_ext^T * F_ext used in formulas (scalars)
        self.F_ext_vec = self.F.copy()
        self.FF = float(np.dot(self.F.T, self.F)[0, 0])

        # Initial state
        self.state = IncrementState(
            λ=0.0,
            Δλ=0.0,
            Kt=global_elastic_stiffness.copy(),
            d=np.zeros_like(global_force_vector),
            Fint=np.zeros_like(global_force_vector),
            Ra=np.zeros((0, 0)),
            fe=np.zeros((0, 0)),
            de=np.zeros((0, 0)),
        )

        # Initialize histories for both force x displacement and convergence data
        self.history = [(0.0, np.zeros((num_dofs, 1)), np.zeros((num_dofs, 1)))]
        self.convergence_data = ConvergenceData()

        # Visualizer (only if nonlinear = True)
        self.visualizer = SolverVisualizer(AnalysisType.ARC_LENGTH, show_window=nonlinear)

        # Control flags
        self.counter = 0
        self.converged = True
        self.converged_prev = True

    def _save_state(self) -> IncrementState:
        """Saves the current state."""
        return IncrementState(
            λ=self.state.λ,
            Δλ=self.state.Δλ,
            Kt=self.state.Kt.copy(),
            d=self.state.d.copy(),
            Fint=self.state.Fint.copy(),
            Ra=self.state.Ra.copy() if self.state.Ra.size > 0 else self.state.Ra,
            fe=self.state.fe.copy() if self.state.fe.size > 0 else self.state.fe,
            de=self.state.de.copy() if self.state.de.size > 0 else self.state.de,
        )

    def _restore_state(self, backup: IncrementState):
        """Restore the last converged state."""
        self.state = copy.deepcopy(backup)

    def _initial_step(self) -> bool:
        """Initial step with Newton-Raphson (force control)."""
        # Set the initial load step
        self.state.λ = getattr(self.params, "lmbda0", 1 / 100)

        iter = 0
        norm0 = 1.0

        while iter < self.params.max_iter:
            # Residual forces: : R = λF - Fint
            R = self.state.λ * self.F - self.state.Fint
            norm = np.linalg.norm(R)

            # Update the initial norm of the increment and calcuate the relative norm
            norm0 = max(norm, 1e-14)
            rel_norm = norm / norm0

            print(
                f"  Iter {iter + 1}: |R| = {norm:.4e}, |R|/|R0| = {rel_norm:.4e}"
            )

            # Verify convergence
            if norm < self.params.abs_tol or rel_norm < self.params.rel_tol:
                # Calculate Δs based on the converged solution
                self.Δs = np.sqrt(
                    np.dot(self.state.d.T, self.state.d)[0, 0]
                    + self.params.psi * self.state.λ**2 * self.FF
                )

                # Set the arc-length limits based on the converged solution
                self.Δs_n = self.Δs
                self.Δs_max = self.Δs * 2
                self.Δs_min = self.Δs / 1024.0

                # Update the counter (steps)
                self.counter = 1

                return True

            # Solve for displacement increment Δd
            Δd = spsolve(self.state.Kt, R).reshape(-1, 1)

            # Update the displacement vector
            self.state.d += Δd

            # Update tangent stiffness and internal forces
            self.state.Kt, self.state.Fint, self.state.Ra, self.state.fe, self.state.de = global_analysis(
                self.state.d, self.structure, self.props, self.num_dofs,
                self.free_dofs, self.el_dofs, self.T, self.nonlinear,
            )

            iter += 1

        return False

    def _compute_predictor(self) -> Tuple[np.ndarray, float]:
        """Compute the predictor step."""
        if self.counter == 1:
            # First step after the Newton-Raphson
            self.d_n = np.zeros_like(self.state.d)
            self.d_n_1 = np.zeros_like(self.state.d)
            self.λ_n = 0.0
            self.λ_n_1 = 0.0

        else:
            # Linear extrapolation: u_pred = (1+α)*u_n - α*u_{n-1}
            alpha = self.Δs / self.Δs_n
            d_pred = (1 + alpha) * self.d_n - alpha * self.d_n_1
            λ_pred = (1 + alpha) * self.λ_n - alpha * self.λ_n_1

            # Apply predictor
            self.state.d = d_pred.copy()
            self.state.λ = λ_pred

        # Increment since last converged step
        Δd = self.state.d - self.d_n
        Δλ = self.state.λ - self.λ_n

        return Δd, Δλ

    def _compute_corrector(
        self, Δd: np.ndarray, Δλ: float, du_1: np.ndarray, du_2: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Compute the corrector step."""
        # Terms of the formula (Kadapa)
        a = 2.0 * Δd
        b = 2.0 * self.params.psi * Δλ * self.FF
        A = np.dot(Δd.T, Δd)[0, 0] + self.params.psi * Δλ**2 * self.FF - self.Δs**2

        # Dot products
        a_dot_du1 = np.dot(a.T, du_1)[0, 0]
        a_dot_du2 = np.dot(a.T, du_2)[0, 0]

        # Compute dlmbda
        dlmbda = (a_dot_du2 - A) / (b + a_dot_du1)

        # Compute du
        du = -du_2 + dlmbda * du_1

        return dlmbda, du

    def _arc_length_iteration(
        self, Δd: np.ndarray, Δλ: float
    ) -> Tuple[bool, int, float]:
        """Compute Arc-Length iterations with corrector."""
        converged = False
        iter = 0
        norm = 1.0
        norm0 = None

        while iter < self.params.max_iter:
            # Residue: R = λ*F - Fint
            R = self.state.Fint - self.state.λ * self.F
            norm_R = np.linalg.norm(R)

            # Arc-Length restriction
            A = np.dot(Δd.T, Δd)[0, 0] + self.params.psi * (Δλ**2) * self.FF - self.Δs**2

            # Combined norm (total)
            norm = np.sqrt(norm_R**2 + A**2)

            # Define relative residual for arc-length solver iteration
            norm0 = max(norm, 1e-14)
            rel_norm = norm / norm0

            print(
                f"    Iter {iter}: |Total| = {norm:.4e}, |R| = {norm_R:.4e}, "
                f"|A| = {abs(A):.4e}, Rel = {rel_norm:.4e}"
            )

            # Check convergence
            if norm < self.params.abs_tol or rel_norm < self.params.rel_tol:
                converged = True
                break

            # Check divergence
            if norm > 1e10 or np.isnan(norm):
                print("    Divergência detectada!")
                return False, iter, float(norm)

            # Solve both linear systems (external and residual forces)
            try:
                du_1 = spsolve(self.state.Kt, self.F_ext_vec).reshape(-1, 1)
                du_2 = spsolve(self.state.Kt, R).reshape(-1, 1)
            except (RuntimeError, np.linalg.LinAlgError):
                print("    Error solving linear systems.")
                return False, iter, float(norm)

            # Compute corrector
            try:
                dlmbda, du = self._compute_corrector(Δd, Δλ, du_1, du_2)
            except ValueError as e:
                print(f"    Error in corrector step: {e}")
                return False, iter, float(norm)

            # Update total increments
            Δd += du
            Δλ += dlmbda

            # Update state
            self.state.d += du
            self.state.λ += dlmbda

            # Update tangent stiffness and internal forces
            self.state.Kt, self.state.Fint, self.state.Ra, self.state.fe, self.state.de = global_analysis(
                self.state.d, self.structure, self.props,  self.num_dofs,
                self.free_dofs, self.el_dofs, self.T, self.nonlinear
            )

            # Store iteration data
            self.convergence_data.iteration_details.append(
                {
                    "iteration": iter,
                    "lambda": float(self.state.λ),
                    "delta_lambda": float(Δλ),
                    "delta_lambda_iter": float(dlmbda),
                    "norm_R": float(norm_R),
                    "norm_A": float(abs(A)),
                    "norm": float(norm),
                    "norm_delta_d": float(np.linalg.norm(du)),
                    "converged": False,
                }
            )

            iter += 1

        # Mark last iteration as converged if true
        if converged and self.convergence_data.iteration_details:
            self.convergence_data.iteration_details[-1]["converged"] = True

        return converged, iter, float(norm)

    def _adapt_arc_length(self):
        """Adapt arc-length size based on the convergence of previous steps."""
        if self.converged:
            if self.converged_prev:
                # Double Δs if both converged
                new_Δs = min(2.0 * self.Δs, self.Δs_max)
                print(
                    f"  Adapt Δs: {self.Δs:.6e} → {new_Δs:.6e} (both converged)"
                )
                self.Δs = new_Δs
        else:
            if self.converged_prev:
                # Halve Δs if failed, but previous converged
                new_Δs = max(self.Δs / 2.0, self.Δs_min)
                print(
                    f"  Adapt Δs: {self.Δs:.6e} → {new_Δs:.6e} (÷2, previous converged)"
                )
                self.Δs = new_Δs
            else:
                # Divide by 4 if failed and previous also failed
                new_Δs = max(self.Δs / 4.0, self.Δs_min)
                print(
                    f"  Adapt Δs: {self.Δs:.6e} → {new_Δs:.6e} (÷4, both failed)"
                )
                self.Δs = new_Δs

    def solve(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List, np.ndarray, Optional[ConvergenceData]]:
        """Run the arc-length method."""
        # Verify if initial step converged
        if not self._initial_step():
            print("\nFAILURE in initial step!")
            return (
                self.state.d,
                self.state.de,
                self.state.fe,
                self.history,
                self.state.Ra,
                self.convergence_data,
            )

        # Store initial force and displacement
        force = np.zeros((self.num_dofs, 1))
        force[self.free_dofs] = self.state.λ * self.F
        displ = np.zeros((self.num_dofs, 1))
        displ[self.free_dofs] = self.state.d
        self.history.append((float(self.state.λ), force, displ))

        # Store initial convergence data
        max_displ = np.max(np.abs(self.state.d))
        self.convergence_data.lambda_history.append(self.state.λ)
        self.convergence_data.max_displ_history.append(max_displ)
        self.convergence_data.accepted_increments += 1

        # Initialize flags
        success = False
        consecutive_failures = 0

        # Arc-Length main loop
        while True:
            print(f"\n{'=' * 70}")
            print(f"Arc-Length step {self.counter}, Δs = {self.Δs:.6e}")
            print(f"{'=' * 70}")

            # Backup previous state
            backup = self._save_state()
            backup_Δs = self.Δs

            # Predictor step
            Δd, Δλ = self._compute_predictor()

            # Recalculate tangent stiffness and internal forces
            self.state.Kt, self.state.Fint, *_ = global_analysis(
                self.state.d,
                self.structure,
                self.props,
                self.num_dofs,
                self.free_dofs,
                self.el_dofs,
                self.T,
                self.nonlinear,
            )

            # Update the previous converged flag
            self.converged_prev = self.converged
            self.converged = False

            # Arc-Length corrector step
            converged, iteracao, norm_total = self._arc_length_iteration(Δd, Δλ)

            # Success - accept increment
            if converged:
                self.converged = True
                self.convergence_data.accepted_increments += 1
                consecutive_failures = 0

                print(f"\n  ✓ CONVERGED in {iteracao} iterations")
                print(f"    λ = {self.state.λ:.6f}")

                # Save Δs used in this step before
                self.Δs_n = self.Δs

                # Update history: n-1 ← n, n ← current
                self.d_n_1 = self.d_n.copy()
                self.λ_n_1 = self.λ_n
                self.d_n = self.state.d.copy()
                self.λ_n = self.state.λ

                # Store force and displacement
                force = np.zeros((self.num_dofs, 1))
                force[self.free_dofs] = self.state.λ * self.F
                displ = np.zeros((self.num_dofs, 1))
                displ[self.free_dofs] = self.state.d
                self.history.append((float(self.state.λ), force, displ))

                # Store convergence data
                max_displ = np.max(np.abs(self.state.d))
                self.convergence_data.lambda_history.append(self.state.λ)
                self.convergence_data.max_displ_history.append(max_displ)

                # Update visualizer
                self.visualizer.update(
                    self.convergence_data.max_displ_history,
                    self.convergence_data.lambda_history,
                    self.counter,
                    iteracao,
                    self.state.λ,
                    max_displ,
                    self.Δs,
                    norm_total,
                    self.params.allow_lambda_exceed,
                    self.params.max_lambda,
                )

                # Verify convergence criteria
                if self.params.allow_lambda_exceed:
                    if self.state.λ >= self.params.max_lambda:
                        success = True
                        print(f"\n✓ λ_max = {self.params.max_lambda}")
                        break
                else:
                    if self.state.λ >= 0.999:
                        success = True
                        print("\n✓ λ ≈ 1.0")
                        break

                # Adapt Δs for next step
                self._adapt_arc_length()

                # Increment counter for next step
                self.counter += 1

            # Failure - restore state and reduce Δs
            else:
                print(f"\n  ✗ FAILED after {iter} iterations")
                self._restore_state(backup)
                self.Δs = backup_Δs
                self.converged = False
                consecutive_failures += 1

                if self.visualizer:
                    self.visualizer.show_failure()

                # Adapt Δs for next step
                self._adapt_arc_length()

                # Verify stopping condition
                if (
                    self.Δs <= self.Δs_min
                    or consecutive_failures > self.params.max_reducoes
                ):
                    print("\n✗ Iteration limit reached:")
                    print(f"  Δλ = {self.Δs:.6e} (minimum = {self.Δs_min:.6e})")
                    print(f"  Consecutive failures = {consecutive_failures}")
                    break

        # End visualizer
        self.visualizer.finalize(
            success,
            self.state.λ,
            self.convergence_data.total_increments,
            self.convergence_data.accepted_increments,
        )

        if self.visualizer.show_window:
            self.visualizer.wait_and_close(self.params.close_delay_sec)

        return (
            self.state.d,
            self.state.de,
            self.state.fe,
            self.history,
            self.state.Ra,
            self.convergence_data,
        )


class NewtonRaphsonSolver:
    def __init__(
        self,
        global_force_vector: npt.NDArray[np.float64],
        global_elastic_stiffness: csc_array,
        structure: Structure,
        properties: Dict[str, Any],
        num_dofs: int,
        free_dofs_mask: npt.NDArray[np.bool_],
        element_dofs: npt.NDArray[np.integer],
        transformation_matrices: npt.NDArray[np.float64],
        nonlinear: bool,
        params: NewtonRaphsonParams,
    ):
        self.F = global_force_vector
        self.structure = structure
        self.props = properties
        self.num_dofs = num_dofs
        self.free_dofs = free_dofs_mask
        self.el_dofs = element_dofs
        self.T = transformation_matrices
        self.nonlinear = nonlinear
        self.params = params

        # Initial state
        self.state = IncrementState(
            λ=0.0,
            Δλ=0.0,
            Kt=global_elastic_stiffness.copy(),
            d=np.zeros_like(global_force_vector),
            Fint=np.zeros_like(global_force_vector),
            Ra=np.zeros((0, 0)),
            fe=np.zeros((0, 0)),
            de=np.zeros((0, 0)),
        )

        # Load control parameters
        self.Δλ = 1.0 / params.initial_step
        self.Δλ_min = self.Δλ / (params.increase_factor**params.max_cut_back)

        # Output history
        self.f_vs_d = [(0.0, np.zeros((num_dofs, 1)), np.zeros((num_dofs, 1)))]
        self.convergence_data = ConvergenceData()

        # Visualizer (only if nonlinear=True)
        self.visualizer = SolverVisualizer(AnalysisType.NEWTON_RAPHSON, show_window=nonlinear)

    def _save_state(self) -> IncrementState:
        """Saves the current state."""
        return IncrementState(
            λ=self.state.λ,
            Δλ=self.state.Δλ,
            Kt=self.state.Kt.copy(),
            d=self.state.d.copy(),
            Fint=self.state.Fint.copy(),
            Ra=self.state.Ra.copy() if self.state.Ra.size > 0 else self.state.Ra,
            fe=self.state.fe.copy() if self.state.fe.size > 0 else self.state.fe,
            de=self.state.de.copy() if self.state.de.size > 0 else self.state.de,
        )

    def _restore_state(self, backup: IncrementState):
        """Restores a saved state."""
        self.state: IncrementState = copy.deepcopy(backup)

    def solve(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List, np.ndarray, Optional[ConvergenceData]]:
        step = 0
        success = False
        consecutive_failures = 0

        # Load control loop
        while self.state.λ < 1.0:
            step += 1
            self.convergence_data.total_increments += 1

            print(f"\n{'=' * 70}")
            print(f"Step {step}, Δλ = {self.Δλ:.6e}")
            print(f"{'=' * 70}")

            # Save state before step
            backup = self._save_state()

            # Limit load factor to not exceed 1.0
            lambda_target = min(self.state.λ + self.Δλ, 1.0)

            # Update load factor
            self.state.λ = lambda_target

            iter = 0
            norm = 1.0
            norm0 = 1.0
            converged = False

            # Newton-Raphson iterations
            while iter < self.params.max_iter:
                # Residual forces: : R = λF - Fint
                R = self.state.λ * self.F - self.state.Fint
                norm = np.linalg.norm(R)

                # Check divergence
                if norm > 1e10 or np.isnan(norm):
                    converged = False
                    break

                # Update the initial norm of the increment and calcuate the relative norm
                norm0 = max(norm, 1e-14)
                rel_norm = norm / norm0

                print(
                    f"  Iter {iter + 1}: |R| = {norm:.4e}, |R|/|R0| = {rel_norm:.4e}"
                )

                # # Check convergence
                # if norm < self.params.abs_tol or rel_norm < self.params.rel_tol:
                #     converged = True
                #     break

                # Solve for displacement increment Δd
                Δd = spsolve(self.state.Kt, R).reshape(-1, 1)

                # Update the displacement vector
                self.state.d += Δd

                # Check convergence based on displacement increment
                converged = check_convergence(
                    self.state.d, Δd, self.state.λ * self.F, R
                )

                # Exit if converged
                if converged:
                    break

                # Update tangent stiffness and internal forces
                self.state.Kt, self.state.Fint, self.state.Ra, self.state.fe, self.state.de = global_analysis(
                    self.state.d, self.structure, self.props, self.num_dofs,
                    self.free_dofs, self.el_dofs, self.T, self.nonlinear,
                )

                iter += 1

            # Success - accept increment
            if converged:
                self.converged = True
                self.convergence_data.accepted_increments += 1
                consecutive_failures = 0

                print(f"\n  ✓ CONVERGED in {iter + 1} iterations")
                print(f"    λ = {self.state.λ:.6f}")

                # Save the data in the history
                force = np.zeros((self.num_dofs, 1), dtype=float)
                force[self.free_dofs] = self.state.λ * self.F
                displ = np.zeros((self.num_dofs, 1), dtype=float)
                displ[self.free_dofs] = self.state.d
                self.f_vs_d.append((float(self.state.λ), force, displ))

                max_displ = np.max(np.abs(self.state.d))
                self.convergence_data.lambda_history.append(self.state.λ)
                self.convergence_data.max_displ_history.append(max_displ)

                # Update visualizer
                self.visualizer.update(
                    self.convergence_data.max_displ_history,
                    self.convergence_data.lambda_history,
                    step,
                    iter,
                    self.state.λ,
                    max_displ,
                    residue=float(norm),
                )

                # Step size adaptation
                if iter <= 3:
                    factor = 1.5
                elif iter <= 5:
                    factor = self.params.increase_factor
                elif iter <= 5 * 1.5:
                    factor = 1.0
                else:
                    factor = self.params.cut_back_slow_convergence

                # Adapt step size
                new_Δλ = np.clip(factor * self.Δλ, self.Δλ_min, 1.0)
                print(f"  Δλ adaptation: {self.Δλ:.6e} → {new_Δλ:.6e}")
                self.Δλ = new_Δλ

                # Check stopping criteria
                if self.state.λ >= 0.999:
                    success = True
                    print("\n✓ λ ≈ 1.0")
                    break

            # Failure - restore state and reduce Δλ
            else:
                print(f"\n  ✗ FAILED after {iter} iterations")
                self._restore_state(backup)
                self.Δλ *= self.params.cut_back_divergence
                self.converged = False
                consecutive_failures += 1

                if self.visualizer:
                    self.visualizer.show_failure()

                # Decrease step size
                step -= 1

                # Verify if Δλ became too small
                if (
                    self.Δλ <= self.Δλ_min
                    or consecutive_failures > self.params.max_cut_back
                ):
                    print("\n✗ Iteration limit reached:")
                    print(f"  Δλ = {self.Δλ:.6e} (minimum = {self.Δλ_min:.6e})")
                    print(f"  Consecutive failures = {consecutive_failures}")
                    break

        # End visualizer
        self.visualizer.finalize(
            success,
            self.state.λ,
            self.convergence_data.total_increments,
            self.convergence_data.accepted_increments,
        )
        self.visualizer.wait_and_close(self.params.close_delay_sec)

        return (
            self.state.d,
            self.state.de,
            self.state.fe,
            self.f_vs_d,
            self.state.Ra,
            self.convergence_data,
        )

# Type for the history (load factor, forces, displacements)
HistoryEntry = Tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]
HistoryType = Optional[List[HistoryEntry]]

# Type for the solver function
SolverType = Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    HistoryType,
    npt.NDArray[np.float64],
    Optional[ConvergenceData],
]

# Type for the solver results
SolverResults = Tuple[
    Dict[str, npt.NDArray[np.float64]],   # displacements
    Dict[str, npt.NDArray[np.float64]],   # forces
    HistoryType,                          # history
    Optional[ConvergenceData]             # convergence_data
]

def run_solver(
    analysis_type: AnalysisType,
    F: npt.NDArray[np.float64],
    KE: csc_array,
    estrutura: Structure,
    propriedades: Dict[str, Any],
    numDOF: int,
    GLL: npt.NDArray[np.bool_],
    GLe: npt.NDArray[np.integer],
    T: npt.NDArray[np.float64],
    nonlinear: bool,
    **solver_kwargs,
) -> SolverType:
    """Run solver based on the specified solver."""
    if analysis_type == AnalysisType.ARC_LENGTH:
        params = ArcLengthParams(**solver_kwargs)
        solver = ArcLengthSolver(
            F, KE, estrutura, propriedades, numDOF, GLL, GLe, T, nonlinear, params
        )

    elif analysis_type == AnalysisType.NEWTON_RAPHSON:
        params = NewtonRaphsonParams(**solver_kwargs)
        solver = NewtonRaphsonSolver(
            F, KE, estrutura, propriedades, numDOF, GLL, GLe, T, nonlinear, params
        )

    else:
        raise ValueError(f"Analysis not recognized: {analysis_type}")

    return solver.solve()

def solve_structure(
    structure: Structure,
    properties: dict,
    num_dofs: int,
    free_dofs_mask: npt.NDArray[np.bool_],
    element_dofs: npt.NDArray[np.int_],
    transformation_matrices: npt.NDArray[np.float64],
    global_elastic_stiffness: csc_array,
    element_elastic_stiffness: npt.NDArray[np.float64],
    global_force_vector: npt.NDArray[np.float64],
    equivalent_nodal_forces: npt.NDArray[np.float64]
) -> SolverResults:
    """
    Solve the structural system, whether it's a linear, nonlinear or buckling analysis.

    Args:
        structure (Structure): Instance of the Structure class.
        properties (dict): Dictionary containing element properties.
        num_dofs (int): Total number of degrees of freedom.
        free_dofs_mask (np.ndarray): Boolean array indicating free degrees of freedom.
        element_dofs (np.ndarray): Integer array of degrees of freedom associated with each element.
        transformation_matrices (np.ndarray): Array of rotation/transformation matrices.
        global_elastic_stiffness (np.ndarray): Global element stiffness matrix.
        element_elastic_stiffness (np.ndarray): Local element stiffness matrix.
        global_force_vector (np.ndarray): Global external force vector.
        equivalent_nodal_forces (np.ndarray): Vector of equivalent nodal forces.

    Returns:
        tuple:
            - displacements (dict): Dictionary of displacements (global and local).
            - forces (dict): Dictionary of forces (global and local).
            - history (dict): Dictionary of history data (force x displacement).
            - convergence_data (dict): Dictionary of convergence data history.
    """
    # Initial data
    T = transformation_matrices
    fq = equivalent_nodal_forces
    analysis = structure.analysis

    # Initialize variables
    forces: Dict[str, npt.NDArray[np.float64]] = {}
    displacements: Dict[str, npt.NDArray[np.float64]] = {}
    history: HistoryType = None
    convergence_data: Optional[ConvergenceData] = None

    # Apply boundary conditions to global force vector and get reactions
    F = global_force_vector[free_dofs_mask]
    Fr = global_force_vector[~free_dofs_mask]

    # Find DOFs with stiffness below the limit (unstable structure)
    diag_Ke = global_elastic_stiffness.diagonal()
    unstable_dofs = np.where(np.abs(diag_Ke) < 1e-9)[0]

    if unstable_dofs.size > 0:
        # UNSTABLE STRUCTURE!
        raise ValueError(
            f"ERROR: Unstable structure (rigid body instability).\n"
            f"DOFs with zero stiffness: {unstable_dofs}\n"
            f"Check boundary conditions and member releases."
        )

    # Solve the initial system of equations for the global displacements
    d = spsolve(global_elastic_stiffness, F).reshape(-1, 1)

    if np.isnan(d).any():
        # UNSTABLE STRUCTURE!
        raise ValueError(
            "ERROR: Unstable structure (rigid body instability).\n"
            "Resulting matrix is singular.\n"
            "Check boundary conditions and member releases."
        )

    # Local displacements vector, {dl}
    dl = get_local_displacements(num_dofs, free_dofs_mask, element_dofs, transformation_matrices, d)

    # Local forces vector, {fl}
    fl = element_elastic_stiffness @ dl - transformation_matrices @ equivalent_nodal_forces

    # Initialize the global displacements vector
    dg = np.zeros((num_dofs, 1))

    if analysis == "linear":
        # Compute displacements, local displacements, local forces, force-displacement history, reaction forces, and convergence data
        d, dl, fl, history, Re, convergence_data = run_solver(
            AnalysisType.NEWTON_RAPHSON,
            F, global_elastic_stiffness, structure, properties, num_dofs, free_dofs_mask, element_dofs, transformation_matrices,
            nonlinear=False, initial_step=1, abs_tol=1e-8, rel_tol=1e-8,
        )

        # Store displacements
        dg[free_dofs_mask] = d
        displacements["d"] = dg
        displacements["de"] = dl

        # Store forces
        forces["F"] = F
        forces["R"] = Re - Fr
        forces["fe"] = fl - T @ fq

    elif analysis == "nonlinear":
        # Compute displacements, local displacements, local forces, force-displacement history, reaction forces, and convergence data
        # d, dnl, fnl, f_vs_d, Ra, convergence_data = run_solver(
        #     AnalysisType.ARC_LENGTH,
        #     F, global_elastic_stiffness, structure, properties, num_dofs, free_dofs_mask, element_dofs, transformation_matrices,
        #     nonlinear=True, lmbda0=0.01, allow_lambda_exceed=False, max_lambda=2.0,
        #     psi=1.0, abs_tol=1e-6, rel_tol=1e-6,
        # )

        d, dnl, fnl, history, Re, convergence_data = run_solver(
            AnalysisType.NEWTON_RAPHSON,
            F, global_elastic_stiffness, structure, properties, num_dofs, free_dofs_mask, element_dofs, transformation_matrices,
            nonlinear=True, initial_step=25, abs_tol=1e-8, rel_tol=1e-8,
        )

        # Atribuir os deslocamentos
        dg[free_dofs_mask] = d
        displacements["d"] = dg
        displacements["de"] = dnl

        # Atribuir os esforços
        forces["F"] = F
        forces["R"] = Re - Fr
        forces["fe"] = fnl

    elif analysis == "buckling":
        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = buckling_analysis(
            structure, properties, num_dofs, element_dofs, free_dofs_mask, transformation_matrices, global_elastic_stiffness, fl
        )

        # Store eigenvalues and null history data
        forces["autovalores"] = eigvals
        history = None
        convergence_data = None

        # Store eigenvectors
        displacements["d"] = eigvecs
        displacements["de"] = T @ eigvecs[:, element_dofs]

    return displacements, forces, history, convergence_data
