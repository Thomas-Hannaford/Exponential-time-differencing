import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

class ODESolver:
    """
    Base class for numerical ODE solvers.
    Solves systems of ODEs on the form: u' = f(u, t), u(0) = U0
    
    This class provides the framework for time-stepping methods but leaves
    the actual time-stepping implementation to child classes via the advance() method.
    """

    def __init__(self, f):
        """Initialize solver with right-hand side function f(u, t)"""
        self.f = f

    def set_initial_conditions(self, U0):
        """
        Set initial conditions for the ODE system.
        
        Handles both scalar ODEs and systems of ODEs by checking the type of U0.
        For systems, stores the number of equations in self.number_of_eqns.
        """
        if isinstance(U0, (int, float)):
            # Scalar ODE case
            self.number_of_eqns = 1
            U0 = float(U0)
        else:
            # System of ODEs case
            U0 = np.asarray(U0)
            self.number_of_eqns = U0.size
        self.U0 = U0

    def solve(self, time_points):
        """
        Solve the ODE over the given time points.
        
        Parameters:
        - time_points: array of time points where solution is computed
        
        Returns:
        - u_vec: array containing solution at each time point
        - t: array of time points (same as input)
        """
        self.t = np.asarray(time_points)
        n = self.t.size

        self.u_vec = np.zeros((n, self.number_of_eqns))
        self.u_vec[0, :] = self.U0

        # Time-stepping loop
        for i in range(n - 1):
            self.i = i
            self.u_vec[i + 1] = self.advance()

        return self.u_vec, self.t

    def advance(self):
        """Advance solution one time step (to be implemented by subclasses)"""
        raise NotImplementedError


class ETD1(ODESolver):
    """
    First-order Exponential Time Differencing (ETD1) method.
    
    Implements the ETD1 scheme for stiff ODEs with linear/nonlinear splitting.
    The method is exact for the linear part and first-order accurate for the nonlinear part.
    """

    def __init__(self, f, L):
        """
        Initialize ETD1 solver.
        
        Parameters:
        - f: nonlinear function f(u,t)
        - L: linear operator matrix
        """
        super().__init__(f)
        self.L = np.asarray(L)
        self.L_inv = np.linalg.inv(self.L)  # Precompute inverse for efficiency

    def advance(self):
        """Perform one time step using ETD1 scheme"""
        u_vec, f, i, t = self.u_vec, self.f, self.i, self.t
        dt = t[i + 1] - t[i]

        # Compute matrix exponential for linear part
        exp_Lh = expm(self.L * dt)

        # Evaluate nonlinear part at current time
        F_n = f(u_vec[i, :])[1:]  # Extract nonlinear components for a and b

        # ETD1 update for a and b components
        ab_next = exp_Lh @ u_vec[i, 1:] + (exp_Lh - np.eye(2)) @ self.L_inv @ F_n
        
        # Simple forward Euler update for u component
        u_next = u_vec[i, 0] + dt * f(u_vec[i, :])[0]

        return np.array([u_next, ab_next[0], ab_next[1]])


class ETD2(ODESolver):
    """
    Second-order Exponential Time Differencing (ETD2) method.
    
    Implements a second-order accurate ETD scheme using information from
    two previous time steps for better accuracy on stiff problems.
    """

    def __init__(self, f, L):
        """Initialize with nonlinear function and linear operator"""
        super().__init__(f)
        self.L = np.asarray(L)
        self.L_inv = np.linalg.inv(self.L)
        self.L2_inv = np.linalg.inv(self.L @ self.L)  # Precompute for efficiency

    def advance(self):
        """Perform one time step using second-order ETD scheme"""
        u_vec, f, i, t = self.u_vec, self.f, self.i, self.t
        dt = t[i + 1] - t[i]

        exp_Lh = expm(self.L * dt)  # Matrix exponential

        # Get nonlinear terms at current and previous steps
        F_n = f(u_vec[i, :])[1:]
        F_n_minus_1 = f(u_vec[i - 1, :])[1:] if i > 0 else np.zeros_like(F_n)

        # ETD2 update for a and b components
        if i == 0:
            # First step falls back to ETD1
            ab_next = exp_Lh @ u_vec[i, 1:] + (exp_Lh - np.eye(2)) @ self.L_inv @ F_n
        else:
            # Full second-order update
            ab_next = (
                exp_Lh @ u_vec[i, 1:]
                + ((np.eye(2) + self.L * dt) @ exp_Lh - np.eye(2) - 2 * self.L * dt) @ self.L2_inv @ F_n / dt
                + (-exp_Lh + np.eye(2) + self.L * dt) @ self.L2_inv @ F_n_minus_1 / dt
            )
        
        # Second-order Adams-Bashforth for u component
        if i == 0:
            u_next = u_vec[i, 0] + dt * f(u_vec[i, :])[0]
        else:
            u_next = u_vec[i, 0] + dt * (1.5 * f(u_vec[i, :])[0] - 0.5 * f(u_vec[i - 1, :])[0])

        return np.array([u_next, ab_next[0], ab_next[1]])


class ETD2RK(ODESolver):
    """
    Second-order Exponential Time Differencing Runge-Kutta (ETD2RK) method.
    
    Implements a second-order accurate ETD scheme using a predictor-corrector
    (Runge-Kutta) approach for better stability and accuracy.
    """

    def __init__(self, f, L):
        """Initialize with nonlinear function and linear operator"""
        super().__init__(f)
        self.L = np.asarray(L)
        self.L_inv = np.linalg.inv(self.L)
        self.L2_inv = np.linalg.inv(self.L @ self.L)

    def advance(self):
        """Perform one time step using ETD2RK scheme"""
        u_vec, f, i, t = self.u_vec, self.f, self.i, self.t
        dt = t[i + 1] - t[i]

        # Precompute matrix exponentials
        exp_Lh = expm(self.L * dt)
        exp_Lh_divide_2 = expm((self.L * dt) / 2)

        # Current nonlinear terms
        F_n = f(u_vec[i, :])[1:]

        # Predictor step (ETD1-like)
        ab_predict = exp_Lh @ u_vec[i, 1:] + self.L_inv @ (exp_Lh - np.eye(2)) @ F_n
        u_predict = u_vec[i, 0] + dt * f(u_vec[i, :])[0]

        # Evaluate nonlinear terms at predicted state
        F_predict = f(np.array([u_predict, ab_predict[0], ab_predict[1]]))[1:]

        # Corrector step (second-order)
        ab_next = (
            exp_Lh @ u_vec[i, 1:]
            + self.L_inv @ (exp_Lh - np.eye(2)) @ F_n
            + self.L2_inv @ (exp_Lh - np.eye(2) - self.L * dt) @ (F_predict - F_n) / dt
        )

        # Trapezoidal rule for u component
        u_next = u_vec[i, 0] + dt * 0.5 * (
            f(u_vec[i, :])[0] + f(np.array([u_predict, ab_predict[0], ab_predict[1]]))[0]
        )

        return np.array([u_next, ab_next[0], ab_next[1]])


# Problem setup and numerical experiments
def F(u_vec):
    """
    Right-hand side function for the ODE system.
    
    Implements the nonlinear terms:
    - du/dt = a² - b²
    - da/dt = -u*b (nonlinear part)
    - db/dt = u*a (nonlinear part)
    """
    u, a, b = u_vec[0], u_vec[1], u_vec[2]
    # Numerical stability safeguards
    a = np.clip(a, -1e10, 1e10)
    b = np.clip(b, -1e10, 1e10)
    return np.array([a**2 - b**2, -u*b, u*a])


def global_error(u_vec_h, u_vec_h2):
    """
    Compute difference between solutions at different resolutions.
    
    Used to estimate convergence rates by comparing solutions computed
    with time steps h and h/2.
    """
    return np.linalg.norm(u_vec_h[-1] - u_vec_h2[-1])


def run_convergence_test():
    """
    Run convergence tests for all ETD methods.
    
    Computes solutions at different temporal resolutions and estimates
    convergence rates by comparing successive refinements.
    """
    # Linear operator for stiff part of the system
    C = 100  # Stiffness parameter
    L = np.array([[0, -C], [C, 0]])  # Skew-symmetric matrix for a and b components

    # Create solvers
    etd1_solver = ETD1(F, L)
    etd2_solver = ETD2(F, L)
    etd2rk_solver = ETD2RK(F, L)

    # Set initial conditions
    U0 = [-1, 1, 0]  # Initial state [u, a, b]
    for solver in [etd1_solver, etd2_solver, etd2rk_solver]:
        solver.set_initial_conditions(U0)

    # Time integration parameters
    t_final = 6 * np.pi  # Final integration time
    number_of_steps = [50000, 100000, 200000]  # Resolution study points

    # Test all methods
    methods = [etd1_solver, etd2_solver, etd2rk_solver]
    method_names = ["ETD1", "ETD2", "ETD2RK"]

    for method, name in zip(methods, method_names):
        print(f"\nMethod: {name}")
        final_solutions = []
        
        # Solve at different resolutions
        for ns in number_of_steps:
            time_points = np.linspace(0, t_final, ns + 1)
            u_vec, _ = method.solve(time_points)
            final_solutions.append(u_vec[-1])

        # Estimate convergence rates
        solution_differences = []
        convergence_orders = []
        
        for i in range(len(final_solutions) - 1):
            diff = np.linalg.norm(final_solutions[i] - final_solutions[i + 1])
            solution_differences.append(diff)
            
            if i > 0:  # Can compute order after at least two differences
                order = np.log(solution_differences[i-1]/solution_differences[i])/np.log(2)
                convergence_orders.append(order)
        
        print("Solution differences between resolutions:", solution_differences)
        print("Estimated convergence orders:", convergence_orders)


if __name__ == "__main__":
    run_convergence_test()
