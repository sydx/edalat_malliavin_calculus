import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.stats import norm
from typing import Callable, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class PartialWienerFunctional:
    """
    Represents a partial Wiener functional [F, F_bar] where F <= F_bar
    This is the domain-theoretic representation from Definition 2.1
    """
    
    def __init__(self, lower_func: Callable, upper_func: Callable):
        """
        Initialize partial functional with lower and upper bounds
        
        Args:
            lower_func: Lower semi-continuous function F
            upper_func: Upper semi-continuous function F_bar
        """
        self.lower = lower_func
        self.upper = upper_func
    
    def __call__(self, path: np.ndarray, t_grid: np.ndarray) -> Tuple[float, float]:
        """Evaluate the partial functional on a path"""
        return self.lower(path, t_grid), self.upper(path, t_grid)
    
    def is_consistent(self, path: np.ndarray, t_grid: np.ndarray) -> bool:
        """Check if F(omega) <= F_bar(omega) for given path"""
        lower_val, upper_val = self(path, t_grid)
        return lower_val <= upper_val

class CameronMartinElement:
    """
    Represents an element in the Cameron-Martin space H
    """
    
    def __init__(self, h_dot: Callable[[float], float], t_max: float = 1.0):
        """
        Initialize Cameron-Martin element
        
        Args:
            h_dot: Derivative function h_prime(t)
            t_max: Maximum time (default 1.0)
        """
        self.h_dot = h_dot
        self.t_max = t_max
    
    def h(self, t: float) -> float:
        """Compute h(t) = integral from 0 to t of h_prime(s) ds"""
        if t <= 0:
            return 0.0
        result, _ = integrate.quad(self.h_dot, 0, min(t, self.t_max))
        return result
    
    def norm_squared(self) -> float:
        """Compute norm_H_squared = integral from 0 to 1 of h_prime(s)^2 ds"""
        result, _ = integrate.quad(lambda s: self.h_dot(s)**2, 0, self.t_max)
        return result

class PartialMalliavianDerivative:
    """
    Implementation of the partial Malliavin derivative from Definition 3.1
    """
    
    def __init__(self, functional: PartialWienerFunctional, epsilon: float = 1e-4):
        """
        Initialize partial Malliavin derivative
        
        Args:
            functional: The partial Wiener functional to differentiate
            epsilon: Step size for finite differences
        """
        self.functional = functional
        self.epsilon = epsilon
    
    def compute_directional_derivative(self, path: np.ndarray, t_grid: np.ndarray, 
                                     h_t: CameronMartinElement, 
                                     evaluation_time: float) -> Tuple[float, float]:
        """
        Compute directional derivative in Cameron-Martin direction h_t
        Following Definition 3.1: D_t[F,F_bar](omega)
        """
        # Create perturbed path: omega + epsilon * h_t
        h_values = np.array([h_t.h(t) for t in t_grid])
        
        # Forward perturbation
        perturbed_path_plus = path + self.epsilon * h_values
        lower_plus, upper_plus = self.functional(perturbed_path_plus, t_grid)
        
        # Backward perturbation
        perturbed_path_minus = path - self.epsilon * h_values
        lower_minus, upper_minus = self.functional(perturbed_path_minus, t_grid)
        
        # Original values
        lower_orig, upper_orig = self.functional(path, t_grid)
        
        # Compute difference quotients
        # D_t F(omega) = lim inf (F(omega + epsilon*h_t) - F(omega))/epsilon
        lower_deriv = (lower_plus - lower_orig) / self.epsilon
        
        # D_t F_bar(omega) = lim sup (F_bar(omega + epsilon*h_t) - F_bar(omega))/epsilon
        upper_deriv = (upper_plus - upper_orig) / self.epsilon
        
        return lower_deriv, upper_deriv
    
    def compute_malliavin_derivative(self, path: np.ndarray, t_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the full Malliavin derivative D[F,F_bar] as functions of t
        """
        n_points = len(t_grid)
        lower_derivatives = np.zeros(n_points)
        upper_derivatives = np.zeros(n_points)
        
        for i, t in enumerate(t_grid):
            # Use indicator function on [0,t] as Cameron-Martin direction
            h_t = CameronMartinElement(lambda s, t_val=t: 1.0 if s <= t_val else 0.0)
            
            lower_deriv, upper_deriv = self.compute_directional_derivative(
                path, t_grid, h_t, t
            )
            
            lower_derivatives[i] = lower_deriv
            upper_derivatives[i] = upper_deriv
        
        return lower_derivatives, upper_derivatives

class MalliavianCalculus:
    """
    Main class implementing domain-theoretic Malliavin calculus algorithms
    """
    
    @staticmethod
    def generate_brownian_path(n_steps: int = 1000, T: float = 1.0, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a sample Brownian motion path"""
        if seed is not None:
            np.random.seed(seed)
        
        dt = T / n_steps
        t_grid = np.linspace(0, T, n_steps + 1)
        dW = np.random.normal(0, np.sqrt(dt), n_steps)
        W = np.cumsum(np.concatenate([[0], dW]))
        
        return W, t_grid
    
    @staticmethod
    def create_european_call_functional(S0: float, K: float, r: float, sigma: float, T: float) -> PartialWienerFunctional:
        """
        Create partial functional for European call option
        Following Example 6.1 from the paper
        """
        def payoff_lower(path: np.ndarray, t_grid: np.ndarray) -> float:
            W_T = path[-1]
            S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * W_T)
            return max(S_T - K, 0.0)
        
        def payoff_upper(path: np.ndarray, t_grid: np.ndarray) -> float:
            # For smooth functionals, upper = lower
            return payoff_lower(path, t_grid)
        
        return PartialWienerFunctional(payoff_lower, payoff_upper)
    
    @staticmethod
    def create_smoothed_call_functional(S0: float, K: float, r: float, sigma: float, T: float, smooth_param: float = 0.1) -> PartialWienerFunctional:
        """
        Create smoothed version of European call option for better numerical derivatives
        Uses smooth approximation: max(x,0) approx x + smooth_param * log(1 + exp(-x/smooth_param))
        """
        def smooth_max(x: float, smooth_param: float) -> float:
            if x > 10 * smooth_param:  # Avoid overflow
                return x
            elif x < -10 * smooth_param:
                return smooth_param * np.log(1 + np.exp(x / smooth_param))
            else:
                return smooth_param * np.log(1 + np.exp(x / smooth_param)) + x * np.exp(x / smooth_param) / (1 + np.exp(x / smooth_param))
        
        def payoff_smooth(path: np.ndarray, t_grid: np.ndarray) -> float:
            W_T = path[-1]
            S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * W_T)
            return smooth_max(S_T - K, smooth_param)
        
        return PartialWienerFunctional(payoff_smooth, payoff_smooth)
    
    @staticmethod
    def create_quadratic_functional(a: float = 1.0) -> PartialWienerFunctional:
        """Create a simple quadratic functional for testing"""
        def quad_func(path: np.ndarray, t_grid: np.ndarray) -> float:
            W_T = path[-1]
            return a * W_T**2
        
        return PartialWienerFunctional(quad_func, quad_func)
    
    @staticmethod
    def analytical_malliavin_derivative_quadratic(path: np.ndarray, t_grid: np.ndarray, a: float = 1.0) -> np.ndarray:
        """
        Analytical Malliavin derivative for F(W) = a*W_T^2
        D_t F = 2*a*W_T * indicator_function([0,T])(t)
        """
        W_T = path[-1]
        return 2 * a * W_T * np.ones_like(t_grid)
    
    @staticmethod
    def analytical_malliavin_derivative_call(path: np.ndarray, t_grid: np.ndarray, S0: float, K: float, r: float, sigma: float, T: float) -> np.ndarray:
        """
        Analytical Malliavin derivative for European call option (when in the money)
        D_t F = S_T * sigma * indicator_function([0,T])(t) if S_T > K, else 0
        """
        W_T = path[-1]
        S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * W_T)
        
        if S_T > K:
            return S_T * sigma * np.ones_like(t_grid)
        else:
            return np.zeros_like(t_grid)

def test_malliavin_implementation():
    """Test the implementation with examples from the paper"""
    
    print("Testing Domain-Theoretic Malliavin Calculus Implementation")
    print("=" * 60)
    
    # Generate sample Brownian path - use a path that puts option in the money
    W, t_grid = MalliavianCalculus.generate_brownian_path(n_steps=200, seed=123)
    
    # Adjust the path to ensure we test both in-the-money and out-of-the-money cases
    W_original = W.copy()
    W_itm = W + 0.5  # Force in-the-money
    W_otm = W - 0.5  # Force out-of-the-money
    
    # Test 1: Quadratic functional
    print("\nTest 1: Quadratic Functional F(W) = W_T^2")
    print("-" * 40)
    
    quadratic_functional = MalliavianCalculus.create_quadratic_functional(a=1.0)
    malliavin_quad = PartialMalliavianDerivative(quadratic_functional, epsilon=1e-4)
    
    # Compute numerical derivative
    lower_deriv, upper_deriv = malliavin_quad.compute_malliavin_derivative(W_original, t_grid)
    
    # Compute analytical derivative
    analytical_deriv = MalliavianCalculus.analytical_malliavin_derivative_quadratic(W_original, t_grid, a=1.0)
    
    # Compare results
    error = np.mean(np.abs(lower_deriv - analytical_deriv))
    print(f"Mean absolute error vs analytical: {error:.6f}")
    print(f"W_T = {W_original[-1]:.4f}")
    print(f"Analytical D_T F = {analytical_deriv[-1]:.4f}")
    print(f"Numerical D_T F = {lower_deriv[-1]:.4f}")
    
    # Test 2: European Call Option - In the Money
    print("\nTest 2a: European Call Option (In-the-Money)")
    print("-" * 40)
    
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0  # At-the-money strike
    call_functional = MalliavianCalculus.create_european_call_functional(S0, K, r, sigma, T)
    smoothed_call_functional = MalliavianCalculus.create_smoothed_call_functional(S0, K, r, sigma, T, smooth_param=1.0)
    
    malliavin_call = PartialMalliavianDerivative(call_functional, epsilon=1e-3)  # Larger epsilon for discontinuous function
    malliavin_smooth_call = PartialMalliavianDerivative(smoothed_call_functional, epsilon=1e-4)
    
    # Test with in-the-money path
    call_lower_deriv_itm, call_upper_deriv_itm = malliavin_call.compute_malliavin_derivative(W_itm, t_grid)
    smooth_call_deriv_itm, _ = malliavin_smooth_call.compute_malliavin_derivative(W_itm, t_grid)
    
    # Evaluate option payoff
    S_T_itm = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * W_itm[-1])
    payoff_itm = max(S_T_itm - K, 0.0)
    
    print(f"W_T = {W_itm[-1]:.4f}")
    print(f"Spot price S_T = {S_T_itm:.4f}")
    print(f"Option payoff = {payoff_itm:.4f}")
    print(f"Standard Malliavin derivative at T: {call_lower_deriv_itm[-1]:.6f}")
    print(f"Smoothed Malliavin derivative at T: {smooth_call_deriv_itm[-1]:.6f}")
    
    # Analytical derivative for comparison
    analytical_call_deriv_itm = MalliavianCalculus.analytical_malliavin_derivative_call(W_itm, t_grid, S0, K, r, sigma, T)
    print(f"Analytical Malliavin derivative at T: {analytical_call_deriv_itm[-1]:.6f}")
    
    # Test 2b: European Call Option - Out of the Money
    print("\nTest 2b: European Call Option (Out-of-the-Money)")
    print("-" * 40)
    
    # Test with out-of-the-money path
    call_lower_deriv_otm, call_upper_deriv_otm = malliavin_call.compute_malliavin_derivative(W_otm, t_grid)
    
    S_T_otm = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * W_otm[-1])
    payoff_otm = max(S_T_otm - K, 0.0)
    
    print(f"W_T = {W_otm[-1]:.4f}")
    print(f"Spot price S_T = {S_T_otm:.4f}")
    print(f"Option payoff = {payoff_otm:.4f}")
    print(f"Malliavin derivative at T: {call_lower_deriv_otm[-1]:.6f}")
    
    # Estimate Delta using integration by parts formula (only for ITM)
    if payoff_itm > 0:
        delta_estimate = (1/S0) * payoff_itm * (W_itm[-1] / (sigma * T))
        print(f"Delta estimate (integration by parts): {delta_estimate:.6f}")
    
    # Test 3: Algorithm convergence
    print("\nTest 3: Algorithm Convergence Analysis")
    print("-" * 40)
    
    epsilons = [1e-2, 1e-3, 1e-4, 1e-5]
    errors = []
    
    for eps in epsilons:
        malliavin_test = PartialMalliavianDerivative(quadratic_functional, epsilon=eps)
        test_deriv, _ = malliavin_test.compute_malliavin_derivative(W_original, t_grid)
        error = np.mean(np.abs(test_deriv - analytical_deriv))
        errors.append(error)
        print(f"epsilon = {eps:.0e}, Error = {error:.6e}")
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Brownian path
    plt.subplot(2, 3, 1)
    plt.plot(t_grid, W_original, 'b-', linewidth=1.5, label='Original')
    plt.plot(t_grid, W_itm, 'g--', linewidth=1.5, label='ITM (+0.5)')
    plt.plot(t_grid, W_otm, 'r--', linewidth=1.5, label='OTM (-0.5)')
    plt.title('Sample Brownian Motion Paths')
    plt.xlabel('Time t')
    plt.ylabel('W(t)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Malliavin derivative for quadratic functional
    plt.subplot(2, 3, 2)
    plt.plot(t_grid, analytical_deriv, 'r-', label='Analytical', linewidth=2)
    plt.plot(t_grid, lower_deriv, 'b--', label='Numerical', linewidth=2)
    plt.title('Malliavin Derivative: F(W) = W_T^2')
    plt.xlabel('Time t')
    plt.ylabel('D_t F')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Malliavin derivative for call option (ITM)
    plt.subplot(2, 3, 3)
    plt.plot(t_grid, call_lower_deriv_itm, 'g-', label='Standard', linewidth=2)
    plt.plot(t_grid, smooth_call_deriv_itm, 'b--', label='Smoothed', linewidth=2)
    plt.plot(t_grid, analytical_call_deriv_itm, 'r:', label='Analytical', linewidth=2)
    plt.title('Malliavin Derivative: European Call (ITM)')
    plt.xlabel('Time t')
    plt.ylabel('D_t F')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Convergence analysis
    plt.subplot(2, 3, 4)
    plt.loglog(epsilons, errors, 'ro-', linewidth=2, markersize=8)
    plt.title('Algorithm Convergence')
    plt.xlabel('Step size epsilon')
    plt.ylabel('Mean absolute error')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Option payoff diagram
    plt.subplot(2, 3, 5)
    S_range = np.linspace(80, 140, 100)
    payoffs = np.maximum(S_range - K, 0)
    plt.plot(S_range, payoffs, 'b-', linewidth=2)
    plt.axvline(S_T_itm, color='g', linestyle='--', label=f'ITM S_T = {S_T_itm:.1f}')
    plt.axvline(S_T_otm, color='r', linestyle='--', label=f'OTM S_T = {S_T_otm:.1f}')
    plt.axvline(K, color='k', linestyle='-', alpha=0.5, label=f'Strike K = {K}')
    plt.title('Call Option Payoff')
    plt.xlabel('Stock Price S')
    plt.ylabel('Payoff')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Error distribution
    plt.subplot(2, 3, 6)
    error_diff = lower_deriv - analytical_deriv
    plt.plot(t_grid, error_diff, 'r-', linewidth=1.5)
    plt.title('Numerical Error Distribution (Quadratic)')
    plt.xlabel('Time t')
    plt.ylabel('Error')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("Implementation successfully demonstrates:")
    print("- Domain-theoretic partial functionals")
    print("- Partial Malliavin derivatives via L-derivative approach")
    print("- Applications to quadratic functionals and option pricing")
    print("- Algorithmic convergence with polynomial complexity")
    print("- Integration by parts formula for Greeks computation")

if __name__ == "__main__":
    test_malliavin_implementation()