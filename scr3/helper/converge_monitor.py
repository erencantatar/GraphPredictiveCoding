import torch

class EnergyConvergence:
    def __init__(self, threshold=1e-5, patience=10):
        """
        Initialize the EnergyConvergence class.
        
        Args:
        - threshold (float): The minimum change in energy to continue updating.
        - patience (int): Number of consecutive steps the energy needs to be within the threshold for convergence.
        """
        self.threshold = threshold
        self.patience = patience
        self.energies = []

    def update_energy(self, energy):
        """
        Update the energy value and check for convergence.
        
        Args:
        - energy (float): The current energy value.
        
        Returns:
        - converged (bool): True if the energy has converged, False otherwise.
        """
        self.energies.append(energy)
        # Keep only the most recent `patience` energies
        if len(self.energies) > self.patience:
            self.energies.pop(0)
        
        # Check for convergence if we have enough energy values
        if len(self.energies) == self.patience:
            diffs = [abs(self.energies[i] - self.energies[i-1]) for i in range(1, len(self.energies))]
            
            # If all recent energy differences are below the threshold, consider it converged
            if all(diff < self.threshold for diff in diffs):
                return True
        return False


class AdaptiveEnergyConvergence:
    def __init__(self, initial_threshold=1e-5, patience=10, decay=0.9):
        self.initial_threshold = initial_threshold
        self.current_threshold = initial_threshold
        self.patience = patience
        self.decay = decay
        self.energies = []

    def update_energy(self, energy):

        if len(self.energies) > 0:
            # Calculate the delta between the last energy and the current energy
            delta = abs(energy - self.energies[-1])
            if delta < self.initial_threshold:
                # Stop immediately if the delta is smaller than the threshold
                return True
            
        self.energies.append(energy)
        # Keep only the most recent `patience` energies
        if len(self.energies) > self.patience:
            self.energies.pop(0)
        
        if len(self.energies) == self.patience:
            diffs = [abs(self.energies[i] - self.energies[i-1]) for i in range(1, len(self.energies))]
            
            # Adjust threshold based on recent diff trends
            avg_diff = sum(diffs) / len(diffs)
            self.current_threshold = max(self.current_threshold * self.decay, avg_diff * 0.1)

            if all(diff < self.current_threshold for diff in diffs):
                return True
        return False


class GradientEnergyConvergence:
    def __init__(self, threshold=1e-5, patience=10):
        self.threshold = threshold
        self.patience = patience
        self.gradients = []

    def update_gradient(self, gradient):
        # Assume gradient is a tensor; calculate the norm as a convergence metric
        grad_norm = gradient.norm().item()
        self.gradients.append(grad_norm)
        
        # Keep only the most recent `patience` gradients
        if len(self.gradients) > self.patience:
            self.gradients.pop(0)
        
        if len(self.gradients) == self.patience:
            if all(g < self.threshold for g in self.gradients):
                return True
        return False


class CombinedConvergence:
    def __init__(self, energy_threshold=1e-5, gradient_threshold=1e-5, patience=10):
        self.energy_tracker = AdaptiveEnergyConvergence(energy_threshold, patience)
        self.gradient_tracker = GradientEnergyConvergence(gradient_threshold, patience)

    def update(self, energy, gradient=None):
        energy_converged = self.energy_tracker.update_energy(energy)
        
        if gradient is not None:
            gradient_converged = self.gradient_tracker.update_gradient(gradient)
        else:
            gradient_converged = True  # If no gradient is provided, assume it's converged (e.g., during testing)
        
        return energy_converged and gradient_converged
