import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Set up figure and axis
fig, ax = plt.subplots(figsize=(14, 6))

# ============================================================================
# 1D Lattice Setup
# ============================================================================
# Create a 1D lattice with atoms at regular spacing
lattice_spacing = 1.0  # Å
num_atoms = 5
equilibrium_positions = np.arange(num_atoms) * lattice_spacing

# Perturb one atom (atom at index 3) slightly to the right
perturbation_magnitude = 0.45  # Å
atom_positions = equilibrium_positions.copy()
perturbed_atom_idx = 2
atom_positions[perturbed_atom_idx] += perturbation_magnitude

# ============================================================================
# Electron Wavefunction Construction
# ============================================================================
# Create a position grid for the wavefunction
x = np.linspace(-1, num_atoms * lattice_spacing, 1000)

# Build electron wavefunction as sum of atomic orbitals (gaussians)
# This represents the electron density localized around each nucleus
sigma = 0.2  # Width of atomic orbitals (Å)
psi_squared = np.zeros_like(x)  # Probability density |ψ|²

for pos in atom_positions:
    # Gaussian orbital centered at each atom position
    psi_squared += np.exp(-((x - pos) ** 2) / (2 * sigma ** 2))

# Normalize
psi_squared /= np.max(psi_squared)
psi_squared /= 3

# ============================================================================
# Plotting
# ============================================================================
# Plot 1: Electron wavefunction (|ψ|²)
ax.fill_between(x, 0, psi_squared, alpha=0.4, color='blue', label='e⁻ wavefunction |ψ|²')
ax.plot(x, psi_squared, color='blue', linewidth=2)

# Plot 2: Atomic positions (lattice)
# Equilibrium positions (dashed, faint)
ax.scatter(equilibrium_positions, np.zeros_like(equilibrium_positions), 
           s=150, marker='o', edgecolors='black', facecolors='none', 
            linewidths=1.5, alpha=0.5, label='Equilibrium positions')

# Current positions (solid)
colors = ['red' if i == perturbed_atom_idx else 'black' for i in range(num_atoms)]
sizes = [250 if i == perturbed_atom_idx else 150 for i in range(num_atoms)]
ax.scatter(atom_positions, np.zeros_like(atom_positions), 
           s=sizes, marker='o', c=colors, edgecolors='darkred' if colors[0]=='red' else 'black', 
           linewidths=2, zorder=5, label='Atomic nuclei')

# Plot 3: Hellmann-Feynman restoring force
# The force on a perturbed atom is proportional to the gradient of electron density
# Calculate gradient of electron density at perturbed atom position
psi_gradient = np.gradient(psi_squared, x)
force_position_idx = np.argmin(np.abs(x - atom_positions[perturbed_atom_idx]))
force_direction = -psi_gradient[force_position_idx]  # Restoring force opposes perturbation

# Normalize force arrow for visibility
force_scale = 3.0
force_arrow_length = force_scale * np.abs(force_direction) / np.max(np.abs(psi_gradient))

# Draw restoring force arrow
arrow_y = -0.00  # Position arrow below the lattice
ax.arrow(atom_positions[perturbed_atom_idx], arrow_y, 
         -force_arrow_length * 0.4, 0,  # Points left (back toward equilibrium)
         head_width=0.02, head_length=0.15, fc='darkred', ec='darkred', 
         linewidth=2.5, zorder=4, label='Hellmann-Feynman force')

# Add annotation for the force
ax.text(atom_positions[perturbed_atom_idx] - 0.15, arrow_y - 0.08, 
        'Restoring\nforce', fontsize=11, color='darkred', fontweight='bold',
        ha='center', style='italic')

# Add perturbation annotation
ax.annotate('', xy=(atom_positions[perturbed_atom_idx], 0.05), 
            xytext=(equilibrium_positions[perturbed_atom_idx], 0.05),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text((atom_positions[perturbed_atom_idx] + equilibrium_positions[perturbed_atom_idx]) / 2, 
        0.09, 'perturbation', fontsize=10, color='green', ha='center', fontweight='bold')

# ============================================================================
# Formatting
# ============================================================================
ax.set_xlabel('Position (Å)', fontsize=13, fontweight='bold')
ax.set_ylabel('Probability Density / Amplitude', fontsize=13, fontweight='bold')

ax.set_ylim(-0.15, 0.4)
ax.set_xlim(-0.8, num_atoms * lattice_spacing )

ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=11, loc='upper right', framealpha=0.95)


plt.tight_layout()
plt.savefig('../figures/hellmann_feynman_forces.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'hellmann_feynman_forces.png'")
#plt.show()