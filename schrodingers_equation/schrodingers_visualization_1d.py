import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os

def schrodinger_calculation(x, t, L, n, m):
    """Analytical solution for the 1D quantum particle in a box"""
    x = x.reshape((-1, 1))
    t = t.reshape((-1, 1))
    h_reduced = 1.0
    E_n = (h_reduced**2 * torch.pi**2 * n**2) / (2 * m * L**2)
    phi_n = torch.sin((n * torch.pi * x) / L) * (2 / L)**0.5
    omega = (E_n * t) / h_reduced
    real = phi_n * torch.cos(omega)
    imag = -phi_n * torch.sin(omega)
    return torch.cat([real, imag], dim=-1)

def schrodinger_superposition(x, t, L, n1, n2, c1, c2, m):
    """
    Analytical solution for superposition of two energy eigenstates
    ψ(x,t) = c1*ψ_n1(x,t) + c2*ψ_n2(x,t)
    where c1 and c2 are complex coefficients (normalized so |c1|² + |c2|² = 1)
    """
    x = x.reshape((-1, 1))
    t = t.reshape((-1, 1))
    h_reduced = 1.0
    
    # First state
    E_n1 = (h_reduced**2 * torch.pi**2 * n1**2) / (2 * m * L**2)
    phi_n1 = torch.sin((n1 * torch.pi * x) / L) * (2 / L)**0.5
    omega1 = (E_n1 * t) / h_reduced
    psi1_real = phi_n1 * torch.cos(omega1)
    psi1_imag = -phi_n1 * torch.sin(omega1)
    
    # Second state
    E_n2 = (h_reduced**2 * torch.pi**2 * n2**2) / (2 * m * L**2)
    phi_n2 = torch.sin((n2 * torch.pi * x) / L) * (2 / L)**0.5
    omega2 = (E_n2 * t) / h_reduced
    psi2_real = phi_n2 * torch.cos(omega2)
    psi2_imag = -phi_n2 * torch.sin(omega2)
    
    # Superposition (c1 and c2 are real for simplicity)
    # For equal superposition: c1 = c2 = 1/sqrt(2)
    real = c1 * psi1_real + c2 * psi2_real
    imag = c1 * psi1_imag + c2 * psi2_imag
    
    return torch.cat([real, imag], dim=-1)

class SchrodingerEquation1D(nn.Module):
    def __init__(self, L):
        super(SchrodingerEquation1D, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 2)
        )
        self.L = L

    def forward(self, x, t, n):
        omega = (torch.pi**2 * n**2) / (2.0 * self.L**2)
        t_encoded = torch.cat([t, torch.sin(omega * t), torch.cos(omega * t)], dim=-1)
        inputs = torch.cat([x, t_encoded, (n-1)/2.0], dim=-1)
        raw_output = self.net(inputs)
        psi0_real = torch.sin(n * torch.pi * x / self.L) * (2.0 / self.L)**0.5
        psi0_imag = torch.zeros_like(psi0_real)
        psi0 = torch.cat([psi0_real, psi0_imag], dim=-1)
        # Enforce boundary conditions
        bc = torch.sin(torch.pi * x / self.L)
        return psi0 + torch.tanh(3.0 * t)*bc*raw_output

def calculate_probability(psi):
    """Calculate probability density from wave function: |ψ|² = u² + v²"""
    return (psi[:, 0]**2 + psi[:, 1]**2)

def pinn_superposition(model, x, t, n1, n2, c1, c2):
    """
    Get PINN prediction for superposition of two energy eigenstates
    """
    n1_expanded = torch.full_like(x, n1)
    n2_expanded = torch.full_like(x, n2)
    
    # Get predictions for each state
    psi1 = model(x, t, n1_expanded)
    psi2 = model(x, t, n2_expanded)
    
    # Combine with coefficients
    psi_real = c1 * psi1[:, 0] + c2 * psi2[:, 0]
    psi_imag = c1 * psi1[:, 1] + c2 * psi2[:, 1]
    
    return torch.stack([psi_real, psi_imag], dim=1)

def visualize_superposition(model_path, device='cpu', L=1.0, m=1.0, 
                           n1=1, n2=2, c1=None, c2=None,
                           num_frames=100, num_points=500, 
                           output_path='schrodinger_superposition.mp4'):
    """
    Create animation showing probability density evolution for superposition state
    
    Args:
        model_path: Path to trained model checkpoint
        device: Device to run on ('cpu', 'cuda', 'mps')
        L: Length of the 1D box
        m: Mass of particle
        n1, n2: Quantum numbers for the two states
        c1, c2: Coefficients for superposition (default: equal superposition)
        num_frames: Number of frames in animation
        num_points: Number of spatial points to evaluate
        output_path: Path to save MP4 video
    """
    
    # Default to equal superposition
    if c1 is None or c2 is None:
        c1 = c2 = 1.0 / np.sqrt(2)
    
    # Load model
    print("Loading model...")
    model = SchrodingerEquation1D(L).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create spatial grid
    x_vals = torch.linspace(0, L, num_points, device=device).reshape(-1, 1)
    
    # Time values from 0 to 1
    t_vals = torch.linspace(0, 1, num_frames, device=device)
    
    print(f"Computing predictions for superposition: {c1:.3f}|n={n1}⟩ + {c2:.3f}|n={n2}⟩...")
    
    # Store all frames data
    pinn_probs = []
    analytical_probs = []
    
    with torch.no_grad():
        for t_val in t_vals:
            t = torch.full_like(x_vals, t_val.item())
            
            # PINN prediction
            psi_pinn = pinn_superposition(model, x_vals, t, n1, n2, c1, c2)
            prob_pinn = calculate_probability(psi_pinn)
            
            # Analytical solution
            psi_analytical = schrodinger_superposition(x_vals, t, L, n1, n2, c1, c2, m)
            prob_analytical = calculate_probability(psi_analytical)
            
            pinn_probs.append(prob_pinn.cpu().numpy())
            analytical_probs.append(prob_analytical.cpu().numpy())
    
    x_cpu = x_vals.cpu().numpy().flatten()
    
    print("Creating animation...")
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Find max probability for consistent y-axis
    max_prob = max(max(np.max(p) for p in pinn_probs), 
                   max(np.max(p) for p in analytical_probs))
    
    # Initialize lines
    line_pinn, = ax.plot([], [], 'b-', linewidth=2, label='PINN Prediction')
    line_analytical, = ax.plot([], [], 'r--', linewidth=2, label='Analytical Solution')
    
    ax.set_xlim(0, L)
    ax.set_ylim(0, max_prob * 1.1)
    ax.set_xlabel('Position x', fontsize=12)
    ax.set_ylabel('Probability Density |ψ(x,t)|²', fontsize=12)
    ax.set_title(f'Superposition State: {c1:.3f}|n={n1}⟩ + {c2:.3f}|n={n2}⟩', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Time text
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Error text
    error_text = ax.text(0.02, 0.87, '', transform=ax.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    def init():
        """Initialize animation"""
        line_pinn.set_data([], [])
        line_analytical.set_data([], [])
        time_text.set_text('')
        error_text.set_text('')
        return line_pinn, line_analytical, time_text, error_text
    
    def animate(frame):
        """Update animation for each frame"""
        t = t_vals[frame].item()
        
        # Update lines
        line_pinn.set_data(x_cpu, pinn_probs[frame])
        line_analytical.set_data(x_cpu, analytical_probs[frame])
        
        # Update time text
        time_text.set_text(f't = {t:.3f}')
        
        # Calculate percent error
        pinn_p = pinn_probs[frame]
        anal_p = analytical_probs[frame]
        relative_error = np.mean(np.abs(pinn_p - anal_p)) / (np.mean(anal_p) + 1e-10)
        percent_error = relative_error * 100
        error_text.set_text(f'Percent Error: {percent_error:.2f}%')
        
        return line_pinn, line_analytical, time_text, error_text
    
    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=num_frames,
                        interval=50, blit=True, repeat=True)
    
    # Save as MP4
    print(f"Saving animation to {output_path}...")
    writer = FFMpegWriter(fps=20, metadata=dict(artist='PINN Visualization'), 
                         bitrate=1800)
    anim.save(output_path, writer=writer)
    
    print(f"Animation saved successfully!")
    plt.close()
    
    # Create summary plot showing multiple time snapshots
    print("Creating summary plot...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    snapshot_times = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    snapshot_indices = [int(t * (num_frames - 1)) for t in snapshot_times]
    
    for idx, (ax, t_idx) in enumerate(zip(axes, snapshot_indices)):
        t = t_vals[t_idx].item()
        ax.plot(x_cpu, pinn_probs[t_idx], 'b-', linewidth=2, label='PINN')
        ax.plot(x_cpu, analytical_probs[t_idx], 'r--', linewidth=2, label='Analytical')
        ax.set_xlabel('Position x')
        ax.set_ylabel('|ψ|²')
        ax.set_title(f't = {t:.2f}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max_prob * 1.1)
        if idx == 0:
            ax.legend()
    
    plt.suptitle(f'Superposition: {c1:.3f}|n={n1}⟩ + {c2:.3f}|n={n2}⟩', fontsize=14, y=0.995)
    plt.tight_layout()
    
    summary_path = output_path.replace('.mp4', '_snapshots.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"Summary plot saved to {summary_path}")
    plt.close()
    
    # Calculate and print statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    
    all_errors = []
    for frame in range(num_frames):
        pinn_p = pinn_probs[frame]
        anal_p = analytical_probs[frame]
        error = np.mean(np.abs(pinn_p - anal_p)) / (np.mean(anal_p) + 1e-10)
        all_errors.append(error)
    
    # Convert to percentages
    all_errors_percent = [e * 100 for e in all_errors]
    
    print(f"Mean Percent Error (all frames): {np.mean(all_errors_percent):.4f}%")
    print(f"Max Percent Error: {np.max(all_errors_percent):.4f}%")
    print(f"Min Percent Error: {np.min(all_errors_percent):.4f}%")
    print(f"Std Percent Error: {np.std(all_errors_percent):.4f}%")
    print("="*60)

if __name__ == "__main__":
    # Configuration
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model_path = "schrodingers_equation_1d.pt"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train the model first using schrodingers_equation_1d.py")
        exit(1)
    
    # Parameters
    L = 1.0          # Box length
    m = 1.0          # Mass
    n = 1            # Energy level (ground state)
    
    print("\nGenerating visualization for superposition: (|n=1⟩ + |n=2⟩)/√2...")
    visualize_superposition(
        model_path=model_path,
        device=device,
        L=L,
        m=m,
        n1=1,
        n2=2,
        c1=1.0/np.sqrt(2),
        c2=1.0/np.sqrt(2),
        num_frames=100,
        num_points=500,
        output_path='./assets/schrodinger_superposition_n1_n2.mp4'
    )
    
    print("\nGenerating visualization for superposition: (|n=1⟩ + |n=3⟩)/√2...")
    visualize_superposition(
        model_path=model_path,
        device=device,
        L=L,
        m=m,
        n1=1,
        n2=3,
        c1=1.0/np.sqrt(2),
        c2=1.0/np.sqrt(2),
        num_frames=100,
        num_points=500,
        output_path='./assets/schrodinger_superposition_n1_n3.mp4'
    )
    
    print("\nGenerating visualization for superposition: (|n=2⟩ + |n=3⟩)/√2...")
    visualize_superposition(
        model_path=model_path,
        device=device,
        L=L,
        m=m,
        n1=2,
        n2=3,
        c1=1.0/np.sqrt(2),
        c2=1.0/np.sqrt(2),
        num_frames=100,
        num_points=500,
        output_path='./assets/schrodinger_superposition_n2_n3.mp4'
    )
    