import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class FSRS6_Standalone:
    """Standalone FSRS v6 implementation"""

    def __init__(self):
        # FSRS v6 default parameters
        # self.w = torch.tensor([
        #     0.212,  # 0
        #     1.2931,  # 1
        #     2.3065,  # 2
        #     8.2956,  # 3
        #     6.4133,  # 4
        #     0.8334,  # 5
        #     3.0194,  # 6
        #     0.001,  # 7
        #     1.8722,  # 8
        #     0.1666,  # 9
        #     0.796,  # 10
        #     1.4835,  # 11
        #     0.0614,  # 12
        #     0.2629,  # 13
        #     1.6483,  # 14
        #     0.6014,  # 15
        #     1.8729,  # 16
        #     0.5425,  # 17
        #     0.0912,  # 18
        #     0.0658,  # 19
        #     0.1542  # 20
        # ], dtype=torch.float32)
        # S.
        self.w = torch.tensor([0.2349, 0.9428, 3.9614, 15.1168, 6.4464, 0.6630, 3.0691, 0.0129, 1.6820, 0.3375, 0.6230, 1.6110, 0.0110, 0.4084, 1.7567, 0.4727, 2.0014, 0.7389, 0.2581, 0.1250, 0.1000], dtype=torch.float32)
        # E.
        # self.w = torch.tensor([1.1181, 1.1181, 14.9561, 100.0000, 6.1425, 0.8527, 3.2886, 0.0096, 2.5799, 0.0000, 1.3824, 1.5719, 0.0455, 0.3267, 1.7273, 0.0493, 2.7913, 0.6267, 0.3912, 0.0451, 0.1000], dtype=torch.float32)
        # D.
        # self.w = torch.tensor([0.4045, 3.3476, 3.3476, 3.3476, 6.3651, 0.5392, 3.1642, 0.0341, 1.7171, 0.1858, 0.6978, 1.5303, 0.0393, 0.3250, 1.7042, 0.2680, 1.8729, 0.6017, 0.1061, 0.1269, 0.1001], dtype=torch.float32)

    def calculate_success_stability(self, s: torch.Tensor, d: float, r: torch.Tensor) -> torch.Tensor:
        """Calculate S_new for 'Good' (3) rating"""
        new_s = s * (
                1
                + torch.exp(self.w[8])
                * (11 - d)
                * torch.pow(s, -self.w[9])
                * (torch.exp((1 - r) * self.w[10]) - 1)
        )
        return new_s

    def calculate_failure_stability(self, s: torch.Tensor, d: float, r: torch.Tensor) -> torch.Tensor:
        """Calculate S_new for 'Again' (1) rating"""

        new_s_calc = (
                self.w[11]
                * torch.pow(torch.tensor(d), -self.w[12])
                * (torch.pow(s + 1, self.w[13]) - 1)
                * torch.exp((1 - r) * self.w[14])
        )

        new_minimum_s = s / torch.exp(self.w[17] * self.w[18])

        return torch.minimum(new_s_calc, new_minimum_s)

    def forgetting_curve(self, t, s, decay=None):
        """Calculate retention R(t) at time t for given stability s"""
        if not decay:
            decay = self.w[20]
        factor = 0.9 ** (1 / decay) - 1
        return (1 + factor * t / s) ** decay

    def interval_from_retention(self, r, s, decay=None):
        """Calculate time t when retention drops to r for given stability s"""
        if not decay:
            decay = self.w[20]
        factor = 0.9 ** (1 / decay) - 1
        return s * (r ** (1 / decay) - 1) / factor


def calculate_workload_reduction(R: torch.Tensor, S: float, D: float, fsrs: FSRS6_Standalone) -> torch.Tensor:
    """Calculate workload reduction using actual review intervals at retention R"""
    S_tensor = torch.tensor(S, dtype=torch.float32)

    S_success = fsrs.calculate_success_stability(S_tensor, D, R)
    S_failure = fsrs.calculate_failure_stability(S_tensor, D, R)

    interval_init = fsrs.interval_from_retention(R, S)
    interval_after_success = torch.max(torch.ones(1), fsrs.interval_from_retention(R, S_success))
    interval_after_failure = torch.max(torch.ones(1), fsrs.interval_from_retention(R, S_failure))

    workload_init = 1 / interval_init
    workload_after = R * (1 / interval_after_success)  + (1 - R) * (1 / interval_after_failure)
    return (workload_init - workload_after) * torch.min(torch.ones(1), workload_after)


def analyze_by_difficulty(S: float, fsrs: FSRS6_Standalone, save_path: str = None):
    """Plot workload reduction varying D and R for fixed S"""
    D_values = torch.linspace(1, 10, steps=50)
    R_values = torch.linspace(0.01, 0.99, steps=50)

    print(f"Calculating 3D surface for S={S}")

    # Calculate workload reduction for all D and R combinations
    R_mesh, D_mesh = torch.meshgrid(R_values, D_values, indexing='ij')
    workload_reduction_surface = np.zeros_like(R_mesh.numpy())

    for j, D in enumerate(D_values):
        workload_reduction_surface[:, j] = calculate_workload_reduction(R_values, S, float(D), fsrs).numpy()

    # Create 3D plot with diverging colormap
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    vmax = np.abs(workload_reduction_surface).max()
    surf = ax.plot_surface(R_mesh.numpy(), D_mesh.numpy(), workload_reduction_surface,
                          cmap='RdYlGn', alpha=0.8, edgecolor='none',
                          vmin=-vmax, vmax=vmax)

    ax.set_xlabel('R (Retention)', fontsize=9)
    ax.set_ylabel('D (Difficulty)', fontsize=9)
    ax.set_zlabel('Workload Reduction', fontsize=9)
    ax.set_title(f'S={S} days', fontsize=11)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    else:
        plt.show()


def analyze_by_stability(D: float, fsrs: FSRS6_Standalone, save_path: str = None, S_min: float = 1, S_max: float = 50):
    """Plot workload reduction varying S and R for fixed D"""
    S_values = torch.linspace(S_min, S_max, steps=50)
    R_values = torch.linspace(0.01, 0.99, steps=50)

    print(f"Calculating 3D surface for D={D}")

    # Calculate workload reduction for all S and R combinations
    R_mesh, S_mesh = torch.meshgrid(R_values, S_values, indexing='ij')
    workload_reduction_surface = np.zeros_like(R_mesh.numpy())

    for j, S in enumerate(S_values):
        workload_reduction_surface[:, j] = calculate_workload_reduction(R_values, float(S), D, fsrs).numpy()

    # Create 3D plot with diverging colormap
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    vmax = np.abs(workload_reduction_surface).max()
    surf = ax.plot_surface(R_mesh.numpy(), S_mesh.numpy(), workload_reduction_surface,
                          cmap='RdYlGn', alpha=0.8, edgecolor='none',
                          vmin=-vmax, vmax=vmax)

    ax.set_xlabel('R (Retention)', fontsize=9)
    ax.set_ylabel('S (Stability, days)', fontsize=9)
    ax.set_zlabel('Workload Reduction', fontsize=9)
    ax.set_title(f'D={D}, S={S_min:.0f}-{S_max:.0f}', fontsize=11)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    else:
        plt.show()


def generate_sample_images():
    """Generate sample images for README"""
    import os
    os.makedirs('images', exist_ok=True)

    fsrs = FSRS6_Standalone()

    # Generate images varying difficulty at different stability values
    for s in [25, 100, 500]:
        analyze_by_difficulty(S=s, fsrs=fsrs, save_path=f'images/difficulty_S{s}.png')

    # Generate images varying stability at different difficulty values for multiple S ranges
    for d in [3, 7, 10]:
        for s_min, s_max in [(10, 50), (50, 100)]:
            analyze_by_stability(D=d, fsrs=fsrs, save_path=f'images/stability_D{d}_{s_min}-{s_max}.png', S_min=s_min, S_max=s_max)


def main():
    # mpl.use('macosx')
    fsrs = FSRS6_Standalone()

    # Varying difficulty analysis
    for stability in [10, 25, 50, 100, 200, 500, 1000]:
        analyze_by_difficulty(S=stability, fsrs=fsrs)

    # Varying stability analysis
    for difficulty in [1, 4, 6, 7, 8, 9, 10]:
        analyze_by_stability(D=difficulty, fsrs=fsrs)


if __name__ == "__main__":
    main()