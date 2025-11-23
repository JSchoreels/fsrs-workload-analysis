import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class FSRS6_Standalone:
    """
    Standalone implementation of FSRS v6 logic extracted from the
    provided file chain (v6 -> v5 -> v4.5 -> v4).
    """

    def __init__(self):
        # Default parameters (init_w) from fsrs_v6.py
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
        """
        Calculates S_new for a 'Good' (3) rating.
        Formula from FSRS4.
        """
        new_s = s * (
                1
                + torch.exp(self.w[8])
                * (11 - d)
                * torch.pow(s, -self.w[9])
                * (torch.exp((1 - r) * self.w[10]) - 1)
            # * 1 (hard_penalty for rating 3)
            # * 1 (easy_bonus for rating 3)
        )
        return new_s

    def calculate_failure_stability(self, s: torch.Tensor, d: float, r: torch.Tensor) -> torch.Tensor:
        """
        Calculates S_new for a 'Again' (1) rating.
        Formula from FSRS5 (fsrs_v5.py), which overrides previous versions.
        """

        new_s_calc = (
                self.w[11]
                * torch.pow(torch.tensor(d), -self.w[12])
                * (torch.pow(s + 1, self.w[13]) - 1)
                * torch.exp((1 - r) * self.w[14])
        )

        new_minimum_s = s / torch.exp(self.w[17] * self.w[18])

        return torch.minimum(new_s_calc, new_minimum_s)


def calculate_workload_reduction(R: torch.Tensor, S: float, D: float, fsrs: FSRS6_Standalone) -> torch.Tensor:
    """
    Calculate workload reduction (workload_init - workload_after) for given R, S, and D values.

    Args:
        R: Retention values (can be a tensor)
        S: Current Stability (days)
        D: Current Difficulty (1-10)
        fsrs: FSRS6_Standalone instance

    Returns:
        Workload reduction (workload_init - workload_after)
        workload_init = 1/S
        workload_after = 1/(S*expected_increase_mean)
        reduction = 1/S - 1/(S*expected_increase_mean) = (1/S) * (1 - 1/expected_increase_mean)
    """
    S_tensor = torch.tensor(S, dtype=torch.float32)

    # Calculate Stability after Success and Failure
    S_success = fsrs.calculate_success_stability(S_tensor, D, R)
    S_failure = fsrs.calculate_failure_stability(S_tensor, D, R)

    # Calculate Relative Changes
    rel_increase_success = S_success / S
    rel_decrease_failure = S_failure / S

    # Calculate Expected Increase (arithmetic mean)
    expected_increase_mean = R * rel_increase_success + (1 - R) * rel_decrease_failure

    # Calculate workload reduction: workload_init - workload_after
    workload_init = 1 / S
    workload_after = 1 / (S * expected_increase_mean)
    workload_reduction = workload_init - workload_after

    return workload_reduction


def analyze_by_difficulty(S: float, fsrs: FSRS6_Standalone, save_path: str = None):
    """
    Create 3D surface plot varying Difficulty (D) and Retention (R) for fixed Stability (S).

    Args:
        S: Fixed Stability value (days)
        fsrs: FSRS6_Standalone instance
        save_path: Optional path to save the figure
    """
    # Create ranges for D and R
    D_values = torch.linspace(1, 10, steps=50)  # Difficulty from 1 to 10
    R_values = torch.linspace(0.01, 0.99, steps=50)  # Retention from 0.01 to 0.99

    print(f"Calculating 3D surface for S={S}")

    # ==========================================
    # CALCULATE WORKLOAD REDUCTION FOR ALL D AND R COMBINATIONS
    # ==========================================
    # Create meshgrid for R and D
    R_mesh, D_mesh = torch.meshgrid(R_values, D_values, indexing='ij')

    # Initialize workload reduction array
    workload_reduction_surface = np.zeros_like(R_mesh.numpy())

    # Calculate workload reduction for each D value
    for j, D in enumerate(D_values):
        workload_reduction_surface[:, j] = calculate_workload_reduction(R_values, S, float(D), fsrs).numpy()

    # ==========================================
    # CREATE 3D SURFACE PLOT
    # ==========================================
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Set color limits symmetric around zero for diverging colormap
    vmax = np.abs(workload_reduction_surface).max()
    vmin = -vmax

    # Create surface plot with red-green diverging colormap
    surf = ax.plot_surface(R_mesh.numpy(), D_mesh.numpy(), workload_reduction_surface,
                          cmap='RdYlGn', alpha=0.8, edgecolor='none',
                          vmin=vmin, vmax=vmax)

    ax.set_xlabel('R (Retention)', fontsize=9)
    ax.set_ylabel('D (Difficulty)', fontsize=9)
    ax.set_zlabel('Workload Reduction', fontsize=9)
    ax.set_title(f'S={S} days', fontsize=11)

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    else:
        plt.show()


def analyze_by_stability(D: float, fsrs: FSRS6_Standalone, save_path: str = None, S_min: float = 1, S_max: float = 50):
    """
    Create 3D surface plot varying Stability (S) and Retention (R) for fixed Difficulty (D).

    Args:
        D: Fixed Difficulty value (1-10)
        fsrs: FSRS6_Standalone instance
        save_path: Optional path to save the figure
        S_min: Minimum stability value (days)
        S_max: Maximum stability value (days)
    """
    # Create ranges for S and R
    S_values = torch.linspace(S_min, S_max, steps=50)  # Stability range
    R_values = torch.linspace(0.01, 0.99, steps=50)  # Retention from 0.01 to 0.99

    print(f"Calculating 3D surface for D={D}")

    # ==========================================
    # CALCULATE WORKLOAD REDUCTION FOR ALL S AND R COMBINATIONS
    # ==========================================
    # Create meshgrid for S and R
    R_mesh, S_mesh = torch.meshgrid(R_values, S_values, indexing='ij')

    # Initialize workload reduction array
    workload_reduction_surface = np.zeros_like(R_mesh.numpy())

    # Calculate workload reduction for each S value
    for j, S in enumerate(S_values):
        workload_reduction_surface[:, j] = calculate_workload_reduction(R_values, float(S), D, fsrs).numpy()

    # ==========================================
    # CREATE 3D SURFACE PLOT
    # ==========================================
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Set color limits symmetric around zero for diverging colormap
    vmax = np.abs(workload_reduction_surface).max()
    vmin = -vmax

    # Create surface plot with red-green diverging colormap
    surf = ax.plot_surface(R_mesh.numpy(), S_mesh.numpy(), workload_reduction_surface,
                          cmap='RdYlGn', alpha=0.8, edgecolor='none',
                          vmin=vmin, vmax=vmax)

    ax.set_xlabel('R (Retention)', fontsize=9)
    ax.set_ylabel('S (Stability, days)', fontsize=9)
    ax.set_zlabel('Workload Reduction', fontsize=9)
    ax.set_title(f'D={D}, S={S_min:.0f}-{S_max:.0f}', fontsize=11)

    # Add colorbar
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

    # Generate images varying stability at different difficulty values (1-50 days)
    for d in [3, 7, 10]:
        analyze_by_stability(D=d, fsrs=fsrs, save_path=f'images/stability_D{d}_1-50.png', S_min=1, S_max=50)

    # Generate images varying stability at different difficulty values (50-100 days)
    for d in [3, 7, 10]:
        analyze_by_stability(D=d, fsrs=fsrs, save_path=f'images/stability_D{d}_50-100.png', S_min=50, S_max=100)


def main():
    # mpl.use('macosx')

    # Initialize Model
    fsrs = FSRS6_Standalone()

    # Analyze varying Difficulty for cards with S around 10-50 days
    # You can try different S values: 10, 20, 30, 40, 50
    for stability in [10, 25, 50, 100, 200, 500, 1000]:
        analyze_by_difficulty(S=stability, fsrs=fsrs)

    # Original analysis: varying Stability for fixed Difficulty
    for difficulty in [1, 4, 6, 7, 8, 9, 10]:
        analyze_by_stability(D=difficulty, fsrs=fsrs)


if __name__ == "__main__":
    main()