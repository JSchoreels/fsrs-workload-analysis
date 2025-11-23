from workload_analysis import FSRS6_Standalone


def test_forgetting_curve_inverse(delta=1e-6):
    """Test that forgetting_curve and interval_from_retention are inverses"""
    fsrs = FSRS6_Standalone()

    # Test various retention and stability values
    test_cases = [
        (0.9, 10),    # 90% retention, 10 days stability
        (0.8, 30),    # 80% retention, 30 days stability
        (0.7, 100),   # 70% retention, 100 days stability
        (0.95, 5),    # 95% retention, 5 days stability
        (0.5, 200),   # 50% retention, 200 days stability
    ]

    print("Testing forgetting_curve ↔ interval_from_retention inverse relationship:\n")
    all_passed = True

    for r, s in test_cases:
        # r -> t -> r_back
        t = fsrs.interval_from_retention(r, s)
        r_back = fsrs.forgetting_curve(t, s)

        error = abs(r - r_back)
        passed = error < delta
        all_passed = all_passed and passed

        status = "✓" if passed else "✗"
        print(f"{status} R={r:.2f}, S={s:3d}d → t={t:6.2f}d → R_back={r_back:.6f} (error={error:.2e})")

    print(f"\n{'All tests passed!' if all_passed else 'Some tests failed!'}")
    return all_passed


if __name__ == "__main__":
    test_forgetting_curve_inverse()
