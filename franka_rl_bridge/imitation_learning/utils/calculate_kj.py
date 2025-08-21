import numpy as np
from typing import Optional

def compute_joint_stiffness(Kc: np.ndarray, J: np.ndarray, verbose: bool = True) -> np.ndarray:
    """
    Convert Cartesian stiffness Kc (6×6) to joint stiffness Kj (7×7):
      Kj = J^T * Kc * J

    Args:
        Kc: Cartesian stiffness matrix, shape (6,6)
        J: Robot Jacobian at current pose, shape (6,7)
        verbose: If True, print detailed information about the computation

    Returns:
        Kj: Joint-space stiffness matrix, shape (7,7)
    """
    if verbose:
        print("="*60)
        print("🤖 COMPUTING JOINT STIFFNESS FROM CARTESIAN STIFFNESS")
        print("="*60)
    
    # Validate shapes
    if verbose:
        print(f"📥 Input validation:")
        print(f"  Cartesian stiffness Kc shape: {Kc.shape}")
        print(f"  Jacobian J shape: {J.shape}")
    
    assert Kc.shape == (6, 6), f"Kc must be 6×6, got {Kc.shape}"
    assert J.shape[0] == 6, f"Jacobian must have 6 rows, got {J.shape[0]}"
    assert J.shape[1] == 7, f"Jacobian must have 7 columns for Franka robot, got {J.shape[1]}"
    
    if verbose:
        print("✅ Input validation passed")
    
    # Check if Kc is diagonal for simplified interpretation
    is_diagonal = np.allclose(Kc, np.diag(np.diag(Kc)), rtol=1e-10)
    if verbose:
        print(f"📊 Cartesian stiffness matrix properties:")
        print(f"  Is diagonal: {is_diagonal}")
        if is_diagonal:
            kc_diag = np.diag(Kc)
            print(f"  Diagonal values: [{', '.join([f'{k:.1f}' for k in kc_diag])}]")
            print(f"  Position stiffness [N/m]: [{', '.join([f'{k:.1f}' for k in kc_diag[:3]])}]")
            print(f"  Orientation stiffness [Nm/rad]: [{', '.join([f'{k:.1f}' for k in kc_diag[3:]])}]")
        
        # Matrix properties
        print(f"  Condition number: {np.linalg.cond(Kc):.2e}")
        print(f"  Determinant: {np.linalg.det(Kc):.2e}")
        print(f"  Trace: {np.trace(Kc):.2f}")
    
    # Analyze Jacobian properties
    if verbose:
        print(f"🔧 Jacobian matrix properties:")
        print(f"  Condition number: {np.linalg.cond(J):.2e}")
        
        # Check for singularities
        singular_values = np.linalg.svd(J, compute_uv=False)
        min_sv = np.min(singular_values)
        max_sv = np.max(singular_values)
        print(f"  Singular values range: [{min_sv:.4f}, {max_sv:.4f}]")
        
        if min_sv < 1e-3:
            print(f"⚠️ Robot may be near singularity (min singular value: {min_sv:.4f})")
        else:
            print(f"✅ Robot configuration is well-conditioned")
        
        # Manipulability measure
        manipulability = np.sqrt(np.linalg.det(J @ J.T))
        print(f"  Manipulability measure: {manipulability:.4f}")
    
    # Compute joint stiffness
    if verbose:
        print("⚙️ Computing Kj = J^T * Kc * J...")
    
    Kj = J.T @ Kc @ J
    
    if verbose:
        print("✅ Joint stiffness computation completed")
        
        # Analyze result
        print(f"📈 Joint stiffness matrix properties:")
        print(f"  Shape: {Kj.shape}")
        print(f"  Condition number: {np.linalg.cond(Kj):.2e}")
        print(f"  Determinant: {np.linalg.det(Kj):.2e}")
        print(f"  Trace: {np.trace(Kj):.2f}")
        
        # Check symmetry
        is_symmetric = np.allclose(Kj, Kj.T, rtol=1e-10)
        print(f"  Is symmetric: {is_symmetric}")
        
        # Check positive definiteness
        eigenvalues = np.linalg.eigvals(Kj)
        is_positive_definite = np.all(eigenvalues > 0)
        print(f"  Is positive definite: {is_positive_definite}")
        
        if not is_positive_definite:
            negative_eigs = eigenvalues[eigenvalues <= 0]
            print(f"⚠️ Matrix has {len(negative_eigs)} non-positive eigenvalue(s): {negative_eigs}")
        
        # Diagonal elements (approximate joint stiffness values)
        joint_stiffness_diag = np.diag(Kj)
        print(f"🎯 Diagonal joint stiffness values:")
        for i, k in enumerate(joint_stiffness_diag):
            print(f"  Joint {i+1}: {k:8.2f} Nm/rad")
        
        print(f"📊 Joint stiffness range: [{joint_stiffness_diag.min():.2f}, {joint_stiffness_diag.max():.2f}] Nm/rad")
        
        # Check for potential issues
        if joint_stiffness_diag.max() / joint_stiffness_diag.min() > 1000:
            print("⚠️ Large stiffness ratio detected - may cause numerical issues")
        
        if np.any(joint_stiffness_diag < 1.0):
            low_stiffness_joints = np.where(joint_stiffness_diag < 1.0)[0] + 1
            print(f"⚠️ Very low stiffness detected in joints: {low_stiffness_joints}")
        
        if np.any(joint_stiffness_diag > 10000):
            high_stiffness_joints = np.where(joint_stiffness_diag > 10000)[0] + 1
            print(f"⚠️ Very high stiffness detected in joints: {high_stiffness_joints}")
        
        print("="*60)
    
    return Kj

def analyze_jacobian_at_pose(J: np.ndarray, pose_name: str = "current") -> None:
    """
    Analyze Jacobian properties at a given pose
    
    Args:
        J: Robot Jacobian at pose, shape (6,7)
        pose_name: Descriptive name for the pose
    """
    print(f"\n{'='*50}")
    print(f"🔍 JACOBIAN ANALYSIS AT {pose_name.upper()} POSE")
    print(f"{'='*50}")
    
    # SVD analysis
    U, s, Vt = np.linalg.svd(J)
    
    print(f"📊 Singular Value Decomposition:")
    print(f"  Singular values: [{', '.join([f'{sv:.4f}' for sv in s])}]")
    print(f"  Condition number: {s[0]/s[-1]:.2e}")
    
    # Velocity transmission ratios
    print(f"🚀 Velocity transmission analysis:")
    for i, sv in enumerate(s):
        print(f"  Direction {i+1}: transmission ratio = {sv:.4f}")
    
    # Manipulability
    manipulability = np.prod(s)
    print(f"  Manipulability (product of singular values): {manipulability:.6f}")
    
    # Check for near-singularity
    if s[-1] < 1e-3:
        print(f"⚠️ Near singularity detected (smallest singular value: {s[-1]:.4f})")
        print("  Robot may have difficulty moving in certain directions")
    
    print(f"{'='*50}")

def suggest_joint_gains(Kj: np.ndarray, safety_factor: float = 0.8) -> np.ndarray:
    """
    Suggest practical joint gains based on computed stiffness matrix
    
    Args:
        Kj: Joint stiffness matrix (7×7)
        safety_factor: Safety factor to reduce gains (0 < factor <= 1)
    
    Returns:
        Suggested diagonal joint gains
    """
    print(f"\n{'='*40}")
    print(f"🎯 JOINT GAIN SUGGESTIONS")
    print(f"{'='*40}")
    
    # Extract diagonal elements
    diagonal_stiffness = np.diag(Kj)
    
    # Apply safety factor
    suggested_gains = diagonal_stiffness * safety_factor
    
    # Practical limits (based on Franka specifications)
    min_gain = 1.0
    max_gain = 5000.0
    
    # Clip to practical range
    clipped_gains = np.clip(suggested_gains, min_gain, max_gain)
    
    print(f"🛡️ Safety factor applied: {safety_factor}")
    print(f"⚙️ Gain limits: [{min_gain}, {max_gain}] Nm/rad")
    print(f"\n🎛️ Suggested joint gains:")
    
    for i, (original, suggested, clipped) in enumerate(zip(diagonal_stiffness, suggested_gains, clipped_gains)):
        status = "✓" if suggested == clipped else "⚠️ (clipped)"
        print(f"  Joint {i+1}: {original:8.2f} → {suggested:8.2f} → {clipped:8.2f} {status}")
    
    if not np.array_equal(suggested_gains, clipped_gains):
        print("⚠️ Some gains were clipped to practical limits")
    
    print(f"{'='*40}")
    
    return clipped_gains

def save_results_to_file(Kc: np.ndarray, J: np.ndarray, Kj: np.ndarray, 
                        filename: Optional[str] = None) -> None:
    """
    Save computation results to file for later analysis
    
    Args:
        Kc: Cartesian stiffness matrix
        J: Jacobian matrix
        Kj: Joint stiffness matrix
        filename: Output filename (auto-generated if None)
    """
    if filename is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"joint_stiffness_computation_{timestamp}.npz"
    
    np.savez(filename, 
             cartesian_stiffness=Kc,
             jacobian=J, 
             joint_stiffness=Kj,
             joint_gains_diagonal=np.diag(Kj))
    
    print(f"💾 Results saved to: {filename}")

# Example usage with improved logging:
if __name__ == "__main__":
    print("🤖 Joint Stiffness Computation Tool")
    print("This tool converts Cartesian stiffness to joint space stiffness")
    
    # Define sample Cartesian stiffness (typical impedance control values)
    Kc = np.diag([1200, 1200, 1200, 80, 80, 80])  # N/m and Nm/rad
    print(f"Using Cartesian stiffness: diag{np.diag(Kc).tolist()}")
    
    # Example Jacobian at a typical pose (6×7) @home position
    J = np.array([
        [0.00245166, -0.0653364,  -0.001655,   0.395646,    0.00190744,  0.213878,     -4.50823e-19],
        [0.468204,    0.0287576,   0.45419,    0.0237196,   0.114093,   -0.000424604, -9.71871e-19],
        [0.0,        -0.429519,   -0.0264536,  0.426819,   -0.00454094,  0.0791719,   -6.88337e-20],
        [0.0,         0.402851,   -0.129912,  -0.00206831,  0.769747,   -0.0167027,   -0.0419622],
        [0.0,         0.915266,    0.0571804, -0.998349,   -0.0382296,  -0.99907,     -0.0390333],
        [1.0,         0.0,         0.989875,   0.0573984,   -0.637203,   0.0397632,    -0.998356]
    ])
    
    # Analyze Jacobian first
    analyze_jacobian_at_pose(J, "example")
    
    # Compute joint stiffness with verbose logging
    Kj = compute_joint_stiffness(Kc, J, verbose=True)
    
    # Get suggested gains
    suggested_gains = suggest_joint_gains(Kj, safety_factor=0.8)
    
    # Save results
    save_results_to_file(Kc, J, Kj)
    
    print("\n✅ Computation completed successfully!")
    print("💡 Use these joint gains in your PD controller for impedance-like behavior")
