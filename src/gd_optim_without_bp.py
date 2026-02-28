"""optim ccm by manual numerical gradient descent (no backpropagation)"""

import numpy as np
from jaxtyping import Float32, UInt8
from src.common_utils import (
    rgb_to_oklab,
    oklab_to_rgb,
    rgb_to_lab,
    make_ccm_from_chromosome,
    calc_lab_distance,
    inv_gamma_vectorized
)


def forward_pass(
    ccm_params: Float32[np.ndarray, "6"],
    rgb_mean_array: Float32[np.ndarray, "18 3"],
    linear_lab_patch_target: Float32[np.ndarray, "18 3"],
    lab_patch_weight: Float32[np.ndarray, "18"],
    gamma_en: bool,
    gamma_curve: UInt8[np.ndarray, "256"],
) -> tuple[float, np.ndarray]:
    """
    Forward pass: compute LAB distance loss given CCM parameters
    
    Args:
        ccm_params: 6 CCM parameters (chromosome)
        rgb_mean_array: RGB values for 18 patches
        linear_lab_patch_target: Target LAB values
        lab_patch_weight: Weights for each patch
        gamma_en: Whether gamma is enabled
        gamma_curve: Gamma correction lookup table
        
    Returns:
        tuple: (total_loss, patch_distances)
    """
    # Build CCM from chromosome
    ccm = make_ccm_from_chromosome(ccm_params)

    # Apply CCM to RGB
    rgb_18_patch = rgb_mean_array.transpose()  # (3, 18)
    ccm_18_rgb = np.clip(np.dot(ccm, rgb_18_patch), 0.0, 255.0)  # (3, 3) * (3, 18) => (3, 18)
    ccm_18_rgb = ccm_18_rgb.transpose()  # (18, 3)

    # Convert to OKLab
    ccm_18_oklab = rgb_to_oklab(ccm_18_rgb)

    # Blend L channel (keep original L)
    oklab_18_patch = rgb_to_oklab(np.clip(rgb_mean_array, 0.0, 255.0))
    ccm_18_oklab[:, 0] = oklab_18_patch[:, 0]

    # Convert back to RGB
    ccm_18_rgb = oklab_to_rgb(ccm_18_oklab)
    ccm_18_rgb = np.clip(ccm_18_rgb * 255.0, 0.0, 255.0)

    # Apply inverse gamma if needed
    if not gamma_en:
        ccm_18_rgb = inv_gamma_vectorized(ccm_18_rgb, gamma_curve)

    # Convert to Lab
    ccm_18_lab = rgb_to_lab(ccm_18_rgb)

    # Calculate LAB distance
    lab_distance, lab_patch_distance = calc_lab_distance(
        linear_lab_patch_target, ccm_18_lab, lab_patch_weight, just_ab=True
    )

    return lab_distance, lab_patch_distance


def compute_numerical_gradient(
    ccm_params: Float32[np.ndarray, "6"],
    rgb_mean_array: Float32[np.ndarray, "18 3"],
    linear_lab_patch_target: Float32[np.ndarray, "18 3"],
    lab_patch_weight: Float32[np.ndarray, "18"],
    gamma_en: bool,
    gamma_curve: UInt8[np.ndarray, "256"],
    eps: float = 1e-6,
) -> Float32[np.ndarray, "6"]:
    """
    Compute numerical gradient using central difference method
    
    For each of the 6 CCM parameters:
        gradient[i] = (f(x + eps) - f(x - eps)) / (2 * eps)
    
    Args:
        ccm_params: Current CCM parameters
        rgb_mean_array: RGB values for 18 patches
        linear_lab_patch_target: Target LAB values
        lab_patch_weight: Weights for each patch
        gamma_en: Whether gamma is enabled
        gamma_curve: Gamma correction lookup table
        eps: Small perturbation for numerical differentiation
        
    Returns:
        Gradient array of shape (6,)
    """
    gradient = np.zeros(6, dtype=np.float32)

    # Compute gradient for each parameter using central difference
    for i in range(6):
        # Perturb parameter i by +eps
        params_plus = ccm_params.copy()
        params_plus[i] += eps
        loss_plus, _ = forward_pass(
            params_plus, rgb_mean_array, linear_lab_patch_target,
            lab_patch_weight, gamma_en, gamma_curve
        )

        # Perturb parameter i by -eps
        params_minus = ccm_params.copy()
        params_minus[i] -= eps
        loss_minus, _ = forward_pass(
            params_minus, rgb_mean_array, linear_lab_patch_target,
            lab_patch_weight, gamma_en, gamma_curve
        )

        # Central difference gradient
        gradient[i] = (loss_plus - loss_minus) / (2.0 * eps)

    return gradient


def calc_cc_matrix_gd_lab_error_no_bp(
    rgb_mean_array: Float32[np.ndarray, "18 3"],
    linear_lab_patch_target: Float32[np.ndarray, "18 3"],
    lab_patch_weight: Float32[np.ndarray, "18 3"],
    gamma_en: bool,
    gamma_curve: UInt8[np.ndarray, "256"],
    init_ccm: Float32[np.ndarray, "6"],
) -> Float32[np.ndarray, "3 3"]:
    """
    Optimize CCM using manual numerical gradient descent (no backpropagation)
    
    For each iteration:
        1. Compute numerical gradient for each of the 6 CCM parameters
        2. Update parameters using gradient descent
        3. Track best result
    
    Args:
        rgb_mean_array: RGB values for 18 patches
        linear_lab_patch_target: Target LAB values
        lab_patch_weight: Weights for each patch
        gamma_en: Whether gamma is enabled
        gamma_curve: Gamma correction lookup table
        init_ccm: Initial CCM parameters (6 values)
        
    Returns:
        Optimized CCM matrix (3x3)
    """
    # Initialize parameters
    ccm_params = init_ccm.copy().astype(np.float32)

    # Hyperparameters
    learning_rate = 0.0002
    num_epochs = 6000
    eps = 1e-5  # Numerical differentiation epsilon

    # Track best result
    best_loss = float('inf')
    best_ccm_params = ccm_params.copy()

    # Training loop
    for epoch in range(num_epochs):
        # Compute numerical gradient
        gradient = compute_numerical_gradient(
            ccm_params, rgb_mean_array, linear_lab_patch_target,
            lab_patch_weight, gamma_en, gamma_curve, eps
        )

        # Update parameters using gradient descent
        ccm_params = ccm_params - learning_rate * gradient

        # Ensure parameters stay in valid range [-16, 16]
        ccm_params = np.clip(ccm_params, -16.0, 16.0)

        # Compute current loss
        current_loss, patch_distances = forward_pass(
            ccm_params, rgb_mean_array, linear_lab_patch_target,
            lab_patch_weight, gamma_en, gamma_curve
        )

        # Track best result
        if current_loss < best_loss:
            best_loss = current_loss
            best_ccm_params = ccm_params.copy()

        # Print progress
        if epoch % 10 == 0:
            print(f"epoch={epoch + 1}, lab error={current_loss:.4f}, mean lab error={patch_distances.mean():.4f}")
            print(patch_distances)

    # Get the best CCM
    res = make_ccm_from_chromosome(best_ccm_params)
    res = (res * 1024).astype(np.int32)
    return res
