"""optim ccm by torch gradient descent"""

import numpy as np
from jaxtyping import Float32, UInt8
import torch
from torch import nn
from torch.optim.adam import Adam
from src.common_utils import (
    rgb_to_oklab,
    make_ccm_from_chromosome
)

class CcmGD(nn.Module):
    """Gradient descent optimizer for CCM parameters"""

    def __init__(
        self,
        init_ccm: Float32[np.ndarray, "6"],
        rgb_mean_array: Float32[np.ndarray, "18 3"],
        linear_lab_patch_target: Float32[np.ndarray, "18 3"],
        lab_patch_weight: Float32[np.ndarray, "18"],
        gamma_en: bool,
        gamma_curve: UInt8[np.ndarray, "256"],
    ):
        super().__init__()

        self.init_ccm = nn.Parameter(torch.from_numpy(init_ccm).float())

        self.register_buffer('linear_lab_patch_target', torch.from_numpy(linear_lab_patch_target).float())
        self.register_buffer('lab_patch_weight', torch.from_numpy(lab_patch_weight).float())
        self.register_buffer('gamma_curve', torch.from_numpy(gamma_curve).float())
        self.gamma_en = gamma_en

        rgb_18_patch = np.clip(rgb_mean_array, 0.0, 255.0)
        oklab_18_patch = rgb_to_oklab(rgb_18_patch)
        self.register_buffer('oklab_18_patch', torch.from_numpy(oklab_18_patch).float())
        self.register_buffer('rgb_18_patch', torch.from_numpy(rgb_18_patch.transpose()).float())  # (3, 18)

    def make_ccm_from_chromosome_torch(self, chromosome: torch.Tensor) -> torch.Tensor:
        """Build CCM matrix from chromosome using PyTorch"""
        res = torch.stack([
            torch.stack([1.0 - chromosome[0] - chromosome[1], chromosome[0], chromosome[1]]),
            torch.stack([chromosome[2], 1.0 - chromosome[2] - chromosome[3], chromosome[3]]),
            torch.stack([chromosome[4], chromosome[5], 1.0 - chromosome[4] - chromosome[5]])
        ])
        return res

    def rgb_to_oklab_torch(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB to OKLab using PyTorch"""
        rgb_uniform = rgb / 255.0
        l = 0.4122214708 * rgb_uniform[:, 0] + 0.5363325363 * rgb_uniform[:, 1] + 0.0514459929 * rgb_uniform[:, 2]
        a = 0.2119034982 * rgb_uniform[:, 0] + 0.6806995451 * rgb_uniform[:, 1] + 0.1073969566 * rgb_uniform[:, 2]
        b = 0.0883024619 * rgb_uniform[:, 0] + 0.2817188376 * rgb_uniform[:, 1] + 0.6299787005 * rgb_uniform[:, 2]
        l = torch.pow(l, 1.0/3.0)
        a = torch.pow(a, 1.0/3.0)
        b = torch.pow(b, 1.0/3.0)
        l_ = 0.2104542553 * l + 0.7936177850 * a - 0.0040720468 * b
        a_ = 1.9779984951 * l - 2.4285922050 * a + 0.4505937099 * b
        b_ = 0.0259040371 * l + 0.7827717662 * a - 0.8086757660 * b
        return torch.stack([l_, a_, b_], dim=1)

    def oklab_to_rgb_torch(self, lab: torch.Tensor) -> torch.Tensor:
        """Convert OKLab to RGB using PyTorch"""
        l_ = lab[:, 0] + 0.3963377774 * lab[:, 1] + 0.2158037573 * lab[:, 2]
        m_ = lab[:, 0] - 0.1055613458 * lab[:, 1] - 0.0638541728 * lab[:, 2]
        s_ = lab[:, 0] - 0.0894841775 * lab[:, 1] - 1.2914855480 * lab[:, 2]

        l = l_ * l_ * l_
        m = m_ * m_ * m_
        s = s_ * s_ * s_

        r_ccm = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
        g_ccm = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
        b_ccm = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
        return torch.stack([r_ccm, g_ccm, b_ccm], dim=1)

    def rgb_to_lab_torch(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB to Lab using PyTorch"""
        lab_arr = torch.zeros((18, 3), dtype=torch.float32, device=rgb.device)
        out_r = rgb[:, 0] * 100.0 / 255.0
        out_g = rgb[:, 1] * 100.0 / 255.0
        out_b = rgb[:, 2] * 100.0 / 255.0
        out_X = out_r * 0.4124564 + out_g * 0.3575761 + out_b * 0.1804375
        out_Y = out_r * 0.2126729 + out_g * 0.7151522 + out_b * 0.0721750
        out_Z = out_r * 0.0193339 + out_g * 0.1191920 + out_b * 0.9503041
        out_X = torch.pow(out_X / 95.047, 1.0/3.0)
        out_Y = torch.pow(out_Y / 100.000, 1.0/3.0)
        out_Z = torch.pow(out_Z / 108.883, 1.0/3.0)
        lab_arr[:, 0] = (116 * out_Y) - 16
        lab_arr[:, 1] = 500 * (out_X - out_Y)
        lab_arr[:, 2] = 200 * (out_Y - out_Z)
        return lab_arr

    def inv_gamma_torch(self, rgb: torch.Tensor) -> torch.Tensor:
        """Apply inverse gamma correction using PyTorch"""
        # gamma_curve = self.gamma_curve.to(rgb.device)
        # rgb_flat = rgb.flatten()
        # indices = torch.searchsorted(gamma_curve, rgb_flat, right=True) - 1
        # indices = torch.clamp(indices, 0, 255)
        # rgb_res = indices.reshape(rgb.shape)

        rgb_res = torch.pow(rgb / 255.0, 2.2) * 255.0
        return rgb_res

    def calc_lab_distance_torch(self, target_patch: torch.Tensor, current_patch: torch.Tensor,
                                patch_weight: torch.Tensor, just_ab: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate LAB distance using PyTorch"""
        if just_ab:
            patch_dist = torch.sqrt(
                (target_patch[:, 1] - current_patch[:, 1]) ** 2 +
                (target_patch[:, 2] - current_patch[:, 2]) ** 2
            )
            dist = torch.sum(patch_dist * patch_weight)
        else:
            patch_dist = torch.sqrt(
                (target_patch[:, 0] - current_patch[:, 0]) ** 2 +
                (target_patch[:, 1] - current_patch[:, 1]) ** 2 +
                (target_patch[:, 2] - current_patch[:, 2]) ** 2
            )
            dist = torch.sum(patch_dist * patch_weight)
        return dist, patch_dist

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: compute LAB distance loss"""
        # Build CCM from chromosome
        ccm = self.make_ccm_from_chromosome_torch(self.init_ccm)

        # Apply CCM to RGB
        ccm_18_rgb = torch.clamp(torch.matmul(ccm, self.rgb_18_patch), 0.0, 255.0)  # (3, 3) * (3, 18) => (3, 18)
        ccm_18_rgb = ccm_18_rgb.transpose(0, 1)  # (18, 3)

        # Convert to OKLab
        ccm_18_oklab = self.rgb_to_oklab_torch(ccm_18_rgb)

        # Blend L channel (keep original L)
        ccm_18_oklab[:, 0] = self.oklab_18_patch[:, 0]

        # Convert back to RGB
        ccm_18_rgb = self.oklab_to_rgb_torch(ccm_18_oklab)
        ccm_18_rgb = torch.clamp(ccm_18_rgb * 255.0, 0.0, 255.0)

        # Apply inverse gamma if needed
        if not self.gamma_en:
            ccm_18_rgb = self.inv_gamma_torch(ccm_18_rgb)

        # Convert to Lab
        ccm_18_lab = self.rgb_to_lab_torch(ccm_18_rgb)

        # Calculate LAB distance
        lab_distance, lab_patch_distance = self.calc_lab_distance_torch(
            self.linear_lab_patch_target, ccm_18_lab, self.lab_patch_weight, just_ab=True
        )

        return lab_distance, lab_patch_distance


def calc_cc_matrix_gd_lab_error(
    rgb_mean_array: Float32[np.ndarray, "18 3"],
    linear_lab_patch_target: Float32[np.ndarray, "18 3"],
    lab_patch_weight: Float32[np.ndarray, "18 3"],
    gamma_en: bool,
    gamma_curve: UInt8[np.ndarray, "256"],
    init_ccm: Float32[np.ndarray, "6"],
) -> Float32[np.ndarray, "3 3"]:
    """optimize ccm in torch gradient descent"""
    # Create the CcmGD model
    model = CcmGD(
        init_ccm=init_ccm,
        rgb_mean_array=rgb_mean_array,
        linear_lab_patch_target=linear_lab_patch_target,
        lab_patch_weight=lab_patch_weight,
        gamma_en=gamma_en,
        gamma_curve=gamma_curve,
    )

    # Set up optimizer
    optimizer = Adam(model.parameters(), lr=0.002)

    # Training loop
    num_epochs = 6000
    best_loss = float('inf')
    best_ccm = model.init_ccm.detach().clone() # pylint: disable=not-callable

    for epoch in range(num_epochs):
        # Forward pass
        optimizer.zero_grad()
        lab_distance, lab_patch_distance = model.forward()

        # Backward pass
        lab_distance.backward()
        optimizer.step()

        # Track best result
        current_loss = lab_distance.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_ccm = model.init_ccm.detach().clone() # pylint: disable=not-callable

        if epoch % 10 == 0:
            print(f"epoch={epoch + 1}, lab error={current_loss:.4f}, mean lab error={lab_patch_distance.mean().item():.4f}")
            print(lab_patch_distance.detach().numpy())

    # Get the best CCM
    res = make_ccm_from_chromosome(best_ccm.cpu().numpy())
    res = (res * 1024).astype(np.int32)
    return res
