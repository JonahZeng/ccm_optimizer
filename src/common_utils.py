'''common algo'''
import numpy as np
from jaxtyping import UInt8

def rgb_to_lab(input_rgb: np.ndarray) -> np.ndarray:
    """
    convert 8bit linear rgb to lab
    http://www.brucelindbloom.com/index.html?ColorCalculator.html
    :param input_rgb: 输入rgb, 18个patch
    :return: 输出18个patch的 lab
    """
    if input_rgb.shape != (18, 3):
        raise ValueError("input_rgb shape != (18, 3)")
    # linear rgb [0, 255]
    lab_arr = np.zeros((18, 3), dtype=np.float64)
    out_r = input_rgb[:, 0] * 100.0 / 255.0
    out_g = input_rgb[:, 1] * 100.0 / 255.0
    out_b = input_rgb[:, 2] * 100.0 / 255.0
    out_X = out_r * 0.4124564 + out_g * 0.3575761 + out_b * 0.1804375
    out_Y = out_r * 0.2126729 + out_g * 0.7151522 + out_b * 0.0721750
    out_Z = out_r * 0.0193339 + out_g * 0.1191920 + out_b * 0.9503041
    # print(out_X.shape, out_Y.shape, out_Z.shape)
    # out_X Y Z > 0.9
    # print(out_X / 95.047, out_Y / 100.000, out_Z / 108.883)
    # out_X = (out_X / 95.047) ** (1 / 3)  # BT709 d65 white point x=0.3127, y=0.3290, z=0.3583
    # out_Y = (out_Y / 100.000) ** (1 / 3)
    # out_Z = (out_Z / 108.883) ** (1 / 3)
    out_X = np.cbrt(out_X / 95.047)
    out_Y = np.cbrt(out_Y / 100.000)
    out_Z = np.cbrt(out_Z / 108.883)
    lab_arr[:, 0] = (116 * out_Y) - 16
    lab_arr[:, 1] = 500 * (out_X - out_Y)
    lab_arr[:, 2] = 200 * (out_Y - out_Z)
    return lab_arr  # L [0, 100], ab: [-128, +128]


def make_ccm_from_chromosome(chromosome: np.ndarray) -> np.ndarray:
    """
    从染色体构建ccm矩阵, 一条染色体有6个数值,各自独立
    :param chromosome: 输入一条染色体
    :return: 输出对应的ccm
    """
    res = np.array([
        [1.0 - chromosome[0] - chromosome[1], chromosome[0], chromosome[1]],
        [chromosome[2], 1.0 - chromosome[2] - chromosome[3], chromosome[3]],
        [chromosome[4], chromosome[5], 1.0 - chromosome[4] - chromosome[5]]
    ], np.float32)
    return res

def calc_lab_distance(target_patch: np.ndarray, current_patch: np.ndarray, patch_weight: np.ndarray,
                      just_ab=True) -> tuple[float, np.ndarray]:
    """
    计算两个lab值的距离,分18个patch,可以配置不同的权重
    :param target_patch: 理想的lab值
    :param current_patch: 当前的lab值
    :param patch_weight: 18 patch的权重
    :param just_ab: 仅仅计算ab之间的距离,不考虑L
    :return: 输出lab距离
    """
    if target_patch.shape != (18, 3) or current_patch.shape != (18, 3) or patch_weight.shape != (18,):
        raise ValueError('lab patch != (18, 3)')
    dist = 0.0
    patch_dist = np.array((18,), dtype=np.float32)
    if just_ab:
        patch_dist = np.sqrt(
            (target_patch[:, 1] - current_patch[:, 1]) ** 2 + (target_patch[:, 2] - current_patch[:, 2]) ** 2)
        dist = np.sum(patch_dist * patch_weight)  # element-wise multiply
    else:
        patch_dist = np.sqrt(
            (target_patch[:, 0] - current_patch[:, 0]) ** 2 +
            (target_patch[:, 1] - current_patch[:, 1]) ** 2 +
            (target_patch[:, 2] - current_patch[:, 2]) ** 2)
        dist = np.sum(patch_dist * patch_weight)  # element-wise multiply
    return dist, patch_dist

def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        rgb (np.ndarray): _description_

    Returns:
        np.ndarray | None: _description_
    """
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"Expected shape (H, 3), got {rgb.shape}")
    rgb_uniform = rgb / 255.0
    l = 0.4122214708 * rgb_uniform[:, 0] + 0.5363325363 * rgb_uniform[:, 1] + 0.0514459929 * rgb_uniform[:, 2]
    a = 0.2119034982 * rgb_uniform[:, 0] + 0.6806995451 * rgb_uniform[:, 1] + 0.1073969566 * rgb_uniform[:, 2]
    b = 0.0883024619 * rgb_uniform[:, 0] + 0.2817188376 * rgb_uniform[:, 1] + 0.6299787005 * rgb_uniform[:, 2]
    l = np.cbrt(l)
    a = np.cbrt(a)
    b = np.cbrt(b)
    l_ = 0.2104542553 * l + 0.7936177850 * a - 0.0040720468 * b
    a_ = 1.9779984951 * l - 2.4285922050 * a + 0.4505937099 * b
    b_ = 0.0259040371 * l + 0.7827717662 * a - 0.8086757660 * b
    return np.stack([l_, a_, b_], axis=1)

def oklab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        rgb (np.ndarray): _description_

    Returns:
        np.ndarray | None: _description_
    """
    if lab.ndim != 2 or lab.shape[1] != 3:
        raise ValueError(f"Expected shape (H, 3), got {lab.shape}")
    l_ = lab[:, 0] + 0.3963377774 * lab[:, 1] + 0.2158037573 * lab[:, 2]
    m_ = lab[:, 0] - 0.1055613458 * lab[:, 1] - 0.0638541728 * lab[:, 2]
    s_ = lab[:, 0] - 0.0894841775 * lab[:, 1] - 1.2914855480 * lab[:, 2]

    l = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_

    r_ccm = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g_ccm = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b_ccm = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return np.stack([r_ccm, g_ccm, b_ccm], axis=1)

def inv_gamma_vectorized(rgb: np.ndarray, gamma_curve: UInt8[np.ndarray, "256"]) -> np.ndarray:
    """
    Apply inverse gamma correction using vectorized operations.
    
    Args:
        rgb (np.ndarray): Input RGB values, shape (H, W) or (H, 3) or (H, W, 3)
        gamma_curve (np.ndarray): Gamma correction lookup table, shape (256,)
        
    Returns:
        np.ndarray: Inverse gamma corrected values as float32
    """
    rgb_res = np.round(rgb).astype(np.uint8)
    original_shape = rgb_res.shape
    rgb_flat = rgb_res.reshape(-1)
    indices = np.searchsorted(gamma_curve, rgb_flat, side='right') - 1
    indices = np.clip(indices, 0, 255)
    rgb_res = indices.reshape(original_shape)
    return rgb_res.astype(np.float32)

def inv_gamma(rgb: np.ndarray, gamma_curve: UInt8[np.ndarray, "256"]) -> np.ndarray:
    """_summary_

    Args:
        rgb (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    rgb_res = np.round(rgb).astype(np.uint8)
    for n in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            for i in range(255):
                if gamma_curve[i] <= rgb_res[n, j] < gamma_curve[i + 1]:
                    rgb_res[n, j] = i
                    break
                if rgb_res[n, j] == gamma_curve[255]:
                    rgb_res[n, j] = 255
    return rgb_res.astype(np.float32)
