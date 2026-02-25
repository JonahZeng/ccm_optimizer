import numpy as np

def linear_rgb_to_lab(rgb):
    """
    rgb: float32 numpy array, shape (..., 3), values in [0,1]
         MUST be linear RGB (not sRGB)
    return: Lab array, same shape
    """

    # --- 1. Linear RGB → XYZ ---
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=np.float32)

    # reshape for matrix multiplication
    shape = rgb.shape
    rgb_flat = rgb.reshape(-1, 3)

    xyz = rgb_flat @ M.T

    # --- 2. Normalize by D65 white point ---
    Xn, Yn, Zn = 95.047, 100.0, 108.883
    x = xyz[:, 0] / Xn
    y = xyz[:, 1] / Yn
    z = xyz[:, 2] / Zn

    # --- 3. f(t) function ---
    def f(t):
        delta = 6/29
        return np.where(
            t > delta**3,
            np.cbrt(t),
            t / (3 * delta**2) + 4/29
        )

    fx = f(x)
    fy = f(y)
    fz = f(z)

    # --- 4. XYZ → Lab ---
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    lab = np.stack([L, a, b], axis=-1)
    return lab.reshape(shape)

if __name__ == '__main__' :
    rgb = np.array(
        [
            [116, 81, 67],
            [199, 147, 129],
            [91, 122, 156],
            [90, 108, 64],
            [130, 128, 176],
            [92, 190, 172],
            [224, 124, 47],
            [68, 91, 170],
            [198, 82, 97],
            [94, 58, 106],
            [159, 189, 63],
            [230, 162, 39],
            [35, 63, 147],
            [67, 149, 74],
            [180, 49, 57],
            [238, 198, 20],
            [193, 84, 151],
            [0, 136, 170],
        ]
    )
    lab = linear_rgb_to_lab(rgb)
    print(lab)