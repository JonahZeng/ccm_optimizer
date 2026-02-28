'''
formMain is main window for ccm optimization
'''

import json
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from PySide6.QtWidgets import QWidget, QApplication, QMessageBox, QFileDialog, QDialog
from PySide6.QtGui import QIcon
from PySide6.QtCore import QFileInfo, QSettings, Signal, Slot, QCoreApplication
from src import demosaic # type: ignore
from src.ui_cc_calib import Ui_eCCM
from src.ui_rawinfo import Ui_rawInfoDlg
from src.common_utils import (
    oklab_to_rgb, rgb_to_oklab, calc_lab_distance, make_ccm_from_chromosome, rgb_to_lab, inv_gamma_vectorized
)
from src.gd_optim import calc_cc_matrix_gd_lab_error
from src.gd_optim_without_bp import calc_cc_matrix_gd_lab_error_no_bp


class RawInfoDlg(QDialog):
    """
    raw fileformat info dialog
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.ui = Ui_rawInfoDlg()
        self.ui.setupUi(self)


class ECcmFrom(QWidget):
    """
    main window
    """
    inform_wb_gain = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.ui = Ui_eCCM()
        self.ui.setupUi(self)
        self.raw_data: NDArray = np.array([0, 0], np.uint16)
        self.bgr_data: NDArray = np.array([0, 0], np.uint16)
        self.raw_bayer: str = 'RGGB'
        self.raw_bits: int = 12
        self.blc: int = 168
        self.linear_rgb_patch_target: Optional[NDArray] = None  # np.zeros((18,), np.uint8)
        self.linear_lab_patch_target: Optional[NDArray] = None  # np.zeros((18,), np.float64)
        self.lab_patch_weight: Optional[NDArray] = None
        self.gamma_en = True
        self.gamma_curve = np.arange(256, dtype=np.uint8)
        self.inform_wb_gain.connect(self.ui.rawPreviewWidget.apply_wb)
        self.setWindowIcon(QIcon('./img/icon.png'))

    @Slot()
    def import_raw_file(self) -> None:
        """
        open raw file
        """
        in_raw = QFileDialog.getOpenFileName(self, "Open raw file", "./", "Raw file (*.raw *.RAW);;All file (*.*)")
        if not in_raw[0].endswith('.raw') and not in_raw[0].endswith('.RAW'):
            return
        raw_info_dlg = RawInfoDlg(self)
        settings = QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope, 'pISP')
        # print(settings.fileName())
        if settings.contains("rawBits"):
            raw_info_dlg.ui.rawBitComboBox.setCurrentText(str(settings.value("rawBits")))
        if settings.contains("rawWidth"):
            raw_info_dlg.ui.rawWidthLine.setText(str(settings.value("rawWidth")))
        if settings.contains("rawHeight"):
            raw_info_dlg.ui.rawHeightLine.setText(str(settings.value("rawHeight")))
        if settings.contains("rawBayer"):
            raw_info_dlg.ui.bayerComboBox.setCurrentText(settings.value("rawBayer"))
        if settings.contains("blc"):
            raw_info_dlg.ui.blcLineEdit.setText(str(settings.value("blc")))
        reply = raw_info_dlg.exec()
        if reply != QDialog.DialogCode.Accepted:
            return
        self.raw_bits = int(raw_info_dlg.ui.rawBitComboBox.currentText())
        self.raw_bayer = raw_info_dlg.ui.bayerComboBox.currentText()
        self.blc = int(raw_info_dlg.ui.blcLineEdit.text())
        raw_width = int(raw_info_dlg.ui.rawWidthLine.text())
        raw_height = int(raw_info_dlg.ui.rawHeightLine.text())

        settings.setValue("rawBits", self.raw_bits)
        settings.setValue("rawWidth", raw_width)
        settings.setValue("rawHeight", raw_height)
        settings.setValue("rawBayer", self.raw_bayer)
        settings.setValue("blc", self.blc)

        raw_size = QFileInfo(in_raw[0]).size()
        if raw_size != raw_width * raw_height * (2 if self.raw_bits > 8 else 1):
            QMessageBox.critical(self, "error", "raw file size != your input", QMessageBox.StandardButton.Ok,
                                 QMessageBox.StandardButton.Cancel)

        self.raw_data = np.fromfile(in_raw[0], (np.uint16 if self.raw_bits > 8 else np.uint8)).reshape(
            (raw_height, raw_width)) - self.blc
        self.show_bgr_on_preview()

    def show_bgr_on_preview(self) -> None:
        """
        show raw image on widget
        """
        assert isinstance(self.raw_data, np.ndarray)
        raw_data_b = (self.raw_data >> (self.raw_bits - 8)).astype(np.uint8)
        # Use our C extension for demosaicing
        self.bgr_data = demosaic.bayer2bgr(raw_data_b, self.raw_bayer) # pylint: disable=c-extension-no-member
        assert isinstance(self.bgr_data, np.ndarray)
        self.ui.rawPreviewWidget.bgr_data = self.bgr_data
        self.ui.rawPreviewWidget.bgr_data_preview = self.bgr_data.copy()
        self.ui.rawPreviewWidget.wb_is_done = False
        self.ui.rawPreviewWidget.update()

    @Slot() # type: ignore
    def show_preview_bright(self, val: int):
        """
        adjust raw image preview brightness
        Args:
            val (int): brightness vaue
        """
        self.ui.previewBrightLabel.setText(f"preview brightness: {val / 10.0: .2f}")

    def calc_cc_matrix_traditional(self, rgb_mean_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """

        Args:
            rgb_mean_array (np.ndarray): color checker 18 patch bgr mean value

        Returns:
            tuple[np.ndarray, np.ndarray]: ccm int param, ccm float param
        """
        res = np.array([[1024, 0, 0], [0, 1024, 0], [0, 0, 1024]], np.int32)
        res_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.float64)
        if self.linear_rgb_patch_target is None or rgb_mean_array is None:
            print('error, no linear rgb target value')
            return res, res_init
        rgb_mean_array = rgb_mean_array[:3, :].reshape((18, 3))
        assert isinstance(self.lab_patch_weight, np.ndarray)
        patch_w = np.diag(self.lab_patch_weight) # (18, 18)

        b = self.linear_rgb_patch_target[:, 0] - rgb_mean_array[:, 0]
        A = np.zeros((18, 2), dtype=np.float64)
        A[:, 0] = rgb_mean_array[:, 1] - rgb_mean_array[:, 0]
        A[:, 1] = rgb_mean_array[:, 2] - rgb_mean_array[:, 0]
        At = np.transpose(A)
        line0 = ((np.linalg.inv(At.dot(patch_w).dot(A))).dot(At).dot(patch_w).dot(b) * 1024).astype(np.int32) # ((2, 18) * (18, 2)) * (2, 18) * (18, 1) = (2, 1)
        res[0, 0] = 1024 - line0[0] - line0[1]
        res[0, 1] = line0[0]
        res[0, 2] = line0[1]

        b = self.linear_rgb_patch_target[:, 1] - rgb_mean_array[:, 1]
        A = np.zeros((18, 2), dtype=np.float64)
        A[:, 0] = rgb_mean_array[:, 0] - rgb_mean_array[:, 1]
        A[:, 1] = rgb_mean_array[:, 2] - rgb_mean_array[:, 1]
        At = np.transpose(A)
        line1 = ((np.linalg.inv(At.dot(patch_w).dot(A))).dot(At).dot(patch_w).dot(b) * 1024).astype(np.int32)
        res[1, 0] = line1[0]
        res[1, 1] = 1024 - line1[0] - line1[1]
        res[1, 2] = line1[1]

        b = self.linear_rgb_patch_target[:, 2] - rgb_mean_array[:, 2]
        A = np.zeros((18, 2), dtype=np.float64)
        A[:, 0] = rgb_mean_array[:, 0] - rgb_mean_array[:, 2]
        A[:, 1] = rgb_mean_array[:, 1] - rgb_mean_array[:, 2]
        At = np.transpose(A)
        line2 = ((np.linalg.inv(At.dot(patch_w).dot(A))).dot(At).dot(patch_w).dot(b) * 1024).astype(np.int32)
        res[2, 0] = line2[0]
        res[2, 1] = line2[1]
        res[2, 2] = 1024 - line2[0] - line2[1]
        return res, np.array([line0[0], line0[1], line1[0], line1[1], line2[0], line2[1]], np.float64) / 1024.0

    def init_ccm_in_random_lab_space(self, rgb_mean_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """_summary_

        Args:
            rgb_mean_array (np.ndarray): _description_

        Returns:
            tuple[np.ndarray, np.ndarray]: _description_
        """
        if self.linear_lab_patch_target is None or rgb_mean_array is None:
            print('error, no linear lab target value')
            return (np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32),
                    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32))

        rgb_18_patch = rgb_mean_array[:3, :].reshape((18, 3))  # 18 个patch rgb
        rgb_18_patch = np.clip(rgb_18_patch, 0.0, 255.0) # clip (18, 3)
        oklab_18_patch = rgb_to_oklab(rgb_18_patch) # (18, 3)
        rgb_18_patch = rgb_18_patch.transpose()  # ccm * rgb^t => shape(3, 18)
        assert isinstance(self.lab_patch_weight, np.ndarray)

        step_unit = 1.0 # [-4, 4] step = 1.0
        init_ccm = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        best_lab_distance = 2**32
        best_i = -1
        best_j = -1
        best_k = -1
        best_l = -1
        best_m = -1
        best_n = -1
        # ---------
        sec_best_lab_distance = 2**32
        sec_best_i = -1
        sec_best_j = -1
        sec_best_k = -1
        sec_best_l = -1
        sec_best_m = -1
        sec_best_n = -1
        for i in range(9):
            for j in range(9):
                for k in range(9):
                    for l in range(9):
                        print(i*(9**5) + j*(9**4) + k*(9**3) + l*(9**2), "total(531441)")
                        for m in range(9):
                            for n in range(9):
                                init_ccm[0] = i * step_unit - 4.0
                                init_ccm[1] = j * step_unit - 4.0
                                init_ccm[2] = k * step_unit - 4.0
                                init_ccm[3] = l * step_unit - 4.0
                                init_ccm[4] = m * step_unit - 4.0
                                init_ccm[5] = n * step_unit - 4.0
                                ccm = make_ccm_from_chromosome(init_ccm)
                                ccm_18_rgb: np.ndarray = np.matmul(ccm, rgb_18_patch)  # (3, 3) * (3, 18) => (3, 18)
                                ccm_18_rgb = ccm_18_rgb.transpose()  # (18, 3)
                                ccm_18_oklab = rgb_to_oklab(ccm_18_rgb) # (18, 3)
                                ccm_18_oklab[:, 0] = oklab_18_patch[:, 0] # blend L
                                ccm_18_rgb = oklab_to_rgb(ccm_18_oklab) # inv oklab --> rgb
                                ccm_18_rgb = np.clip(ccm_18_rgb * 255.0, 0.0, 255.0)
                                if not self.gamma_en: # json 配置gamm_en = false的时候，说明此时rgb转lab需要逆gamma
                                    ccm_18_rgb = inv_gamma_vectorized(ccm_18_rgb, self.gamma_curve)
                                ccm_18_lab = rgb_to_lab(ccm_18_rgb)
                                lab_distance, _ = calc_lab_distance(
                                    self.linear_lab_patch_target, ccm_18_lab, self.lab_patch_weight
                                )
                                if lab_distance < best_lab_distance:
                                    sec_best_lab_distance, best_lab_distance = best_lab_distance, lab_distance
                                    sec_best_i, best_i = best_i, i
                                    sec_best_j, best_j = best_j, j
                                    sec_best_k, best_k = best_k, k
                                    sec_best_l, best_l = best_l, l
                                    sec_best_m, best_m = best_m, m
                                    sec_best_n, best_n = best_n, n
                                elif lab_distance < sec_best_lab_distance:
                                    sec_best_lab_distance = lab_distance
                                    sec_best_i = i
                                    sec_best_j = j
                                    sec_best_k = k
                                    sec_best_l = l
                                    sec_best_m = m
                                    sec_best_n = n
        ccm0 = np.array([step_unit * best_i,
                         step_unit * best_j,
                         step_unit * best_k,
                         step_unit * best_l,
                         step_unit * best_m,
                         step_unit * best_n], dtype=np.float32)
        ccm1 = np.array([step_unit * sec_best_i,
                         step_unit * sec_best_j,
                         step_unit * sec_best_k,
                         step_unit * sec_best_l,
                         step_unit * sec_best_m,
                         step_unit * sec_best_n], dtype=np.float32)
        return ccm0, ccm1

    def calc_cc_matrix_lab_space(self,
                                 rgb_mean_array: np.ndarray,
                                 init_ccm0: np.ndarray,
                                 init_ccm1: np.ndarray) -> np.ndarray:
        """
        使用遗传算法迭代优化ccm
        :param bgr_mean_array: 输入色卡bgr均值, [0, 255] 8bit均值, 浮点表示
        :param init_ccm0: 初始化cc参数, 1.0表示1x
        :param init_ccm1: 初始化cc参数, 1.0表示1x
        :return: ccm结果
        """
        if self.linear_lab_patch_target is None or rgb_mean_array is None:
            print('error, no linear lab target value')
            return np.array([[1024, 0, 0], [0, 1024, 0], [0, 0, 1024]], np.int32)
        rgb_18_patch = rgb_mean_array[:3, :].reshape((18, 3))  # 18 个patch rgb
        rgb_18_patch = np.clip(rgb_18_patch, 0.0, 255.0) # clip (18, 3)
        oklab_18_patch = rgb_to_oklab(rgb_18_patch) # (18, 3)
        rgb_18_patch = rgb_18_patch.transpose()  # ccm * rgb^t => shape(3, 18)
        origin_ccm_arr = 6.0 * (np.random.rand(9, 6) - 0.5) # 7组ccm，每组ccm长度6（剩余的3个值通过约束求出）
        origin_ccm_arr[0] = init_ccm0 # 初始化第一组
        origin_ccm_arr[1] = init_ccm1 # 初始化第二组

        epoch = 50000
        while epoch > 0:
            ccm_list: list[np.ndarray] = [make_ccm_from_chromosome(origin_ccm_arr[0]),
                                          make_ccm_from_chromosome(origin_ccm_arr[1]),
                                          make_ccm_from_chromosome(origin_ccm_arr[2]),
                                          make_ccm_from_chromosome(origin_ccm_arr[3]),
                                          make_ccm_from_chromosome(origin_ccm_arr[4]),
                                          make_ccm_from_chromosome(origin_ccm_arr[5]),
                                          make_ccm_from_chromosome(origin_ccm_arr[6]),
                                          make_ccm_from_chromosome(origin_ccm_arr[7]),
                                          make_ccm_from_chromosome(origin_ccm_arr[8])]
            lab_distance = np.zeros((9, ), np.float64)
            lab_patch_distance = np.zeros((9, 18), np.float64)
            assert isinstance(self.lab_patch_weight, np.ndarray)
            for idx, ccm in enumerate(ccm_list):
                ccm_18_rgb: np.ndarray = np.matmul(ccm, rgb_18_patch)  # (3, 3) * (3, 18) => (3, 18)
                ccm_18_rgb = ccm_18_rgb.transpose()  # (18, 3)
                ccm_18_oklab = rgb_to_oklab(ccm_18_rgb) # (18, 3)
                ccm_18_oklab[:, 0] = oklab_18_patch[:, 0] # blend L
                ccm_18_rgb = oklab_to_rgb(ccm_18_oklab) # inv oklab --> rgb
                ccm_18_rgb = np.clip(ccm_18_rgb * 255.0, 0.0, 255.0)
                if not self.gamma_en: # json 配置gamm_en = false的时候，说明此时rgb转lab需要逆gamma
                    ccm_18_rgb = inv_gamma_vectorized(ccm_18_rgb, self.gamma_curve)
                ccm_18_lab = rgb_to_lab(ccm_18_rgb)
                lab_distance[idx], lab_patch_distance[idx] = calc_lab_distance(
                    self.linear_lab_patch_target, ccm_18_lab, self.lab_patch_weight
                )
            sorted_idx = np.argsort(lab_distance)  # distance从小到大的索引
            if epoch % 1000 == 0: # or epoch == 1:
                print(f'epoch={epoch}, '
                      f'best lab error={lab_distance[sorted_idx[0]]}, '
                      f'mean lab error={np.mean(lab_patch_distance[sorted_idx[0]])}')
                print(lab_patch_distance[sorted_idx[0]])
                # print_epoch_lab_error(self.linear_lab_patch_target, ccm_18_lab, self.lab_patch_weight)
            # 更新origin_ccm_arr, 距离最小的两组不变，剩下5组丢弃，然后前两组分别组合，变异，得到新的5组
            origin_ccm_arr[0], origin_ccm_arr[1] = origin_ccm_arr[sorted_idx[0]], origin_ccm_arr[sorted_idx[1]]
            origin_ccm_arr[2] = (origin_ccm_arr[0] + origin_ccm_arr[1]) / 2.0
            origin_ccm_arr[3, 0::2] = origin_ccm_arr[0, 0::2]
            origin_ccm_arr[3, 1::2] = origin_ccm_arr[1, 1::2]
            origin_ccm_arr[4, 0::2] = origin_ccm_arr[0, 1::2]
            origin_ccm_arr[4, 1::2] = origin_ccm_arr[1, 0::2]

            origin_ccm_arr[5, 0:3] = origin_ccm_arr[0, 0:3]
            origin_ccm_arr[5, 3:6] = origin_ccm_arr[1, 3:6]
            origin_ccm_arr[6, 0:3] = origin_ccm_arr[1, 0:3]
            origin_ccm_arr[6, 3:6] = origin_ccm_arr[0, 3:6]

            # 剩余两组ccm参数采用复制，然后随机变异变6个值中的两个
            origin_ccm_arr[7], origin_ccm_arr[8] = origin_ccm_arr[0], origin_ccm_arr[1]
            variation_pos = np.random.randint(low=0, high=6, size=(4, ))
            variation_val = 12.0 * (np.random.rand(4) - 0.5)
            origin_ccm_arr[7, variation_pos[0]] = variation_val[0]
            origin_ccm_arr[7, variation_pos[1]] = variation_val[1]
            origin_ccm_arr[8, variation_pos[2]] = variation_val[2]
            origin_ccm_arr[8, variation_pos[3]] = variation_val[3]
            epoch -= 1
        # 选择最优的结果
        res = make_ccm_from_chromosome(origin_ccm_arr[0])
        res = (res * 1024).astype(np.int32)
        return res

    @Slot()
    def calculate_ccm(self) -> None:
        """
        calculate color correction matrix param
        :return 
        """
        if self.bgr_data is None:
            return
        if self.ui.rawPreviewWidget.start_pt == self.ui.rawPreviewWidget.end_pt:
            return
        patch_size = self.ui.rawPreviewWidget.patch_size  # float
        start_pt_x = self.bgr_data.shape[1] * self.ui.rawPreviewWidget.start_x_percent
        start_pt_y = self.bgr_data.shape[0] * self.ui.rawPreviewWidget.start_y_percent
        end_pt_x = self.bgr_data.shape[1] * self.ui.rawPreviewWidget.end_x_percent
        end_pt_y = self.bgr_data.shape[0] * self.ui.rawPreviewWidget.end_y_percent
        x_step = (end_pt_x - start_pt_x) / 6
        y_step = (end_pt_y - start_pt_y) / 4
        gap = (1.0 - patch_size) / 2
        rgb_mean_array = np.zeros((4, 6, 3), dtype=np.float32)
        assert isinstance(self.raw_data, np.ndarray)
        for row in range(1, 5):
            p0_y = int(start_pt_y + (row - 1) * y_step + y_step * gap)
            p1_y = int(p0_y + y_step * patch_size)
            p0_y = (p0_y + 1) if p0_y & 1 == 1 else p0_y
            p1_y = (p1_y + 1) if p1_y & 1 == 1 else p1_y
            for col in range(1, 7):
                p0_x = int(start_pt_x + (col - 1) * x_step + x_step * gap)
                p1_x = int(p0_x + x_step * patch_size)
                p0_x = (p0_x + 1) if p0_x & 1 == 1 else p0_x
                p1_x = (p1_x + 1) if p1_x & 1 == 1 else p1_x
                r, gr, gb, b = self.raw_data[p0_y:p1_y:2, p0_x:p1_x:2], self.raw_data[p0_y:p1_y:2, p0_x + 1:p1_x:2], \
                    self.raw_data[p0_y + 1:p1_y:2, p0_x:p1_x:2], self.raw_data[p0_y + 1:p1_y:2, p0_x + 1:p1_x:2]
                if self.raw_bayer == 'RGGB':
                    pass
                elif self.raw_bayer == 'GRBG':
                    gr, r, b, gb = r, gr, gb, b
                elif self.raw_bayer == 'GBRG':
                    gb, b, r, gr = r, gr, gb, b
                elif self.raw_bayer == 'BGGR':
                    b, gb, gr, r = r, gr, gb, b
                r_mean = np.mean(r, dtype=np.float32)
                g_mean = (np.mean(gr, dtype=np.float32) + np.mean(gb, dtype=np.float32)) / 2
                b_mean = np.mean(b, dtype=np.float32)
                rgb_mean_array[row - 1, col - 1, 0] = r_mean
                rgb_mean_array[row - 1, col - 1, 1] = g_mean
                rgb_mean_array[row - 1, col - 1, 2] = b_mean
        # print(bgr_mean_array)
        r_gain0 = rgb_mean_array[3, 1, 1] / rgb_mean_array[3, 1, 0]
        b_gain0 = rgb_mean_array[3, 1, 1] / rgb_mean_array[3, 1, 2]
        r_gain1 = rgb_mean_array[3, 2, 1] / rgb_mean_array[3, 2, 0]
        b_gain1 = rgb_mean_array[3, 2, 1] / rgb_mean_array[3, 2, 2]
        r_gain2 = rgb_mean_array[3, 3, 1] / rgb_mean_array[3, 3, 0]
        b_gain2 = rgb_mean_array[3, 3, 1] / rgb_mean_array[3, 3, 2]
        r_gain3 = rgb_mean_array[3, 4, 1] / rgb_mean_array[3, 4, 0]
        b_gain3 = rgb_mean_array[3, 4, 1] / rgb_mean_array[3, 4, 2]
        r_gain = (b_gain0 + b_gain1 + b_gain2 + b_gain3) / 4.0
        b_gain = (r_gain0 + r_gain1 + r_gain2 + r_gain3) / 4.0
        self.inform_wb_gain.emit(b_gain, r_gain)

        cur_app = QApplication.instance()
        assert isinstance(cur_app, QCoreApplication)
        cur_app.processEvents()

        rgb_scl = 16.0
        rgb_mean_array[:, :, 0] = (rgb_mean_array[:, :, 0] * r_gain) / rgb_scl
        rgb_mean_array[:, :, 1] = rgb_mean_array[:, :, 1] / rgb_scl
        rgb_mean_array[:, :, 2] = (rgb_mean_array[:, :, 2] * b_gain) / rgb_scl

        print('请注意这里的rgb_scl, 改成和你的raw曝光相符的系数')
        print('目标色卡第四行的灰度值(8bit)依次为 245, 200, 161, 121, 82, 49')
        print('反gamma后的值(8bit)为 232, 148, 91, 48, 21, 7')
        print('你的raw色卡第四行值为：')
        print(rgb_mean_array[3, :, :])

        ccm0, init_cc = self.calc_cc_matrix_traditional(rgb_mean_array)  # 8bit
        print('使用rgb线性转换到rgb，最小二乘法的结果：')
        print(ccm0)
        # init_cc = np.array([0.0, 0.0, 0.0, 0., 0.0, 0.0], np.float32)
        print('初始cc参数', init_cc)
        # ccm1 = self.calc_cc_matrix_lab_space(rgb_mean_array, init_cc, init_cc)

        rgb_mean_array_18 = rgb_mean_array[:3, :].reshape((18, 3))
        assert isinstance(self.linear_lab_patch_target, np.ndarray) and isinstance(self.lab_patch_weight, np.ndarray)
        if self.ui.optimizeModeComboBox.currentIndex() ==  0:
            ccm1 = calc_cc_matrix_gd_lab_error(rgb_mean_array_18, self.linear_lab_patch_target, self.lab_patch_weight, self.gamma_en, self.gamma_curve, init_cc)
        elif self.ui.optimizeModeComboBox.currentIndex() == 1:
            ccm1 = calc_cc_matrix_gd_lab_error_no_bp(rgb_mean_array_18, self.linear_lab_patch_target, self.lab_patch_weight, self.gamma_en, self.gamma_curve, init_cc)
        else:
            QMessageBox.critical(self, "error", "请选择正确的优化模式", QMessageBox.StandardButton.Ok, QMessageBox.StandardButton.Cancel)
            return

        # init_cc0, init_cc1 = self.init_ccm_in_random_lab_space(rgb_mean_array)
        # print('初始cc参数', init_cc)
        # ccm1 = self.calc_cc_matrix_lab_space(rgb_mean_array, init_cc0, init_cc1)
        print('使用rgb转换到lab，迭代优化的结果：')
        print(ccm1)

    @Slot()
    def import_target_file(self) -> None:
        """
        import setting from json file
        """
        target_file = QFileDialog.getOpenFileName(self,
                                                  'open json file', './', 'json file (*.json *.JSON);;All file (*.*)')
        if not target_file[0].endswith('.json') and not target_file[0].endswith('.JSON'):
            return
        with open(target_file[0], encoding='utf8') as fp:
            target_data = json.load(fp)  # dict
            if target_data['gamma'] is None or target_data['patch rgb target val'] is None:
                return
            if target_data['patch lab target val'] is None:
                return
            if target_data['lab patch weight'] is None:
                return
            if not isinstance(target_data['gamma'], list) or len(target_data['gamma']) != 256:
                return
            if not isinstance(target_data['patch rgb target val'], list) or len(
                    target_data['patch rgb target val']) != 18:
                return
            if not isinstance(target_data['patch lab target val'], list) or len(
                    target_data['patch lab target val']) != 18:
                return
            if not isinstance(target_data['patch lab target val'], list) or len(
                    target_data['patch lab target val']) != 18:
                return
            if not isinstance(target_data['lab patch weight'], list) or len(
                    target_data['lab patch weight']) != 18:
                return
            if not isinstance(target_data["lab sat coeff"], float):
                return
            self.gamma_curve = np.array(target_data['gamma'], np.uint8)
            assert self.gamma_curve.ndim == 1 and self.gamma_curve.shape[0] == 256
            self.gamma_en = target_data['gamma_en']
            target_patch_val = np.array(target_data['patch rgb target val'], np.uint8)
            if self.gamma_en:
                for n in range(18):
                    for j in range(3):
                        for i in range(255):
                            if self.gamma_curve[i] <= target_patch_val[n][j] < self.gamma_curve[i + 1]:
                                target_patch_val[n][j] = i
                                break
                        if target_patch_val[n][j] == self.gamma_curve[255]:
                            target_patch_val[n][j] = 255
            self.linear_rgb_patch_target = target_patch_val
            self.linear_lab_patch_target = np.array(target_data['patch lab target val'], np.float32)
            sat_coeff: float = target_data["lab sat coeff"]
            self.linear_lab_patch_target[:, 1] = self.linear_lab_patch_target[:, 1] * sat_coeff
            self.linear_lab_patch_target[:, 2] = self.linear_lab_patch_target[:, 2] * sat_coeff
            self.lab_patch_weight = np.array(target_data['lab patch weight'], np.float32)
