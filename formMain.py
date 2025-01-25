from ui_cc_calib import Ui_eCCM
from ui_rawinfo import Ui_rawInfoDlg
from PySide6.QtWidgets import QWidget, QApplication, QMessageBox, QFileDialog, QDialog
from PySide6.QtGui import QFont, QIcon
from PySide6.QtCore import QFileInfo, QSettings, Signal, Slot
import sys
import numpy as np
import cv2 as cv
import json


def rgb_to_lab(input_rgb: np.ndarray) -> np.ndarray | None:
    """
    convert 8bit linear rgb to lab
    http://www.brucelindbloom.com/index.html?ColorCalculator.html
    :param input_rgb: 输入rgb， 18个patch
    :return: 输出18个patch的 lab
    """
    if input_rgb.shape != (18, 3):
        print('input rgb shape != (18, 3)')
        return None
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
    out_X = (out_X / 95.047) ** (1 / 3)  # BT709 d65 white point x=0.3127, y=0.3290, z=0.3583
    out_Y = (out_Y / 100.000) ** (1 / 3)
    out_Z = (out_Z / 108.883) ** (1 / 3)
    lab_arr[:, 0] = (116 * out_Y) - 16
    lab_arr[:, 1] = 500 * (out_X - out_Y)
    lab_arr[:, 2] = 200 * (out_Y - out_Z)
    return lab_arr  # L [0, 100], ab: [-128, +128]


def make_ccm_from_chromosome(chromosome: np.ndarray) -> np.ndarray:
    """
    从染色体构建ccm矩阵， 一条染色体有6个数值，各自独立
    :param chromosome: 输入一条染色体
    :return: 输出对应的ccm
    """
    res = np.array([
        [1.0 - chromosome[0] - chromosome[1], chromosome[0], chromosome[1]],
        [chromosome[2], 1.0 - chromosome[2] - chromosome[3], chromosome[3]],
        [chromosome[4], chromosome[5], 1.0 - chromosome[4] - chromosome[5]]
    ], np.float64)
    return res


def calc_lab_distance(target_patch: np.ndarray, current_patch: np.ndarray, patch_weight: np.ndarray,
                      just_ab=True) -> float:
    """
    计算两个lab值的距离，分18个patch，可以配置不同的权重
    :param target_patch: 理想的lab值
    :param current_patch: 当前的lab值
    :param patch_weight: 18 patch的权重
    :param just_ab: 仅仅计算ab之间的距离，不考虑L
    :return: 输出lab距离
    """
    if target_patch.shape != (18, 3) or current_patch.shape != (18, 3) or patch_weight.shape != (18,):
        print('lab patch != (18, 3)')
        return np.inf
    dist = 0.0
    if just_ab:
        patch_dist = np.sqrt(
            (target_patch[:, 1] - current_patch[:, 1]) ** 2 + (target_patch[:, 2] - current_patch[:, 2]) ** 2)
        dist = np.sum(patch_dist * patch_weight)  # element-wise multiply
    else:
        patch_dist = np.sqrt((target_patch[:, 0] - current_patch[:, 0]) ** 2 + (target_patch[:, 1] - current_patch[:, 1]) ** 2 + (target_patch[:, 2] - current_patch[:, 2]) ** 2)
        dist = np.sum(patch_dist * patch_weight)  # element-wise multiply
    return dist


class RawInfoDlg(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.ui = Ui_rawInfoDlg()
        self.ui.setupUi(self)


class ECcmFrom(QWidget):
    inform_wb_gain = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.ui = Ui_eCCM()
        self.ui.setupUi(self)
        self.rawData: np.ndarray = np.array([0, 0], np.uint16)
        self.bgrData: np.ndarray = np.array([0, 0], np.uint16)
        self.rawBayer: str = 'RGGB'
        self.rawBits: int = 12
        self.blc: int = 168
        self.linearRGBPatchTarget: None | np.ndarray = None  # np.zeros((18,), np.uint8)
        self.linearLABPatchTarget: None | np.ndarray = None  # np.zeros((18,), np.float64)
        self.labPatchWeight: None | np.ndarray = None
        self.inform_wb_gain.connect(self.ui.rawPreviewWidget.doWb)
        self.setWindowIcon(QIcon('./img/icon.png'))

    def import_raw_file(self) -> None:
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
        self.rawBits = int(raw_info_dlg.ui.rawBitComboBox.currentText())
        self.rawBayer = raw_info_dlg.ui.bayerComboBox.currentText()
        self.blc = int(raw_info_dlg.ui.blcLineEdit.text())
        raw_width = int(raw_info_dlg.ui.rawWidthLine.text())
        raw_height = int(raw_info_dlg.ui.rawHeightLine.text())

        settings.setValue("rawBits", self.rawBits)
        settings.setValue("rawWidth", raw_width)
        settings.setValue("rawHeight", raw_height)
        settings.setValue("rawBayer", self.rawBayer)
        settings.setValue("blc", self.blc)

        raw_size = QFileInfo(in_raw[0]).size()
        if raw_size != raw_width * raw_height * (2 if self.rawBits > 8 else 1):
            QMessageBox.critical(self, "error", "raw file size != your input", QMessageBox.StandardButton.Ok,
                                 QMessageBox.StandardButton.Cancel)

        self.rawData = np.fromfile(in_raw[0], (np.uint16 if self.rawBits > 8 else np.uint8)).reshape(
            (raw_height, raw_width)) - self.blc
        self.show_bgr_on_preview()

    def show_bgr_on_preview(self) -> None:
        raw_data_b = (self.rawData >> (self.rawBits - 8)).astype(np.uint8)
        if self.rawBayer == 'RGGB':
            self.bgrData = cv.cvtColor(raw_data_b, cv.COLOR_BayerRGGB2BGR)
        elif self.rawBayer == 'GRBG':
            self.bgrData = cv.cvtColor(raw_data_b, cv.COLOR_BayerGRBG2BGR)
        elif self.rawBayer == 'GBRG':
            self.bgrData = cv.cvtColor(raw_data_b, cv.COLOR_BayerGBRG2BGR)
        elif self.rawBayer == 'BGGR':
            self.bgrData = cv.cvtColor(raw_data_b, cv.COLOR_BayerBGGR2BGR)
        self.ui.rawPreviewWidget.bgrData = self.bgrData
        self.ui.rawPreviewWidget.bgrDataPreview = self.bgrData.copy()
        self.ui.rawPreviewWidget.hadDoWb = False
        self.ui.rawPreviewWidget.update()

    @Slot()
    def show_preview_bright(self, val: int):
        self.ui.previewBrightLabel.setText(f"preview brightness: {val / 10.0: .2f}")

    def calc_cc_matrix_traditional(self, bgr_mean_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        res = np.array([[1024, 0, 0], [0, 1024, 0], [0, 0, 1024]], np.int32)
        res_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.float64)
        if self.linearRGBPatchTarget is None or bgr_mean_array is None:
            print('error, no linear rgb target value')
            return res, res_init
        rgb_mean_array = bgr_mean_array[:3, :, ::-1].reshape((18, 3))

        b = self.linearRGBPatchTarget[:, 0] - rgb_mean_array[:, 0]
        A = np.zeros((18, 2), dtype=np.float64)
        A[:, 0] = rgb_mean_array[:, 1] - rgb_mean_array[:, 0]
        A[:, 1] = rgb_mean_array[:, 2] - rgb_mean_array[:, 0]
        At = np.transpose(A)
        line0 = ((np.linalg.inv(At.dot(A))).dot(At).dot(b) * 1024).astype(np.int32)
        res[0, 0] = 1024 - line0[0] - line0[1]
        res[0, 1] = line0[0]
        res[0, 2] = line0[1]

        b = self.linearRGBPatchTarget[:, 1] - rgb_mean_array[:, 1]
        A = np.zeros((18, 2), dtype=np.float64)
        A[:, 0] = rgb_mean_array[:, 0] - rgb_mean_array[:, 1]
        A[:, 1] = rgb_mean_array[:, 2] - rgb_mean_array[:, 1]
        At = np.transpose(A)
        line1 = ((np.linalg.inv(At.dot(A))).dot(At).dot(b) * 1024).astype(np.int32)
        res[1, 0] = line1[0]
        res[1, 1] = 1024 - line1[0] - line1[1]
        res[1, 2] = line1[1]

        b = self.linearRGBPatchTarget[:, 2] - rgb_mean_array[:, 2]
        A = np.zeros((18, 2), dtype=np.float64)
        A[:, 0] = rgb_mean_array[:, 0] - rgb_mean_array[:, 2]
        A[:, 1] = rgb_mean_array[:, 1] - rgb_mean_array[:, 2]
        At = np.transpose(A)
        line2 = ((np.linalg.inv(At.dot(A))).dot(At).dot(b) * 1024).astype(np.int32)
        res[2, 0] = line2[0]
        res[2, 1] = line2[1]
        res[2, 2] = 1024 - line0[0] - line0[1]
        return res, np.array([line0[0], line0[1], line1[0], line1[1], line2[0], line2[1]], np.float64) / 1024.0

    def calc_cc_matrix_lab_space(self, bgr_mean_array: np.ndarray, init_arr: np.ndarray) -> np.ndarray:
        """
        使用遗传算法迭代优化ccm
        :param init_arr: 初始化cc参数
        :param bgr_mean_array: 输入色卡bgr均值
        :return: ccm结果
        """
        if self.linearLABPatchTarget is None or bgr_mean_array is None:
            print('error, no linear lab target value')
            return np.array([[1024, 0, 0], [0, 1024, 0], [0, 0, 1024]], np.int32)
        rgb_18_patch = bgr_mean_array[:3, :, ::-1].reshape((18, 3))  # 18 个patch rgb
        rgb_18_patch = rgb_18_patch.transpose()  # ccm * rgb^t => shape(3, 18)
        origin_ccm_arr = 6.0 * (np.random.rand(7, 6) - 0.5)
        origin_ccm_arr[0] = init_arr

        epoch = 10000
        while epoch > 0:
            ccm_list: list[np.ndarray] = [make_ccm_from_chromosome(origin_ccm_arr[0]),
                                          make_ccm_from_chromosome(origin_ccm_arr[1]),
                                          make_ccm_from_chromosome(origin_ccm_arr[2]),
                                          make_ccm_from_chromosome(origin_ccm_arr[3]),
                                          make_ccm_from_chromosome(origin_ccm_arr[4]),
                                          make_ccm_from_chromosome(origin_ccm_arr[5]),
                                          make_ccm_from_chromosome(origin_ccm_arr[5])]
            lab_distance = np.zeros((7, ), np.float64)
            for idx, ccm in enumerate(ccm_list):
                ccm_18_rgb: np.ndarray = np.matmul(ccm, rgb_18_patch)  # (3, 3) * (3, 18) => (3, 18)
                ccm_18_rgb = ccm_18_rgb.transpose()  # (18, 3)
                ccm_18_lab = rgb_to_lab(ccm_18_rgb)
                if ccm_18_lab is None:
                    print('rgb2lab error')
                    break
                lab_distance[idx] = calc_lab_distance(self.linearLABPatchTarget, ccm_18_lab, self.labPatchWeight)
            sorted_idx = np.argsort(lab_distance)  # distance从小到大的索引
            print(f'epoch={epoch}, best lab distance={lab_distance[sorted_idx[0]]}')
            # 更新origin_ccm_arr, 距离最小的两组不变，剩下5组丢弃，然后前两组分别组合，变异，得到新的5组
            origin_ccm_arr[0], origin_ccm_arr[1] = origin_ccm_arr[sorted_idx[0]], origin_ccm_arr[sorted_idx[1]]
            origin_ccm_arr[2] = (origin_ccm_arr[0] + origin_ccm_arr[1]) / 2.0
            origin_ccm_arr[3, 0::2] = origin_ccm_arr[0, 0::2]
            origin_ccm_arr[3, 1::2] = origin_ccm_arr[1, 1::2]
            origin_ccm_arr[4, 0::2] = origin_ccm_arr[0, 1::2]
            origin_ccm_arr[4, 1::2] = origin_ccm_arr[1, 0::2]
            # 剩余两组ccm参数采用复制，然后随机变异变6个值中的两个
            origin_ccm_arr[5], origin_ccm_arr[6] = origin_ccm_arr[0], origin_ccm_arr[1]
            variation_pos = np.random.randint(low=0, high=6, size=(4, ))
            variation_val = 5.0 * (np.random.rand(4) - 0.5)
            origin_ccm_arr[5, variation_pos[0]] = variation_val[0]
            origin_ccm_arr[5, variation_pos[1]] = variation_val[1]
            origin_ccm_arr[6, variation_pos[2]] = variation_val[2]
            origin_ccm_arr[6, variation_pos[3]] = variation_val[3]
            epoch -= 1
        # 选择最优的结果
        res = make_ccm_from_chromosome(origin_ccm_arr[0])
        res = (res * 1024).astype(np.int32)
        return res

    @Slot()
    def calculate_ccm(self):
        if self.bgrData is None:
            return
        if self.ui.rawPreviewWidget.startPt == self.ui.rawPreviewWidget.endPt:
            return
        patch_size = self.ui.rawPreviewWidget.patchSize  # float
        start_pt_x = self.bgrData.shape[1] * self.ui.rawPreviewWidget.startXPercent
        start_pt_y = self.bgrData.shape[0] * self.ui.rawPreviewWidget.startYPercent
        end_pt_x = self.bgrData.shape[1] * self.ui.rawPreviewWidget.endXPercent
        end_pt_y = self.bgrData.shape[0] * self.ui.rawPreviewWidget.endYPercent
        x_step = (end_pt_x - start_pt_x) / 6
        y_step = (end_pt_y - start_pt_y) / 4
        gap = (1.0 - patch_size) / 2
        bgr_mean_array = np.zeros((4, 6, 3), dtype=np.float64)
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
                r, gr, gb, b = self.rawData[p0_y:p1_y:2, p0_x:p1_x:2], self.rawData[p0_y:p1_y:2, p0_x + 1:p1_x:2], \
                    self.rawData[p0_y + 1:p1_y:2, p0_x:p1_x:2], self.rawData[p0_y + 1:p1_y:2, p0_x + 1:p1_x:2]
                if self.rawBayer == 'RGGB':
                    pass
                elif self.rawBayer == 'GRBG':
                    gr, r, b, gb = r, gr, gb, b
                elif self.rawBayer == 'GBRG':
                    gb, b, r, gr = r, gr, gb, b
                elif self.rawBayer == 'BGGR':
                    b, gb, gr, r = r, gr, gb, b
                r_mean = np.mean(r)
                g_mean = (np.mean(gr) + np.mean(gb)) / 2
                b_mean = np.mean(b)
                bgr_mean_array[row - 1, col - 1, 2] = r_mean
                bgr_mean_array[row - 1, col - 1, 1] = g_mean
                bgr_mean_array[row - 1, col - 1, 0] = b_mean
        # print(bgr_mean_array)
        b_gain0 = bgr_mean_array[3, 1, 1] / bgr_mean_array[3, 1, 0]
        r_gain0 = bgr_mean_array[3, 1, 1] / bgr_mean_array[3, 1, 2]
        b_gain1 = bgr_mean_array[3, 2, 1] / bgr_mean_array[3, 2, 0]
        r_gain1 = bgr_mean_array[3, 2, 1] / bgr_mean_array[3, 2, 2]
        b_gain2 = bgr_mean_array[3, 3, 1] / bgr_mean_array[3, 3, 0]
        r_gain2 = bgr_mean_array[3, 3, 1] / bgr_mean_array[3, 3, 2]
        b_gain3 = bgr_mean_array[3, 4, 1] / bgr_mean_array[3, 4, 0]
        r_gain3 = bgr_mean_array[3, 4, 1] / bgr_mean_array[3, 4, 2]
        b_gain = (b_gain0 + b_gain1 + b_gain2 + b_gain3) / 4.0
        r_gain = (r_gain0 + r_gain1 + r_gain2 + r_gain3) / 4.0
        self.inform_wb_gain.emit(b_gain, r_gain)

        QApplication.instance().processEvents()

        rgb_scl = 16
        bgr_mean_array[:, :, 0] = (bgr_mean_array[:, :, 0] * b_gain) / rgb_scl
        bgr_mean_array[:, :, 1] = bgr_mean_array[:, :, 1] / rgb_scl
        bgr_mean_array[:, :, 2] = (bgr_mean_array[:, :, 2] * r_gain) / rgb_scl

        print('请注意这里的rgb_scl, 改成和你的raw曝光相符的系数')
        print('目标色卡第四行的灰度值(8bit)依次为 245, 200, 161, 121, 82, 49')
        print('反gamma后的值(8bit)为 232, 148, 91, 48, 21, 7')
        print('你的raw色卡第四行值为：')
        print(bgr_mean_array[3, :, :])

        ccm0, init_cc = self.calc_cc_matrix_traditional(bgr_mean_array)  # 8bit
        print('使用rgb线性转换到rgb，最小二乘法的结果：')
        print(ccm0)
        print('初始cc参数', init_cc)
        ccm1 = self.calc_cc_matrix_lab_space(bgr_mean_array, init_cc)
        print('使用rgb转换到lab，迭代优化的结果：')
        print(ccm1)

    @Slot()
    def import_target_file(self):
        target_file = QFileDialog.getOpenFileName(self,
                                                  'open json file', './', 'json file (*.json *.JSON);;All file (*.*)')
        if not target_file[0].endswith('.json') and not target_file[0].endswith('.JSON'):
            return
        with open(target_file[0]) as fp:
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
            gamma_in = np.array(target_data['gamma'], np.uint8)
            target_patch_val = np.array(target_data['patch rgb target val'], np.uint8)
            for n in range(18):
                for j in range(3):
                    for i in range(255):
                        if gamma_in[i] <= target_patch_val[n][j] < gamma_in[i + 1]:
                            target_patch_val[n][j] = i
                            break
                    if target_patch_val[n][j] == gamma_in[255]:
                        target_patch_val[n][j] = 255
            self.linearRGBPatchTarget = target_patch_val
            self.linearLABPatchTarget = np.array(target_data['patch lab target val'], np.float64)
            self.labPatchWeight = np.array(target_data['lab patch weight'], np.float64)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei UI", 10))
    QApplication.setApplicationName("eCCM")
    QApplication.setOrganizationName("vISP")
    QApplication.setOrganizationDomain("vISP.dev")
    window = ECcmFrom()
    window.show()
    sys.exit(app.exec())
