# CCM 优化工具

运行测试结果：

![test](./img/screen-shot.png)

## 安装依赖

```bash
pip install pyside6 numpy opencv-python
```



## 使用说明

1. 首先点击`import raw`按钮，子目录`raw_img`内附带了一张16bit hdr raw， blc=168, bayer=GRBG, 1920*1080，供测试使用。
2. 点击`import target + gamma`，导入`target.json`当中的设置。
3. 拖动preivew brightness滑块，调整预览亮度。
4. 用鼠标在色卡上，从左上到右下角拉出一个矩形，拖动patch size滑块调整色块的roi大小。
5. 用`adjust ROI`里面的按钮微调边界。
6. 点击`calucate ccm`，观察结果，可以多试几次找到最优解。

