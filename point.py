# coding:utf-8
import cv2

# 回调函数，用于获取鼠标点击位置的坐标
def get_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"点击位置的坐标为：({x}, {y})")

# 读取图片
image = cv2.imread(r"F:\lubiao\testfile\006.png")  # 替换为你的图片路径

# 创建一个窗口
cv2.namedWindow("image")

# 设置鼠标回调函数
cv2.setMouseCallback("image", get_mouse_click)

# 显示图片
cv2.imshow("image", image)

# 等待按键操作，按下任意键退出
cv2.waitKey(0)

# 销毁窗口
cv2.destroyAllWindows()
