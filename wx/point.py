from pynput.mouse import Controller
import time

# 暂停3秒，给你时间移动鼠标到目标位置
print("请在3秒内将鼠标移动到微信窗口的目标位置...")
time.sleep(10)

# 获取鼠标当前位置
mouse = Controller()
current_mouse_position = mouse.position

# 输出鼠标当前位置的坐标
print(f"当前鼠标坐标为: {current_mouse_position}")
