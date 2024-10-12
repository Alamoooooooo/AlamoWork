import pyautogui
import pyperclip
import time
from openpyxl import Workbook

# 设置微信群成员列表区域的起始坐标和滚动步长
MEMBER_LIST_START = (731, 218)  # 替换为你鼠标移动到第一个成员头像处获取的坐标
SCROLL_STEP = -500  # 每次滚动的像素值
SCROLL_PAUSE_TIME = 1.5  # 每次滚动后的暂停时间
MAX_MEMBERS = 100  # 设置一个最大成员数，视情况调整
DETAILS_POSITION = (744, 185)  # 群成员详细信息的初始位置，设置为屏幕上的某个空白区域
BACK_BUTTON_POSITION = (744, 185)  # 返回群成员列表的按钮坐标，视窗口布局调整

# 创建 Excel 工作簿
wb = Workbook()
ws = wb.active
ws.title = "微信群成员"
ws.append(["群昵称", "微信号"])  # Excel 表头


def get_member_info():
    """获取当前可见成员的群昵称"""
    pyautogui.click(MEMBER_LIST_START)  # 点击成员头像区域
    time.sleep(0.5)
    pyautogui.hotkey("ctrl", "c")  # 复制成员信息
    time.sleep(0.5)
    member_info = pyperclip.paste()  # 从剪贴板获取信息
    return member_info


def get_wechat_id():
    """获取当前群成员的微信号"""
    time.sleep(1)  # 等待群成员信息页面加载
    pyautogui.click(DETAILS_POSITION)  # 点击进入详情区域，确保详细信息可见
    time.sleep(0.5)

    # 模拟复制微信号操作
    pyautogui.hotkey("ctrl", "c")
    time.sleep(0.5)
    wechat_id = pyperclip.paste()
    return wechat_id


def scroll_down_member_list():
    """向下滚动群成员列表"""
    pyautogui.moveTo(MEMBER_LIST_START)
    pyautogui.scroll(SCROLL_STEP)


def go_back_to_member_list():
    """返回群成员列表"""
    pyautogui.click(BACK_BUTTON_POSITION)
    time.sleep(1.5)  # 等待群成员列表页面加载


def main():
    member_set = set()  # 用于存储已获取的成员，防止重复
    for _ in range(MAX_MEMBERS):
        info = get_member_info()  # 获取当前成员信息
        if info not in member_set and info.strip():
            member_set.add(info)

            # 点击头像进入个人资料
            pyautogui.click(MEMBER_LIST_START)  # 点击头像进入个人信息页
            time.sleep(1.5)  # 等待个人资料页加载

            # 获取微信号
            wechat_id = get_wechat_id()

            # 返回群成员列表
            go_back_to_member_list()

            # 将群昵称和微信号写入 Excel
            ws.append([info, wechat_id])

        # 滚动到下一个成员
        scroll_down_member_list()
        time.sleep(SCROLL_PAUSE_TIME)

    # 保存到 Excel 文件
    wb.save("wechat_group_members_with_ids.xlsx")
    print("群成员信息已保存到 wechat_group_members_with_ids.xlsx")


if __name__ == "__main__":
    main()
