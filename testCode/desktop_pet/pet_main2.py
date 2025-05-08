import sys
sys.stdout.reconfigure(encoding='utf-8')

import random
import datetime
import os
from PyQt5.QtWidgets import QApplication, QLabel, QMenu, QAction
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import Qt, QTimer, QPoint
from pygame import mixer

class DesktopPet(QLabel):
    def __init__(self):
        super().__init__()
        self.is_fixed = False
        self.skins = self.load_skins()
        self.current_skin = "default"
        self.dx, self.dy = 2, 2
        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_position)
        self.timer.start(16)
        self.setMouseTracking(True)

    def init_ui(self):
        # 设置窗口标志和透明背景
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.SubWindow)
        self.setAttribute(Qt.WA_TranslucentBackground, True)  # 确保背景透明

        # 初始化声音
        mixer.init()
        self.click_sound = "click.mp3"

        # 加载GIF并连接尺寸调整信号
        self.movie = QMovie(self.get_skin_path(self.current_skin))
        self.movie.started.connect(self.initial_adjust_size)  # 连接信号
        self.setMovie(self.movie)
        self.movie.start()

        # 右键菜单
        self.menu = QMenu(self)
        self.create_menu()

        # 检查节日
        self.check_festival()

    def initial_adjust_size(self):
        """初始化时调整窗口大小到GIF尺寸"""
        size = self.movie.frameRect().size()
        if size.isValid():
            self.resize(size)
            # 移动到屏幕中央
            screen = QApplication.primaryScreen().geometry()
            x = (screen.width() - self.width()) // 2
            y = (screen.height() - self.height()) // 2
            self.move(x, y)
            print(f"初始位置：x={x}, y={y}")
        self.movie.started.disconnect(self.initial_adjust_size)  # 断开连接

    def update_position(self):
        if not self.is_fixed:  # 如果桌宠未被固定
            current_x = self.x()
            current_y = self.y()

            # 移动位置
            new_x = current_x + self.dx
            new_y = current_y + self.dy

            # 获取屏幕尺寸，防止超出边界
            screen = QApplication.primaryScreen().geometry()
            if new_x <= 0 or new_x + self.width() >= screen.width():
                self.dx = -self.dx  # 碰到左右边界，反向
            if new_y <= 0 or new_y + self.height() >= screen.height():
                self.dy = -self.dy  # 碰到上下边界，反向

            # 移动桌宠
            self.move(new_x, new_y)

    def create_menu(self):
        # 添加右键菜单选项
        fix_action = QAction("固定/取消固定", self)
        fix_action.triggered.connect(self.toggle_fixed)
        self.menu.addAction(fix_action)

        # 皮肤切换
        skin_menu = QMenu("切换皮肤", self)
        for skin in self.skins:
            skin_action = QAction(skin, self)
            skin_action.triggered.connect(lambda _, s=skin: self.change_skin(s))
            skin_menu.addAction(skin_action)
        self.menu.addMenu(skin_menu)

        open_file_action = QAction("打开文件", self)
        open_file_action.triggered.connect(self.open_file)
        self.menu.addAction(open_file_action)

        link_action = QAction("跳转链接", self)
        link_action.triggered.connect(lambda: self.open_link("https://www.example.com"))
        self.menu.addAction(link_action)

        quit_action = QAction("退出", self)
        quit_action.triggered.connect(self.close)
        self.menu.addAction(quit_action)

    
    def load_skins(self):
        skin_folder = os.path.join(os.path.dirname(__file__), "skins")  # 获取脚本所在目录的 skins 文件夹
        if not os.path.exists(skin_folder):
            os.makedirs(skin_folder)
        skins = [f.replace(".gif", "") for f in os.listdir(skin_folder) if f.endswith(".gif")]
        if "default" not in skins:
            skins.insert(0, "default")
        return skins

    def get_skin_path(self, skin_name):
        skin_folder = os.path.join(os.path.dirname(__file__), "skins")
        if skin_name == "default":
            return os.path.join(skin_folder, "pet.gif")  # 默认皮肤路径
        return os.path.join(skin_folder, f"{skin_name}.gif")


    def toggle_fixed(self):
        self.is_fixed = not self.is_fixed

    def open_file(self):
        os.system("notepad README.txt")

    def open_link(self, url):
        import webbrowser
        webbrowser.open(url)

    def change_skin(self, skin_name):
        self.current_skin = skin_name
        self.movie.stop()
        self.movie.setFileName(self.get_skin_path(skin_name))
        self.movie.started.connect(self.adjust_size_after_skin_change)  # 连接新信号
        self.movie.start()

    def adjust_size_after_skin_change(self):
        """切换皮肤后调整窗口大小"""
        size = self.movie.frameRect().size()
        if size.isValid():
            self.resize(size)
        self.movie.started.disconnect(self.adjust_size_after_skin_change)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.menu.exec_(event.globalPos())
        elif event.button() == Qt.LeftButton:
            self.old_pos = event.globalPos()
            if os.path.exists(self.click_sound):
                mixer.Sound(self.click_sound).play()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and not self.is_fixed:
            delta = QPoint(event.globalPos() - self.old_pos)
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.old_pos = event.globalPos()

    def check_festival(self):
        today = datetime.datetime.now().date()
        if today == datetime.date(today.year, 1, 1):
            self.change_skin("new_year")
        elif today == datetime.date(today.year, 2, 10):
            self.change_skin("spring_festival")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    pet = DesktopPet()
    pet.show()
    sys.exit(app.exec_())
