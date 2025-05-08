import sys
sys.stdout.reconfigure(encoding='utf-8')

import random
import datetime
import os
from PyQt5.QtWidgets import QApplication, QLabel, QMenu, QAction
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import Qt, QTimer, QPoint, QSize
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtWidgets import QWidget, QListWidget, QPushButton, QVBoxLayout
from PyQt5.QtGui import QPixmap
import shutil
from pygame import mixer

class DesktopPet(QLabel):
    def __init__(self):
        super().__init__()
        self.is_fixed = False  # 是否固定
        self.skins = self.load_skins()  # 加载皮肤
        self.current_skin = "default"  # 初始化默认皮肤
        self.dx, self.dy = 1, 1  # 每次移动的速度（像素）
        self.init_ui()  # 初始化界面

        # 定时器，用于更新位置
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_position)
        self.timer.start(16)  # 每16毫秒更新一次，相当于60帧/秒
        self.setMouseTracking(True)  # 开启鼠标追踪
        
        self.target_pos = self.get_random_target()  # 目标点
        self.move_progress = 0  # 移动进度 (0 到 1)
        self.wait_time = 0  # 停留时间
        self.moving = True  # 是否正在移动


    def init_ui(self):
        # 设置窗口标志和透明背景
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.SubWindow)
        self.setAttribute(Qt.WA_TranslucentBackground, True)  # 确保背景透明

        # 初始化声音
        mixer.init()
        self.click_sound = "click.mp3"

        # 加载GIF并调整大小
        self.movie = QMovie(self.get_skin_path(self.current_skin))
        # self.movie.setScaledSize(QSize(128, 128))  # 设置统一大小
        self.movie.frameChanged.connect(self.adjust_size)  # 每次帧变化时调整窗口大小
        self.setMovie(self.movie)
        self.movie.start()

        # 右键菜单
        self.menu = QMenu(self)
        self.create_menu()

        # 检查节日
        self.check_festival()

    def adjust_size(self):
        """ 调整窗口大小，使其匹配 GIF """
        if self.movie is None:
            return

        size = self.movie.currentImage().size()
        if size.isValid():
            self.resize(size)  # 适配 GIF 大小

    def update_position(self):
        if not self.is_fixed:
            if self.moving:
                self.move_progress += 0.0003  # 控制移动速度（0.02 可调整）
                if self.move_progress >= 1:
                    self.move_progress = 1
                    self.moving = False  # 移动结束，开始停留
                    self.wait_time = random.randint(1000, 30000)  # 随机停留 1-30 秒
                    QTimer.singleShot(self.wait_time, self.start_moving)  # 计时后继续移动

                # 贝塞尔缓动曲线 (t=progress, x = (1-t)*start_x + t*target_x)
                ease_t = self.ease_in_out(self.move_progress)  # 计算缓动
                new_x = int((1 - ease_t) * self.x() + ease_t * self.target_pos.x())
                new_y = int((1 - ease_t) * self.y() + ease_t * self.target_pos.y())
                self.move(new_x, new_y)

    def start_moving(self):
        """ 选择新的目标点并开始移动 """
        self.target_pos = self.get_random_target()
        self.move_progress = 0  # 重置进度
        self.moving = True
    
    def get_random_target(self):
        """ 在屏幕范围内随机选择一个目标点 """
        screen = QApplication.primaryScreen().geometry()
        target_x = random.randint(50, screen.width() - 150)
        target_y = random.randint(50, screen.height() - 150)
        return QPoint(target_x, target_y)

    def ease_in_out(self, t):
        """ 改进的贝塞尔缓动函数，使移动更平滑 """
        return t ** 3 * (t * (t * 6 - 15) + 10)
    
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

        open_file_action = QAction("管理皮肤", self)
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
        """ 打开皮肤管理窗口 """
        self.skin_manager = SkinManager(self)
        self.skin_manager.show()

    def open_link(self, url):
        import webbrowser
        webbrowser.open(url)

    def change_skin(self, skin_name):
        """ 切换皮肤，保持比例缩放 """
        self.current_skin = skin_name
        self.movie.stop()
        self.movie.setFileName(self.get_skin_path(skin_name))
        self.movie.start()

        # 延迟调整大小，确保 GIF 已加载
        QTimer.singleShot(100, self.adjust_size)  
        
    def update_movie_size(self):
        """ 在 GIF 载入后，动态调整大小 """
        original_size = self.movie.currentImage().size()

        if original_size.isValid() and original_size.width() > 0 and original_size.height() > 0:
            width, height = original_size.width(), original_size.height()
            
            if width > height:  # 宽图，固定宽度 128
                new_width = 128
                new_height = int((height / width) * 128)
            else:  # 高图，固定高度 128
                new_height = 128
                new_width = int((width / height) * 128)

            self.movie.setScaledSize(QSize(new_width, new_height))
            self.resize(new_width, new_height)  # 调整窗口大小


    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.menu.exec_(event.globalPos())  # 右键打开菜单
        elif event.button() == Qt.LeftButton:
            self.old_pos = event.globalPos()
            self.is_dragging = True  # 标记正在拖动
            self.moving = False  # 停止自动移动

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.is_dragging:
            delta = event.globalPos() - self.old_pos
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.old_pos = event.globalPos()
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_dragging:
            self.is_dragging = False  # 结束拖动
            self.moving = False  # 停止当前运动
            QTimer.singleShot(3000, self.start_moving)  # 5 秒后重新开始

    def check_festival(self):
        today = datetime.datetime.now().date()
        if today == datetime.date(today.year, 1, 1):
            self.change_skin("new_year")
        elif today == datetime.date(today.year, 2, 10):
            self.change_skin("spring_festival")


class SkinManager(QWidget):
    """ 皮肤管理窗口 """

    def __init__(self, parent):
        super().__init__()
        self.parent = parent  # 关联主窗口
        self.setWindowTitle("皮肤管理")
        self.setGeometry(100, 100, 300, 400)

        # 列表组件
        self.skin_list = QListWidget()
        self.skin_list.addItems(self.parent.skins)
        self.skin_list.setCurrentRow(0)
        self.skin_list.itemClicked.connect(self.update_preview)

        # 皮肤预览
        self.preview_label = QLabel("预览：")
        self.preview_image = QLabel()
        self.preview_image.setFixedSize(128, 128)

        # 按钮
        self.add_button = QPushButton("添加皮肤")
        self.add_button.clicked.connect(self.add_skin)

        self.delete_button = QPushButton("删除皮肤")
        self.delete_button.clicked.connect(self.delete_skin)

        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self.skin_list)
        layout.addWidget(self.preview_label)
        layout.addWidget(self.preview_image)
        layout.addWidget(self.add_button)
        layout.addWidget(self.delete_button)
        self.setLayout(layout)

        self.update_preview()  # 初始化预览

    def update_preview(self):
        """ 更新预览图 """
        selected_skin = self.skin_list.currentItem().text()
        skin_path = self.parent.get_skin_path(selected_skin)
        self.preview_image.setPixmap(QPixmap(skin_path).scaled(128, 128))

    def add_skin(self):
        """ 让用户选择一个 GIF 作为新皮肤 """
        file_path, _ = QFileDialog.getOpenFileName(self, "选择GIF皮肤", "", "GIF 文件 (*.gif)")
        if file_path:
            skin_name, _ = os.path.splitext(os.path.basename(file_path))  # 获取文件名
            dest_path = os.path.join(os.path.dirname(__file__), "skins", f"{skin_name}.gif")

            if os.path.exists(dest_path):
                QMessageBox.warning(self, "皮肤已存在", f"皮肤 '{skin_name}' 已存在！请更换名称。")
                return
            
            shutil.copy(file_path, dest_path)  # 复制到 skins 目录
            self.parent.skins.append(skin_name)  # 更新皮肤列表
            self.skin_list.addItem(skin_name)  # 更新列表
            self.parent.create_menu()  # 更新菜单
            QMessageBox.information(self, "添加成功", f"皮肤 '{skin_name}' 添加成功！")

    def delete_skin(self):
        """ 删除选定的皮肤 """
        selected_skin = self.skin_list.currentItem().text()
        if selected_skin == "default":
            QMessageBox.warning(self, "无法删除", "默认皮肤不可删除！")
            return

        reply = QMessageBox.question(self, "删除确认", f"确定要删除皮肤 '{selected_skin}' 吗？", 
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            skin_path = self.parent.get_skin_path(selected_skin)
            if os.path.exists(skin_path):
                os.remove(skin_path)  # 删除文件
                self.parent.skins.remove(selected_skin)  # 更新主窗口的皮肤列表
                self.skin_list.takeItem(self.skin_list.currentRow())  # 从列表移除
                self.parent.create_menu()  # 更新菜单
                QMessageBox.information(self, "删除成功", f"皮肤 '{selected_skin}' 已删除！")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    pet = DesktopPet()
    pet.show()
    sys.exit(app.exec_())