import holidays

import locale

# 设置区域为英文
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

# 获取英文节日
cn_holidays = holidays.China(years=2024)
print(cn_holidays.items())
