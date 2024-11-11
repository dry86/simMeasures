import matplotlib.font_manager as fm

# 获取所有支持的字体名称
font_names = sorted([f.name for f in fm.fontManager.ttflist])
for font_name in font_names:
    print(font_name)
