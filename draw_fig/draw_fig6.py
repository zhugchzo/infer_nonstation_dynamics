from pypdf import PdfReader, PdfWriter, Transformation

# 读取 PDF
bg = PdfReader('../figures/FIG6.2.pdf')
ol = PdfReader('../figures/FIG6.1.pdf')

bg_page = bg.pages[0]
ol_page = ol.pages[0]

# 背景页大小
bg_w = float(bg_page.mediabox.width)
bg_h = float(bg_page.mediabox.height)

# 覆盖页大小
ol_w = float(ol_page.mediabox.width)
ol_h = float(ol_page.mediabox.height)

# 1) 固定缩小比例（比如 50%）
scale = 0.62

ol_w_s = ol_w * scale
ol_h_s = ol_h * scale

# 2) 相对位置 -> 转换为绝对坐标
rel_x, rel_y = 0, 0.45   # 相对位置
x = bg_w * rel_x          # 背景宽度 * 相对比例
y = bg_h * rel_y          # 背景高度 * 相对比例

# 默认以 overlay 的左下角为锚点
t = Transformation().scale(scale).translate(x, y)

# 3) 合并
bg_page.merge_transformed_page(ol_page, t)

# 保存
writer = PdfWriter()
writer.add_page(bg_page)
with open('../figures/FIG6.pdf', 'wb') as f:
    writer.write(f)


