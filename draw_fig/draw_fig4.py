from pypdf import PdfReader, PdfWriter, Transformation

# 读取 PDF
bg = PdfReader('../figures/FIG4.0.pdf')
ol_a1 = PdfReader('../figures/FIG4.a1.pdf')
ol_b1 = PdfReader('../figures/FIG4.b1.pdf')
ol_c1 = PdfReader('../figures/FIG4.c1.pdf')

bg_page = bg.pages[0]
ol_a1_page = ol_a1.pages[0]
ol_b1_page = ol_b1.pages[0]
ol_c1_page = ol_c1.pages[0]

# 背景页大小
bg_w = float(bg_page.mediabox.width)
bg_h = float(bg_page.mediabox.height)

# 覆盖页大小
ol_a1_w = float(ol_a1_page.mediabox.width)
ol_a1_h = float(ol_a1_page.mediabox.height)

ol_b1_w = float(ol_b1_page.mediabox.width)
ol_b1_h = float(ol_b1_page.mediabox.height)

ol_c1_w = float(ol_c1_page.mediabox.width)
ol_c1_h = float(ol_c1_page.mediabox.height)

# 1) 固定缩小比例（比如 50%）
scale = 0.5

ol_a1_w_s = ol_a1_w * scale
ol_a1_h_s = ol_a1_h * scale

ol_b1_w_s = ol_b1_w * scale
ol_b1_h_s = ol_b1_h * scale

ol_c1_w_s = ol_c1_w * scale
ol_c1_h_s = ol_c1_h * scale

# a1 location
a1_rel_x, a1_rel_y = 0.01, 0.5   # 相对位置
a1_x = bg_w * a1_rel_x          # 背景宽度 * 相对比例
a1_y = bg_h * a1_rel_y          # 背景高度 * 相对比例

# 默认以 overlay 的左下角为锚点
a1_t = Transformation().scale(scale).translate(a1_x, a1_y)

# b1 location
b1_rel_x, b1_rel_y = 0.52, 0.5   # 相对位置
b1_x = bg_w * b1_rel_x          # 背景宽度 * 相对比例
b1_y = bg_h * b1_rel_y          # 背景高度 * 相对比例

# 默认以 overlay 的左下角为锚点
b1_t = Transformation().scale(scale).translate(b1_x, b1_y)

# c1 location
c1_rel_x, c1_rel_y = 0.01, 0.075   # 相对位置
c1_x = bg_w * c1_rel_x          # 背景宽度 * 相对比例
c1_y = bg_h * c1_rel_y          # 背景高度 * 相对比例

# 默认以 overlay 的左下角为锚点
c1_t = Transformation().scale(scale).translate(c1_x, c1_y)

# 3) 合并
bg_page.merge_transformed_page(ol_a1_page, a1_t)
bg_page.merge_transformed_page(ol_b1_page, b1_t)
bg_page.merge_transformed_page(ol_c1_page, c1_t)

# 保存
writer = PdfWriter()
writer.add_page(bg_page)
with open('../figures/FIG4.pdf', 'wb') as f:
    writer.write(f)


