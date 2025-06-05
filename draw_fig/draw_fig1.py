import fitz  # PyMuPDF

# 打开所有 PDF 文件
doc1 = fitz.open("../figures/FIG1.1.pdf")
doc2 = fitz.open("../figures/FIG1.2.pdf")
doc3 = fitz.open("../figures/FIG1.3.pdf")
doc4 = fitz.open("../figures/FIG-LINE.pdf")

# 获取页面尺寸
pages = [
    (doc1, 0),
    (doc4, 0),
    (doc2, 0),
    (doc4, 0),
    (doc3, 0),
]

# 计算总高度和最大宽度
total_height = 0
max_width = 0
sizes = []

for doc, pno in pages:
    page = doc.load_page(pno)
    w, h = page.rect.width, page.rect.height
    sizes.append((w, h))
    total_height += h
    max_width = max(max_width, w)

# 创建新页面
output = fitz.open()
new_page = output.new_page(width=max_width, height=total_height)

# 依次插入每页，纵向拼接，居中对齐
y_offset = 0
for (doc, pno), (w, h) in zip(pages, sizes):
    x_offset = (max_width - w) / 2
    rect = fitz.Rect(x_offset, y_offset, x_offset + w, y_offset + h)
    new_page.show_pdf_page(rect=rect, src=doc, pno=pno)
    y_offset += h

# 保存输出文件
output.save("../figures/FIG1.pdf")
output.close()




