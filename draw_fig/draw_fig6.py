import fitz  # PyMuPDF

# 读取两个PDF
doc1 = fitz.open("../figures/FIG6.1.pdf")
doc2 = fitz.open("../figures/FIG6.2.pdf")

page1 = doc1.load_page(0)  # 第1页
page2 = doc2.load_page(0)  # 第1页

# 获取页面宽高
w1, h1 = page1.rect.width, page1.rect.height
w2, h2 = page2.rect.width, page2.rect.height

# 总宽度 = 最大宽度；总高度 = h1（上）+ h2（下）
width = max(w1, w2)
height = h1 + h2

# 新 PDF 页
output = fitz.open()
new_page = output.new_page(width=width, height=height)

new_page.show_pdf_page(
    fitz.Rect(0, 0, w1, h1),
    doc1,
    0
)

new_page.show_pdf_page(
    fitz.Rect(0, h1, w2, h1 + h2),
    doc2,
    0
)

# 保存合并后的 PDF
output.save("../figures/FIG6.pdf")
output.close()



