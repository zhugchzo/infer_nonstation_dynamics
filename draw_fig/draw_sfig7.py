from pypdf import PdfReader, PdfWriter
from pypdf._page import PageObject

# 读取两个 PDF
doc1 = PdfReader("../figures/SFIG7.1.pdf")
doc2 = PdfReader("../figures/SFIG7.2.pdf")

page1 = doc1.pages[0]
page2 = doc2.pages[0]

# 获取页面宽高
w1, h1 = page1.mediabox.width, page1.mediabox.height
w2, h2 = page2.mediabox.width, page2.mediabox.height

# 总宽度 = 最大宽度；总高度 = h1（上）+ h2（下）
width = max(w1, w2)
height = h1 + h2

# 创建一个新的空白页面
new_page = PageObject.create_blank_page(width=width, height=height)

# 计算水平居中的偏移量
tx1 = (width - w1) / 2
tx2 = (width - w2) / 2

# 把第1页放到顶部
new_page.merge_translated_page(page1, tx=tx1, ty=h2)

# 把第2页放到底部（水平居中）
new_page.merge_translated_page(page2, tx=tx2, ty=0)

# 输出 PDF
writer = PdfWriter()
writer.add_page(new_page)

with open("../figures/SFIG7.pdf", "wb") as f:
    writer.write(f)




