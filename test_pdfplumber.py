import pdfplumber

with pdfplumber.open("กฏหมายจราจร.pdf") as pdf:
    page1 = pdf.pages[0]
    text = page1.extract_text()
    print(text)