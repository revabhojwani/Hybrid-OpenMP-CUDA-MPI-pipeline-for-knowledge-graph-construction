import pdfplumber

with pdfplumber.open("/mnt/c/Users/reva/Internship/10k/apple.pdf") as pdf:
    print(f"Total pages: {len(pdf.pages)}")
    # Print first 2000 characters of page 1 to eyeball quality
    page = pdf.pages[0]
    text = page.extract_text()
    print(text[:2000] if text else "No text extracted")