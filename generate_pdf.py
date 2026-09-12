"""Generate hostel_handbook.pdf from hostel_handbook.md using fpdf2."""
from pathlib import Path
from fpdf import FPDF


def md_to_pdf(md_path: Path, pdf_path: Path):
    content = md_path.read_text(encoding='utf-8')

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.add_page()

    lines = content.split('\n')
    in_table = False

    for line in lines:
        line = line.rstrip()

        # Skip separator lines in tables
        if line.strip().startswith('|') and all(
            c in '|-: ' for c in line.replace('|', '')
        ):
            continue

        if line.startswith('# '):
            in_table = False
            pdf.set_font('Helvetica', 'B', 16)
            text = line[2:].strip()
            text = text.encode('latin-1', errors='replace').decode('latin-1')
            pdf.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)

        elif line.startswith('## '):
            in_table = False
            pdf.set_font('Helvetica', 'B', 13)
            pdf.ln(5)
            text = line[3:].strip()
            text = text.encode('latin-1', errors='replace').decode('latin-1')
            pdf.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

        elif line.startswith('### '):
            in_table = False
            pdf.set_font('Helvetica', 'B', 11)
            pdf.ln(3)
            text = line[4:].strip()
            text = text.encode('latin-1', errors='replace').decode('latin-1')
            pdf.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)

        elif line.startswith('---'):
            in_table = False
            continue

        elif line.strip() == '':
            in_table = False
            pdf.ln(3)

        elif line.startswith('|'):
            # Table row
            in_table = True
            pdf.set_font('Helvetica', '', 7)
            cells = [c.strip() for c in line.split('|')]
            # Remove empty first/last from split
            cells = [c for c in cells if c]
            if not cells:
                continue

            usable = pdf.w - pdf.l_margin - pdf.r_margin
            col_w = usable / len(cells)

            for cell_text in cells:
                cell_text = cell_text.replace('**', '').replace('*', '')
                cell_text = cell_text.encode('latin-1', errors='replace').decode('latin-1')
                # Truncate to fit
                max_chars = max(int(col_w / 1.5), 5)
                pdf.cell(col_w, 5, cell_text[:max_chars], border=1)
            pdf.ln()

        else:
            in_table = False
            pdf.set_font('Helvetica', '', 10)
            clean = line.replace('**', '').replace('*', '')
            safe_text = clean.encode('latin-1', errors='replace').decode('latin-1')
            # Reset X to left margin before multi_cell
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin, 5, safe_text)

    pdf.output(str(pdf_path))
    print(f"Generated PDF: {pdf_path}")
    print(f"Pages: {pdf.pages_count}")


if __name__ == "__main__":
    corpus_dir = Path("corpus")
    md_path = corpus_dir / "hostel_handbook.md"
    pdf_path = corpus_dir / "hostel_handbook.pdf"

    if not md_path.exists():
        print(f"Source file {md_path} not found.")
    else:
        md_to_pdf(md_path, pdf_path)
        print("Done! The hostel_handbook.pdf has been generated.")
