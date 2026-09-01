from pathlib import Path
import pymupdf

from langchain_core.documents import Document


def load_pdfs(folder_path: str, max_pages_per_pdf: int = None):

    pdf_folder = Path(folder_path)
    pdf_files = sorted(pdf_folder.glob("*.pdf"))

    documents = []

    print(f"PDF files found: {len(pdf_files)}")

    for pdf_file in pdf_files:

        print(f"\nLoading: {pdf_file.name}")

        pdf = pymupdf.open(pdf_file)

        total_pages = len(pdf)

        print(f"Total pages: {total_pages}")

        pages_to_process = total_pages

        if max_pages_per_pdf is not None:
            pages_to_process = min(
                max_pages_per_pdf,
                total_pages
            )

        for page_index in range(pages_to_process):

            page = pdf[page_index]

            text = page.get_text()

            if text and text.strip():

                documents.append(
                    Document(
                        page_content=text.strip(),
                        metadata={
                            "source": pdf_file.name,
                            "page": page_index + 1,
                        },
                    )
                )

        pdf.close()

        print(
            f"Processed {pages_to_process}/{total_pages} pages"
        )

    return documents