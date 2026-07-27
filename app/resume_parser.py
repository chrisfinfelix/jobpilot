"""
resume_parser.py
-----------------
Extracts raw text from an uploaded resume file.
Supports: PDF (.pdf), Word (.docx), and plain text.

This module has ONE job: turn "whatever the user uploaded" into a clean
text string. It does not judge, score, or interpret the content — that
happens later in the matching step.
"""

import io
from docx import Document
import pdfplumber


class UnsupportedFileType(Exception):
    """Raised when the uploaded file extension isn't one we support."""
    pass


class EmptyResumeError(Exception):
    """Raised when we couldn't extract any usable text from the file."""
    pass


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file's raw bytes."""
    text_chunks = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)
    return "\n".join(text_chunks)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from a .docx file's raw bytes."""
    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

    # Also pull text out of any tables (some resumes use table layouts)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    paragraphs.append(cell.text)

    return "\n".join(paragraphs)


def parse_resume(filename: str | None, file_bytes: bytes | None, pasted_text: str | None) -> str:
    """
    Main entry point. Accepts EITHER an uploaded file OR pasted text and
    returns clean extracted text.

    Priority: if pasted_text is provided, use it directly (no parsing needed).
    Otherwise, look at the filename extension to decide how to parse the file.
    """
    if pasted_text and pasted_text.strip():
        text = pasted_text.strip()
    elif filename and file_bytes:
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        if ext == "pdf":
            text = extract_text_from_pdf(file_bytes)
        elif ext == "docx":
            text = extract_text_from_docx(file_bytes)
        elif ext == "txt":
            text = file_bytes.decode("utf-8", errors="ignore")
        else:
            raise UnsupportedFileType(
                f"'.{ext}' is not supported. Please upload a PDF, DOCX, or paste text."
            )
    else:
        raise EmptyResumeError("No resume file or pasted text was provided.")

    text = text.strip()
    if len(text) < 30:
        # A real resume is always longer than this. This usually means
        # the PDF was a scanned image with no selectable text layer.
        raise EmptyResumeError(
            "Couldn't extract readable text from this resume. "
            "If it's a scanned/image-based PDF, try pasting the text instead."
        )

    return text
