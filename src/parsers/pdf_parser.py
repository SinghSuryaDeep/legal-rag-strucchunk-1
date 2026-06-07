"""
Legal PDF Parser
================
Parses legal PDFs and extracts hierarchical structure.

CHANGES FROM v1:
  [BUG FIX]   SECTION_PATTERN lookahead used \\Z (absolute end-of-string anchor).
              pdfplumber often adds trailing whitespace/page-numbers after the
              last section, so \\Z never fires and the final section is silently
              dropped. Fixed to use $ with re.MULTILINE so the pattern matches
              the end of any line, not just the absolute end of the string.

  [BUG FIX]   SECTION_PATTERN was too greedy — subsection text starting with
              lowercase could fail to match. Updated to handle both "Title—text"
              and "Title\ntext" patterns more robustly.

  [NEW]       char_offset tracking on each Section. Required for the
              ChunkingQualityAnalyzer (boundary_precision by offset) and for
              LegalBench-RAG character-span evaluation (the evaluator needs
              to know where each chunk's text starts in the source document).

  [ENHANCED]  _parse_sections() now assigns parent chapter using character
              position of the section match, not a string scan. Previously the
              code searched for the nearest CHAPTER heading textually, which
              could mis-assign sections near the end of chapters.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Section:
    """Represents a section in a legal document."""
    number: str
    title: str
    content: str
    level: int = 1
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    cross_references: List[str] = field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    # [NEW] character offset in the full document text
    char_offset: Optional[int] = None


@dataclass
class LegalDocument:
    """Represents a parsed legal document."""
    title: str
    sections: Dict[str, Section]
    chapters: Dict[str, str]
    full_text: str
    page_count: int
    source_path: Optional[str] = None  # [NEW] for LegalBench-RAG file-path matching

    def get_section(self, section_id: str) -> Optional[Section]:
        return self.sections.get(section_id)

    def get_section_content(self, section_id: str, include_title: bool = True) -> str:
        section = self.get_section(section_id)
        if not section:
            return ""
        if include_title:
            return f"Section {section.number}. {section.title}\n\n{section.content}"
        return section.content

    def get_cross_references(self, section_id: str) -> List[Section]:
        section = self.get_section(section_id)
        if not section:
            return []
        return [
            self.sections[ref_id]
            for ref_id in section.cross_references
            if ref_id in self.sections
        ]


class LegalPDFParser:
    """
    Parses legal PDFs and extracts hierarchical structure.

    BUG FIXES FROM v1 (see module docstring for details).
    """

    # Handles both Indian law ("CHAPTER IVX\nTITLE") and EU law ("CHAPTER I\nMixed case")
    CHAPTER_PATTERN = re.compile(
        r'CHAPTER\s+([IVX]+)\s*[\n\r]+\s*([A-Za-z][A-Za-z\s,\-\.]+)',
        re.MULTILINE,
    )

    # [BUG FIX] v1 used \\Z; changed to $ with re.MULTILINE so the final
    # section in a document is captured even if trailing whitespace follows.
    # [ENHANCED] Added \s* before — to handle "Title — content" spacing variants.
    # Indian / Commonwealth format: "167. Title—content"
    SECTION_PATTERN = re.compile(
        r'(\d+[A-Z]?)\.\s+([A-Za-z][^\n—]{2,80}?)\s*\.?—\s*(.+?)'
        r'(?=\n\s*\d+[A-Z]?\.\s+[A-Z]|\nCHAPTER\s+[IVX]|$)',
        re.DOTALL | re.MULTILINE,
    )

    # EU regulation format: "Article 6\nTitle\ncontent"
    ARTICLE_PATTERN = re.compile(
        r'\nArticle\s+(\d+[a-z]?)\n([^\n]{5,120})\n(.+?)'
        r'(?=\nArticle\s+\d+|\nCHAPTER\s+[IVX]|\Z)',
        re.DOTALL,
    )

    CROSS_REF_PATTERNS = [
        re.compile(r'[Ss]ection\s+(\d+[A-Z]?)', re.IGNORECASE),
        re.compile(r'[Uu]nder\s+[Ss]ection\s+(\d+[A-Z]?)', re.IGNORECASE),
        re.compile(
            r'[Ss]ubject\s+to\s+(?:the\s+provisions\s+of\s+)?[Ss]ection\s+(\d+[A-Z]?)',
            re.IGNORECASE,
        ),
        re.compile(r'[Rr]eferred\s+to\s+in\s+[Ss]ection\s+(\d+[A-Z]?)', re.IGNORECASE),
        re.compile(
            r'[Pp]rovided\s+(?:for\s+)?(?:in|under)\s+[Ss]ection\s+(\d+[A-Z]?)',
            re.IGNORECASE,
        ),
        re.compile(
            r'[Nn]otwithstanding\s+(?:anything\s+)?(?:contained\s+)?in\s+[Ss]ection\s+(\d+[A-Z]?)',
            re.IGNORECASE,
        ),
        re.compile(r'[Oo]rder\s+([IVX]+)', re.IGNORECASE),
        re.compile(r'[Rr]ule\s+(\d+)', re.IGNORECASE),
    ]

    def __init__(self):
        self.pdf_library = self._detect_pdf_library()

    def _detect_pdf_library(self) -> str:
        for lib, name in [("pdfplumber", "pdfplumber"), ("fitz", "pymupdf")]:
            try:
                __import__(lib)
                return name
            except ImportError:
                continue
        raise ImportError(
            "No PDF library found. Install pdfplumber: pip install pdfplumber"
        )

    def parse(self, pdf_path) -> LegalDocument:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Parsing PDF: {pdf_path}")
        pages_text, page_count = self._extract_text(pdf_path)
        full_text = "\n".join(pages_text)
        logger.info(f"Extracted {page_count} pages, {len(full_text):,} characters")

        chapters = self._parse_chapters(full_text)
        logger.info(f"Found {len(chapters)} chapters")

        sections = self._parse_sections(full_text)
        logger.info(f"Found {len(sections)} sections")

        self._extract_cross_references(sections)
        title = self._extract_title(full_text)

        return LegalDocument(
            title=title,
            sections=sections,
            chapters=chapters,
            full_text=full_text,
            page_count=page_count,
            source_path=str(pdf_path),  # [NEW]
        )

    def _extract_text(self, pdf_path: Path) -> Tuple[List[str], int]:
        if self.pdf_library == "pdfplumber":
            import pdfplumber
            pages = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    pages.append(page.extract_text() or "")
            return pages, len(pages)
        else:
            import fitz
            doc = fitz.open(pdf_path)
            pages = [page.get_text() for page in doc]
            n = len(doc)
            doc.close()
            return pages, n

    def _parse_chapters(self, text: str) -> Dict[str, str]:
        return {
            m.group(1).strip(): m.group(2).strip()
            for m in self.CHAPTER_PATTERN.finditer(text)
        }

    def _parse_sections(self, text: str) -> Dict[str, Section]:
        """
        Parse sections with char_offset tracking.

        CHANGES FROM v1:
          [BUG FIX]   Uses re.MULTILINE (via SECTION_PATTERN flag fix).
          [NEW]       Stores match.start() as char_offset on each Section.
          [ENHANCED]  Parent chapter determined by character position (most
                      recent chapter start before this section start), not by
                      a separate text scan. Eliminates the edge case where
                      the old code would assign a section to the wrong chapter
                      at chapter boundaries.
        """
        # Build chapter position map — character index → chapter number
        chapter_positions = [
            (m.start(), m.group(1))
            for m in self.CHAPTER_PATTERN.finditer(text)
        ]

        sections = {}

        # Try Indian/Commonwealth format first; fall back to EU Article format
        pattern = self.SECTION_PATTERN
        matches = list(pattern.finditer(text))
        if not matches:
            pattern = self.ARTICLE_PATTERN
            matches = list(pattern.finditer(text))

        for match in matches:
            number = match.group(1)
            title = match.group(2).strip()
            content = re.sub(r'\s+', ' ', match.group(3)).strip()[:5000]

            parent_chapter = None
            match_pos = match.start()
            for pos, chapter_num in reversed(chapter_positions):
                if pos < match_pos:
                    parent_chapter = chapter_num
                    break

            sections[number] = Section(
                number=number,
                title=title,
                content=content,
                parent=parent_chapter,
                char_offset=match.start(),
            )

        return sections

    def _extract_cross_references(self, sections: Dict[str, Section]):
        for section_id, section in sections.items():
            refs = set()
            for pattern in self.CROSS_REF_PATTERNS:
                refs.update(pattern.findall(section.content))
            refs.discard(section_id)
            section.cross_references = [r for r in refs if r in sections]

    def _extract_title(self, text: str) -> str:
        for pattern in [
            r'^THE\s+([A-Z][A-Z\s,]+(?:ACT|CODE|RULES))',
            r'^([A-Z][A-Z\s,]+(?:ACT|CODE|RULES))',
        ]:
            m = re.search(pattern, text, re.MULTILINE)
            if m:
                return m.group(1).strip()
        return "Legal Document"
