"""Bounded extraction of user-supplied documents for long-term memory."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import PurePath
from typing import Any

MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_EXTRACTED_BYTES = 32 * 1024
SUPPORTED_SUFFIXES = {".txt", ".md", ".csv", ".json", ".pdf"}


class UploadValidationError(ValueError):
    """User-facing upload validation failure."""


def extract_document(filename: str, content: bytes) -> dict[str, Any]:
    safe_name = PurePath(filename.replace("\\", "/")).name
    if not safe_name or len(safe_name) > 200:
        raise UploadValidationError("文件名无效或过长")
    suffix = PurePath(safe_name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UploadValidationError("只支持 PDF、TXT、MD、CSV 和 JSON 文件")
    if not content or len(content) > MAX_FILE_BYTES:
        raise UploadValidationError("文件不能为空，且不能超过 2 MB")
    if suffix == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise UploadValidationError("文件不是有效的 PDF")
        try:
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(content), strict=False)
            if reader.is_encrypted:
                raise UploadValidationError("不支持加密 PDF")
            text = "\n".join(page.extract_text() or "" for page in reader.pages[:100])
        except UploadValidationError:
            raise
        except Exception as exc:
            raise UploadValidationError("PDF 文本提取失败") from exc
        mime_type = "application/pdf"
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise UploadValidationError("文本文件必须使用 UTF-8 编码") from exc
        mime_type = "text/plain; charset=utf-8"
    if not text.strip():
        raise UploadValidationError("未提取到文本；扫描版 PDF 暂不支持 OCR")
    encoded = text.encode("utf-8")
    truncated = len(encoded) > MAX_EXTRACTED_BYTES
    selected = encoded[:MAX_EXTRACTED_BYTES].decode("utf-8", errors="ignore")
    return {
        "filename": safe_name,
        "text": selected,
        "mime_type": mime_type,
        "byte_size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "truncated": truncated,
    }
