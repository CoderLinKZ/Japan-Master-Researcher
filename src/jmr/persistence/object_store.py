"""Small object-store implementations used by persistence ports and tests."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlparse
from uuid import uuid4


@dataclass(slots=True)
class InMemoryObjectStore:
    """Deterministic fake for PDF/HTML payloads kept outside Graph State."""

    _objects: dict[str, bytes] = field(default_factory=dict)

    def put(self, *, object_key: str, content: bytes, content_type: str) -> str:
        if not isinstance(object_key, str) or not object_key.strip():
            raise ValueError("object_key must be a non-empty string")
        if not isinstance(content, bytes):
            raise TypeError("content must be bytes")
        if not isinstance(content_type, str) or not content_type.strip():
            raise ValueError("content_type must be a non-empty string")
        uri = f"memory://{object_key.strip()}"
        self._objects[uri] = bytes(content)
        return uri

    def get(self, *, object_uri: str) -> bytes:
        try:
            return self._objects[object_uri]
        except KeyError as exc:
            raise KeyError(f"unknown object URI: {object_uri}") from exc

    def metadata(self, *, object_uri: str, content_type: str) -> dict[str, object]:
        content = self.get(object_uri=object_uri)
        return {
            "object_uri": object_uri,
            "sha256": hashlib.sha256(content).hexdigest(),
            "mime_type": content_type,
            "byte_size": len(content),
        }

    def delete(self, *, object_uri: str) -> None:
        try:
            del self._objects[object_uri]
        except KeyError as exc:
            raise KeyError(f"unknown object URI: {object_uri}") from exc

    def iter_uris(self, *, prefix: str = "") -> list[str]:
        return sorted(
            uri
            for uri in self._objects
            if not prefix or uri.removeprefix("memory://").startswith(prefix)
        )


@dataclass(slots=True)
class FileObjectStore:
    """Persistent, traversal-safe object store for a mounted volume.

    Production installations may replace this port with S3-compatible storage;
    this implementation provides durable local and CI semantics, atomic writes,
    immutable object keys, and a directory that can be backed up independently.
    """

    root: Path | str

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not self.root.is_dir():
            raise ValueError("object store root must be a directory")

    def put(self, *, object_key: str, content: bytes, content_type: str) -> str:
        del content_type  # metadata remains owned by the business repository
        path, normalized_key = self._path_for_key(object_key)
        if not isinstance(content, bytes):
            raise TypeError("content must be bytes")
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if path.exists():
            if path.read_bytes() != content:
                raise FileExistsError("object key already contains different content")
            return self._uri(normalized_key)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                os.chmod(temporary, 0o600)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != content:
                    raise FileExistsError(
                        "object key already contains different content"
                    ) from None
        finally:
            if temporary.exists():
                temporary.unlink()
        return self._uri(normalized_key)

    def get(self, *, object_uri: str) -> bytes:
        path = self._path_for_uri(object_uri)
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise KeyError(f"unknown object URI: {object_uri}") from exc

    def delete(self, *, object_uri: str) -> None:
        path = self._path_for_uri(object_uri)
        try:
            path.unlink()
        except FileNotFoundError as exc:
            raise KeyError(f"unknown object URI: {object_uri}") from exc

    def iter_uris(self, *, prefix: str = "") -> list[str]:
        normalized_prefix = ""
        if prefix:
            _, normalized_prefix = self._path_for_key(prefix)
        result = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            key = path.relative_to(self.root).as_posix()
            if key.startswith(normalized_prefix):
                result.append(self._uri(key))
        return result

    def _path_for_uri(self, object_uri: str) -> Path:
        parsed = urlparse(object_uri)
        if parsed.scheme != "jmr-object" or parsed.netloc:
            raise ValueError("object_uri must use the jmr-object scheme")
        path, _ = self._path_for_key(unquote(parsed.path.lstrip("/")))
        return path

    def _path_for_key(self, object_key: str) -> tuple[Path, str]:
        if not isinstance(object_key, str) or not object_key.strip():
            raise ValueError("object_key must be a non-empty string")
        if "\\" in object_key:
            raise ValueError("object_key must use POSIX separators")
        key = PurePosixPath(object_key.strip())
        if key.is_absolute() or any(part in {"", ".", ".."} for part in key.parts):
            raise ValueError("object_key contains an unsafe path")
        normalized = key.as_posix()
        candidate = (self.root / normalized).resolve()
        if self.root not in candidate.parents:
            raise ValueError("object_key escapes the object store root")
        return candidate, normalized

    @staticmethod
    def _uri(object_key: str) -> str:
        return f"jmr-object:///{quote(object_key, safe='/')}"
