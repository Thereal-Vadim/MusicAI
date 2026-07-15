"""Export TabDocument to Guitar Pro and AlphaTex."""

from tab_export.alphatex_exporter import document_to_alphatex
from tab_export.gp5_exporter import document_to_gp5_bytes

__all__ = ["document_to_alphatex", "document_to_gp5_bytes"]
