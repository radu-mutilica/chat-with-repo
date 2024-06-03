from langchain_text_splitters import Language

_language_map = {
    ".cpp": Language.CPP,
    ".go": Language.GO,
    ".java": Language.JAVA,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.JS,
    ".tsx": Language.JS,
    ".php": Language.PHP,
    ".proto": Language.PROTO,
    ".py": Language.PYTHON,
    ".rst": Language.RST,
    ".rb": Language.RUBY,
    ".rs": Language.RUST,
    ".scala": Language.SCALA,
    ".swift": Language.SWIFT,
    ".md": Language.MARKDOWN,
    ".tex": Language.LATEX,
    ".html": Language.HTML,
    ".htm": Language.HTML,
    ".sol": Language.SOL,
    ".css": Language.HTML,
    ".txt": Language.MARKDOWN,
    ".json": Language.MARKDOWN,
}


def identify_language(extension: str) -> Language:
    """Helper mapper function"""
    return _language_map.get(extension, Language.MARKDOWN)
