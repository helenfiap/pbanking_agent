"""
Helper to extract plain text from LangChain LLM responses.

Newer LangChain versions (+ some models) return response.content as a list
of content blocks: [{"type": "text", "text": "..."}] instead of a plain string.
This helper handles both formats transparently.
"""


def extract_text(response) -> str:
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Extract text from first text block
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "")
            if isinstance(block, str):
                return block
    return str(content)
