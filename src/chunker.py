import re


def _code_fence_ranges(text):
    """Return list of (start, end) char ranges that are inside ```...``` blocks."""
    ranges = []
    pos = 0
    while True:
        open_ = text.find('```', pos)
        if open_ == -1:
            break
        close = text.find('```', open_ + 3)
        if close == -1:
            break
        ranges.append((open_, close + 3))
        pos = close + 3
    return ranges


def _inside_code_fence(pos, fence_ranges):
    return any(s <= pos <= e for s, e in fence_ranges)


def chunk_text(text, chunk_size=1500, overlap=350):
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    fence_ranges = _code_fence_ranges(text)
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # prefer paragraph boundary outside a code fence
        break_pos = -1
        search_end = end
        while search_end > start:
            candidate = text.rfind('\n\n', start, search_end)
            if candidate <= start:
                break
            if not _inside_code_fence(candidate, fence_ranges):
                break_pos = candidate
                break
            search_end = candidate  # try earlier

        # fall back to sentence boundary outside a code fence
        if break_pos <= start:
            search_end = end
            while search_end > start:
                candidate = text.rfind('. ', start, search_end)
                if candidate <= start:
                    break
                if not _inside_code_fence(candidate, fence_ranges):
                    break_pos = candidate + 2
                    break
                search_end = candidate

        if break_pos <= start:
            break_pos = end

        chunk = text[start:break_pos].strip()
        if chunk:
            chunks.append(chunk)
        start = max(0, break_pos - overlap)

    return chunks
