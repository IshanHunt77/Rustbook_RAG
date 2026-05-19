import re
import bisect


def _code_fence_ranges(text):
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


def _inside_code_fence(pos, starts, ends):
    idx = bisect.bisect_right(starts, pos) - 1
    return idx >= 0 and pos <= ends[idx]


def _valid_breaks(text, pattern, offset, fence_starts, fence_ends):
    positions = []
    pos = 0
    while True:
        idx = text.find(pattern, pos)
        if idx == -1:
            break
        if not _inside_code_fence(idx, fence_starts, fence_ends):
            positions.append(idx + offset)
        pos = idx + 1
    return positions


def chunk_text(text, chunk_size=1500, overlap=350):
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    fence_ranges = _code_fence_ranges(text)
    fence_starts = [s for s, _ in fence_ranges]
    fence_ends   = [e for _, e in fence_ranges]

    para_breaks = _valid_breaks(text, '\n\n', 0,  fence_starts, fence_ends)
    sent_breaks = _valid_breaks(text, '. ',  +2, fence_starts, fence_ends)

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        break_pos = -1

        idx = bisect.bisect_right(para_breaks, end) - 1
        if idx >= 0 and para_breaks[idx] > start:
            break_pos = para_breaks[idx]

        if break_pos <= start:
            idx = bisect.bisect_right(sent_breaks, end) - 1
            if idx >= 0 and sent_breaks[idx] > start:
                break_pos = sent_breaks[idx]

        if break_pos <= start:
            break_pos = end

        chunk = text[start:break_pos].strip()
        if chunk:
            chunks.append(chunk)
        next_start = max(0, break_pos - overlap)
        start = next_start if next_start > start else break_pos

    return chunks
