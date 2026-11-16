"""Read the end of the platform's flat, chronological CSV files."""
from io import BytesIO
from pathlib import Path


def csv_tail(path: Path, max_rows: int) -> BytesIO:
    """Return the header and the final data lines without reading earlier blocks."""
    with path.open("rb") as file:
        header = file.readline().rstrip(b"\r\n")
        header_end = file.tell()
        file.seek(0, 2)
        position = file.tell()
        chunks: list[bytes] = []
        newline_count = 0
        block_size = 64 * 1024

        while position > header_end and newline_count <= max_rows:
            start = max(header_end, position - block_size)
            file.seek(start)
            block = file.read(position - start)
            chunks.insert(0, block)
            newline_count += block.count(b"\n")
            position = start

    body = b"".join(chunks)
    lines = body.splitlines()
    # If reading began in the middle of a record, discard that partial line.
    if position > header_end and lines:
        lines = lines[1:]
    lines = lines[-max_rows:]
    return BytesIO(header + b"\n" + b"\n".join(lines))
