"""Read the end of the platform's flat, chronological CSV files."""
from datetime import datetime, timezone
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


def csv_tail_through(path: Path, max_rows: int, time_column: str, end_time: str) -> BytesIO:
    """Find the last matching rows through a UTC cutoff in an operational CSV."""
    cutoff = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    lines = []
    with path.open("rb") as file:
        header = file.readline().rstrip(b"\r\n")
        start = file.tell()
        columns = header.decode("utf-8-sig").split(",")
        if time_column not in columns:
            return BytesIO(header + b"\n")
        column = columns.index(time_column)
        file.seek(0, 2)
        position = file.tell()
        carry = b""
        while position > start and len(lines) < max_rows:
            left = max(start, position - 64 * 1024)
            file.seek(left)
            block = file.read(position - left) + carry
            position = left
            parts = block.split(b"\n")
            carry = parts.pop(0) if left > start else b""
            for line in reversed(parts):
                fields = line.split(b",", column + 1)
                if len(fields) <= column or not fields[column]:
                    continue
                try:
                    timestamp = datetime.fromisoformat(fields[column].decode().strip('"').replace("Z", "+00:00"))
                except ValueError:
                    continue
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                if timestamp <= cutoff:
                    lines.append(line.rstrip(b"\r"))
                    if len(lines) == max_rows:
                        break
    return BytesIO(header + b"\n" + b"\n".join(reversed(lines)))
