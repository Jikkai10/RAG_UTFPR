import hashlib
from typing import List, Optional, Tuple


def stableDocId(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:32]


def elementPosition(element) -> Tuple[Optional[int], Optional[List[float]]]:
    meta = getattr(element, "metadata", None)
    page = getattr(meta, "page_number", None)
    coords = getattr(meta, "coordinates", None)

    bbox = None
    if coords is not None and coords.points and coords.system is not None:
        xs = [point[0] for point in coords.points]
        ys = [point[1] for point in coords.points]
        width = coords.system.width or 1
        height = coords.system.height or 1
        bbox = [
            min(xs) / width,
            min(ys) / height,
            max(xs) / width,
            max(ys) / height,
        ]

    return page, bbox


def mergePositions(positions: List[Tuple]) -> Tuple[Optional[int], Optional[int], Optional[List[float]]]:
    pages = [page for page, _ in positions if page is not None]
    start = min(pages) if pages else None
    end = max(pages) if pages else None

    boxes = [
        bbox for page, bbox in positions
        if bbox and (start is None or page == start)
    ]

    bbox = None
    if boxes:
        bbox = [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ]

    return start, end, bbox
