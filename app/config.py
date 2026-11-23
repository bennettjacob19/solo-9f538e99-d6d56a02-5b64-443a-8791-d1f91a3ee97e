"""环境变量 / 配置解析工具。"""
from __future__ import annotations


_SIZE_UNITS = {
    "": 1,
    "B": 1,
    "K": 1024, "KB": 1024,
    "M": 1024 ** 2, "MB": 1024 ** 2,
    "G": 1024 ** 3, "GB": 1024 ** 3,
    "KIB": 1024, "MIB": 1024 ** 2, "GIB": 1024 ** 3,
}


def parse_size(value: str | int) -> int:
    """把 ``4.5MiB`` / ``1024`` 这类大小字符串解析成字节数。"""
    if isinstance(value, int):
        return value
    text = str(value).strip().upper().replace(" ", "")
    if not text:
        raise ValueError("空的大小配置")
    unit = ""
    for suffix in ("MIB", "GIB", "KIB", "MB", "GB", "KB", "M", "G", "K", "B"):
        if text.endswith(suffix):
            unit = suffix
            text = text[: -len(suffix)] if suffix else text
            break
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"无法解析大小: {value!r}") from exc
    return int(number * _SIZE_UNITS[unit])


def parse_float(value: str | float | int) -> float:
    return float(value)


def parse_int(value: str | int) -> int:
    return int(value)


def parse_csv(value: str | list | None) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]
