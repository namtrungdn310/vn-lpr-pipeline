"""Giữ mọi model trên cùng bốn lõi vật lý; không đổi power plan Windows."""

import ctypes
import hashlib
import os
from pathlib import Path
import struct

import psutil
import onnxruntime as ort


def signature():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def topology():
    if os.name != "nt":
        raise RuntimeError("Protocol này cần topology Windows đã kiểm")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    api = kernel.GetLogicalProcessorInformationEx
    api.argtypes = [ctypes.c_ulong, ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    api.restype = ctypes.c_int
    size = ctypes.c_ulong()
    api(0, None, ctypes.byref(size))
    data = ctypes.create_string_buffer(size.value)
    if not api(0, data, ctypes.byref(size)):
        raise ctypes.WinError(ctypes.get_last_error())
    rows, offset = [], 0
    while offset < size.value:
        relation, length = struct.unpack_from("<II", data.raw, offset)
        if not length:
            raise RuntimeError("Topology có bản ghi rỗng")
        if relation == 0:
            groups = struct.unpack_from("<H", data.raw, offset + 30)[0]
            for i in range(groups):
                mask, group = struct.unpack_from("<QH", data.raw, offset + 32 + i * 16)
                if group != 0:
                    raise RuntimeError("Chưa hỗ trợ nhiều processor group")
                rows.append({"efficiency": data.raw[offset + 9],
                             "logical": [j for j in range(64) if mask & (1 << j)]})
        offset += length
    return rows


def select(rows, count=4):
    highest = max(r["efficiency"] for r in rows)
    cores = [r["logical"][0] for r in rows if r["efficiency"] == highest]
    if len(cores) < count:
        raise RuntimeError("Không đủ bốn lõi vật lý cùng loại hiệu năng")
    return sorted(cores[:count])


def apply(expected=None):
    rows = topology()
    selected = select(rows)
    context = {"topology": rows, "logical_cpus": selected, "physical_cores": 4,
               "policy": "one logical CPU per physical core in highest EfficiencyClass", "code_signature": signature(),
               "session_options": {"session.intra_op.allow_spinning": "0", "session.inter_op.allow_spinning": "0"}}
    if expected is not None and expected != context:
        raise RuntimeError("Topology/affinity khác lượt đo đã chốt")
    psutil.Process().cpu_affinity(selected)
    if sorted(psutil.Process().cpu_affinity()) != selected:
        raise RuntimeError("Không áp dụng được CPU affinity")
    configure_sessions()
    return context


def configure_sessions():
    """Áp dụng cùng option cho mọi adapter mà không đổi code/model nguồn."""
    if getattr(ort.SessionOptions, "_fixed_cpu_protocol", False):
        return
    original = ort.SessionOptions
    def options():
        result = original()
        result.add_session_config_entry("session.intra_op.allow_spinning", "0")
        result.add_session_config_entry("session.inter_op.allow_spinning", "0")
        return result
    options._fixed_cpu_protocol = True
    ort.SessionOptions = options


def check(expected):
    if sorted(psutil.Process().cpu_affinity()) != expected["logical_cpus"]:
        raise RuntimeError("CPU affinity đã thay đổi")
