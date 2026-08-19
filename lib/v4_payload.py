"""One canonical reader for v4 payloads, and an explicit legacy reader for v3.

Different modules were each doing their own ``np.load(...).item()`` against
files whose extension did not match their contents -- the v3 payloads are
``.pkl`` names holding NumPy object containers. That worked until it didn't, and
it left no single place to state what a payload must contain.

v4 files are real pickles with a fixed schema and are validated on read. There
is deliberately no silent fallback: a v4 file that fails the schema raises
rather than being reinterpreted as legacy.
"""

import pickle
from pathlib import Path

import numpy as np

PICKLE_PROTOCOL = 5
SCHEMA_VERSION = 4
# The payload declares its own schema. Sniffing the container was tried first and
# is wrong: data/METR-LA/raw_data.pkl is a genuine pickle holding float64 arrays,
# so "is it a pickle?" classifies a v3 payload as v4 and then rejects it for a
# defect it was never claiming not to have.
V4_FIELDS = ("schema_version", "data", "adj", "time_stamp", "pos_stamp")


class InvalidPayload(ValueError):
    """A payload does not satisfy the canonical v4 schema."""


def save_v4_payload(path, data, adj, time_stamp, pos_stamp):
    """Write the canonical v4 pickle. ``pos_stamp`` may be None."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "data": np.ascontiguousarray(data, dtype=np.float32),
        "adj": np.ascontiguousarray(adj, dtype=np.float32),
        "time_stamp": np.asarray(time_stamp),
        "pos_stamp": (None if pos_stamp is None
                      else np.ascontiguousarray(pos_stamp, dtype=np.float32)),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=PICKLE_PROTOCOL)
    return path


def load_v4_payload(path):
    """Read and validate a canonical v4 payload. No fallback, no repair."""
    path = Path(path)
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as error:                                # noqa: BLE001
        raise InvalidPayload(
            f"{path}: not a canonical v4 pickle ({type(error).__name__}: {error}). "
            "v3 payloads are NumPy object containers and must be read with "
            "load_legacy_payload instead.") from None

    if not isinstance(payload, dict):
        raise InvalidPayload(f"{path}: expected a dict, got {type(payload).__name__}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise InvalidPayload(
            f"{path}: declares schema_version {payload.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION}")
    missing = [f for f in V4_FIELDS if f not in payload]
    extra = sorted(set(payload) - set(V4_FIELDS))
    if missing or extra:
        raise InvalidPayload(f"{path}: missing {missing}, unexpected {extra}")

    for field in ("data", "adj"):
        array = payload[field]
        if not isinstance(array, np.ndarray) or array.dtype != np.float32:
            raise InvalidPayload(
                f"{path}: {field} must be a float32 ndarray, got "
                f"{getattr(array, 'dtype', type(array).__name__)}")
    if payload["data"].ndim != 3:
        raise InvalidPayload(f"{path}: data must be (time, node, channel)")
    adjacency = payload["adj"]
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise InvalidPayload(f"{path}: adj must be square, got {adjacency.shape}")
    if adjacency.shape[0] != payload["data"].shape[1]:
        raise InvalidPayload(
            f"{path}: adj has {adjacency.shape[0]} nodes but data has "
            f"{payload['data'].shape[1]}")
    if len(payload["time_stamp"]) != payload["data"].shape[0]:
        raise InvalidPayload(f"{path}: time_stamp length != data time steps")

    positions = payload["pos_stamp"]
    if positions is not None:
        if not isinstance(positions, np.ndarray) or positions.dtype != np.float32:
            raise InvalidPayload(f"{path}: pos_stamp must be float32 or None")
        if positions.shape[0] != adjacency.shape[0]:
            raise InvalidPayload(f"{path}: pos_stamp rows != node count")
    return payload


def load_legacy_payload(path):
    """Read a v3 NumPy object container. Explicit, never a fallback."""
    try:
        blob = np.load(path, allow_pickle=True)
    except ValueError:
        with Path(path).open("rb") as handle:      # a plain v3 pickle, not a .npy
            blob = pickle.load(handle)
    if isinstance(blob, np.ndarray) and blob.shape == ():
        blob = blob.item()
    if isinstance(blob, dict):
        return blob
    if hasattr(blob, "files"):
        return {key: blob[key] for key in blob.files}
    raise InvalidPayload(f"{path}: unsupported legacy payload type {type(blob).__name__}")


NPY_MAGIC = b"\x93NUMPY"


def payload_schema(path):
    """Which schema a file declares.

    v4 payloads carry ``schema_version``; anything else is v3 legacy, whether it
    is a NumPy object container or a plain pickle. This is a declaration, not a
    guess -- a file that says it is v4 is then held to the full v4 schema and
    never quietly reinterpreted as legacy.
    """
    path = Path(path)
    with path.open("rb") as handle:
        if handle.read(6) == NPY_MAGIC:
            return "v3_legacy"
        handle.seek(0)
        try:
            payload = pickle.load(handle)
        except Exception:                                          # noqa: BLE001
            return "v4"      # unreadable: let the strict reader report why
    declared = isinstance(payload, dict) and payload.get("schema_version") == SCHEMA_VERSION
    return "v4" if declared else "v3_legacy"


def load_raw_payload(path, schema="auto"):
    """The single entry point for reading a dataset payload.

    Modules used to carry their own ``np.load(..., allow_pickle=True).item()``
    spelling, which meant there was no one place that said what a payload must
    contain and no way to tighten it. Pass an explicit ``schema`` to assert the
    format; ``auto`` follows the payload's own declaration and still validates
    v4 strictly.
    """
    path = Path(path)
    if schema not in ("auto", "v4", "v3_legacy"):
        raise ValueError(f"unknown schema {schema!r}")
    detected = payload_schema(path)
    if schema not in ("auto", detected):
        raise InvalidPayload(f"{path}: expected {schema} payload, found {detected}")
    return load_v4_payload(path) if detected == "v4" else load_legacy_payload(path)
