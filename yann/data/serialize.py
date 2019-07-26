import pyarrow


pyarrow.serialization.register_torch_serialization_handlers(
  pyarrow.serialization.default_serialization_context()
)


def deserialize_arrow(buf):
  return pyarrow.deserialize(buf)


def serialize_arrow(x):
  return pyarrow.serialize(x).to_buffer()


serialize = serialize_arrow
deserialize = deserialize_arrow


def to_bytes(string: str) -> bytes:
  return string.encode(encoding='utf-8', errors='strict')


def to_unicode(b: bytes) -> str:
  return b.decode(encoding='utf-8', errors='strict')