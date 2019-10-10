import pyarrow


ctx = pyarrow.serialization.default_serialization_context()

def enable_torch_serialization(context=ctx):
  pyarrow.serialization.register_torch_serialization_handlers(
    context
  )

# enable by default
enable_torch_serialization()


def deserialize_arrow(buf, context=ctx):
  return pyarrow.deserialize(buf, context)


def serialize_arrow(x, context=ctx):
  return pyarrow.serialize(x, context=context).to_buffer()


# use arrow as default serialization method
serialize = serialize_arrow
deserialize = deserialize_arrow


def to_bytes(string: str) -> bytes:
  return string.encode(encoding='utf-8', errors='strict')


def to_unicode(b: bytes) -> str:
  return b.decode(encoding='utf-8', errors='strict')