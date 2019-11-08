from PIL import Image
import io


def image_to_bytes(image: Image.Image, format='jpeg'):
  buff = io.BytesIO()
  image.save(buff, format=format)
  return buff.getvalue()


def image_from_bytes(buffer):
  return Image.open(io.BytesIO(buffer))


def enable_loading_truncated_images():
  from PIL import ImageFile
  ImageFile.LOAD_TRUNCATED_IMAGES = True