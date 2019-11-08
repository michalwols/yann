

class YannException(Exception):
  pass


class ShapeInferenceError(YannException):
  pass


class CheckFailure(YannException):
  pass