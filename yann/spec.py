# import dataclasses
# from typing import Optional, Dict, Tuple
#
# import uuid
# from enum import Enum
#
#
# class ModelFormat(Enum):
#   onnx = 'onnx'
#   torchscript = 'torchscript'
#   torchmodule = 'torchmodule'
#   scikitlearn = 'scikitlearn'
#   python_function = 'python_function'
#
#
# class TabularFormats(Enum):
#   parquet = 'parquet'
#   csv = 'csv'
#   json = 'json'
#
#
#
# @dataclasses.dataclass()
# class Spec:
#   uri: str
#   name: Optional[str]
#   description: Optional[str]
#
#   slug: str  # human readable id, used in urls
#   version: str
#
#   time_created: int
#
#   metadata: Dict
#
#   source: str
#   tags: Tuple[str]
#
#
# @dataclasses.dataclass()
# class Dataset(Spec):
#   fields: Dict
#   type: str
#   params: dict
#
#
#
# @dataclasses.dataclass()
# class Data(Spec):
#   format: str
#   shape: Tuple
#
# class Vector(Data):
#   pass
#
# class Image(Data):
#   pass
#
# class TabularData(Dataset)
#   fields:List
#
# class Image(Data):
#   pass
#
# @dataclasses.dataclass()
# class Operator(Spec):
#   inputs = Tuple
#   outputs = Tuple
#
#
# @dataclasses.dataclass()
# class Model(Operator):
#   experiment: Optional[str]
#   checkpoint: str
#
#   type: str
#
#
# @dataclasses.dataclass()
# class Predictor(Operator):
#   pass
#
#
#
#
# dataset = TabularData(
#   fields=[
#     Image()
#
#   ]
# )
#
#
#
#
#
# dataset = Data(
#   name='imagenet-b',
#   fields=dict(
#     image=Image(),
#
#   )
# )
#
#
#
#
# model_spec = Model(
#   id=uuid.uuid4(),
#   inputs=(
#     Image(),
#   ),
#   outputs=
# )
#
#
# index_spec = Data(
#   id=uuid.uuid4(),
#   name='index',
#   type='annoy',
#   params=dict()
# )
#
#
#
