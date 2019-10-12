# Registry

Yann provides an object registry that enables it to resolve instances from strings.
You can register your own objects to make them discoverable by the system.


## Resolving by name
```python
import yann


resnet = yann.resolve.model('resnet18', pretrained=True)
```


## Registering your own objects

```python
import yann

@yann.register.loss
def custom_loss():
  pass
```