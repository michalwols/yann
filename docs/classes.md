# Classes


```python

from yann.data import Classes

classes = Classes(['apple', 'banana', 'fruit'])

classes.names[0]
classes.indices['apple']
classes.counts[0]
classes.weights()


classes.encode(['apple'])

classes.decode([0, 1])

"apple" in classes


classes.ranked_decode([.3, .5])

classes.one_hot_encode(['apple'])
classes.index_encode(['apple'])
classes.index_decode([1, 2])
```

### Label Smoothing

```python
from yann.data.classes import smooth

smooth(classes.encode(['apple']), num_classes=len(classes))
```


### Class Weights