# IO 

## Saving and Loading

```python
import yann

yann.save(x, 'data.yml')
yann.save(x, 'data.json')
yann.save(x, 'data.th')
yann.save(x, 'data.pkl')
yann.save(x, 'data.csv')
yann.save(x, 'data.parq')
yann.save(x, 'data.parquet')


yann.save(x, 's3://bucket-name/data.parquet')



x = yann.load('data.npz')
```



## Downloader


## Serialization


## Syncing