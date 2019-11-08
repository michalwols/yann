from yann.data.io.download import Downloader


class RemoteDataset:
  def __init__(self, downloader: Downloader):
    self.downloader = downloader

  def prefetch(self, indices=None):
    for n in indices:
      self.downloader.enqueue(self[n][0])

  def __getitem__(self, item):
    return 'id', ['labels']

def test_downloader():
  downloader = Downloader('.')

  local_path = downloader['gs://foo']

  dataset = RemoteDataset(downloader)

  for uri, labels in dataset:
    downloader.enqueue(uri)


  for key, result in downloader.results:
    pass

  for key, result in downloader.as_completed():
    pass


  for key, error in downloader.errors:
    pass


  downloader.enqueue('asd')


  downloader.on_error(lambda key, error: print(error))
  downloader.on_success(lambda key, result: print('completed', key))

  downloader.complete()


  def unpack(x):
    pass


  dataset_downloader = Downloader('./')

  dataset_downloader.on_success(unpack)
  dataset_downloader.enqueue('s3://1.tar')


  dataset_downloader.get_all(['1', '2', '3'])
  dataset_downloader.stream(['1', '2', '3'])