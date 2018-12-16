import urllib
import urllib.request
import json

from ..callbacks.base import Callback

def post(url, data):
  data = json.dumps(data).encode('utf8')
  req = urllib.request.Request(
    url, data=data, headers={'content-type': 'application/json'})
  return urllib.request.urlopen(req)


DEFAULT_CHANNEL = '#training'

def send(text, attachments=None, channel=None, username=None, icon=None,
         url=None):
  """https://api.slack.com/docs/message-attachments"""
  return post(url, {
    'text': text,
    'channel': channel or DEFAULT_CHANNEL,
    'username': username,
    'icon_emoji': icon and (icon if icon.startswith(':') else f':{icon}:'),
    'attachments': attachments,
  })


def atch(title=None, text=None, fields=None, color=None, **kwargs):
  return dict(
    title=title,
    text=text,
    fields=[
      *(fields or {}),
      *({'title': k, 'value': v} for k, v in kwargs.items())],
    color=color)




class Slack(Callback):
  # TODO: subclass Logging callback instead
  def __init__(self, channel=None, username=None, url=None, validation=False):
    self.channel = channel
    self.username = username
    self.url = url

    self.validation = validation

  def send(self, *args, **kwargs):
    send(
      *args,
      channel=self.channel,
      username=self.username,
      url=self.url,
      **kwargs
    )

  def on_train_start(self, trainer=None):
    self.send(
      text='Starting train run',
      attachments=[
        atch(experiment=trainer.name, text=trainer.description),
        atch('Configuration', f"```{trainer}```", color='good'),
      ]
    )

  def on_validation_end(self, targets=None, outputs=None, loss=None,
                        trainer=None):
    if self.validation:
      self.send(
        text=f'Completed epoch {trainer.num_epochs} with loss: {loss.item()}'
      )

  def on_error(self, error, trainer=None):
    self.send(
      text='Training run failed due to an exception',
      attachments=[
        atch(experiment=trainer.name),
        atch(epoch=trainer.num_epochs + 1),
        atch('Exception', f"```{str(error)}```", color='danger'),
      ]
    )

  def on_train_end(self, trainer=None):
    self.send(text='Train run completed')