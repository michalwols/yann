import click


@click.group()
def cli():
  pass


@cli.command()
@click.option('-n', '--name', default=None)
@click.option('-d', '--dataset', default=None)
@click.option('-opt', '--optimizer', default='SGD', show_default=True)
@click.option('-m', '--model', default='resnet18', show_default=True)
@click.option('-e', '--epochs', default=10, show_default=True)
@click.option('-l', '--loss')
@click.option('-cp', '--checkpoint')
def train(
    name,
    model,
    dataset,
    loss=None,
    transform=None,
    optimizer='SGD',
    checkpoint=None,
    lr=0.01,
    momentum=.9,
    epochs=10
):
  """Train model"""
  from yann.train import Trainer
  from yann.callbacks import get_callbacks

  t = Trainer(
    name=name,
    model=model,
    optimizer=optimizer,
    dataset=dataset,
    transform=transform,
    loss=loss,
    callbacks=get_callbacks(interactive=False)
  )

  if checkpoint:
    t.load_checkpoint(checkpoint)

  t(epochs)


@cli.command()
def tune():
  pass


@cli.command()
def evaluate(self):
  """Evaluate model on data"""
  pass


@cli.command()
def validate():
  pass


def main():
  cli()


if __name__ == '__main__':
  cli()
