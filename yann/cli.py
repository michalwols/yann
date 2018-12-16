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
@click.option('-c', '--continue')
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
  from .train import Trainer
  from .callbacks import get_callbacks

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


@cli.command()
@click.argument('names', nargs=-1)
def resolve(names):
  import yann

  if len(names) == 1:
    print(yann.resolve(names[0]))
  else:
    x = getattr(yann.resolve, names[0])
    for n in names[1:-1]:
      x = getattr(x, n)
    print(x(names[-1]))


@cli.command()
@click.argument('names', nargs=-1)
def registry(names):
  """List contents of registry"""
  import yann

  if not names:
    print(yann.registry)
  else:
    x = yann.registry
    for n in names:
      x = getattr(x, n)
    print(x)


def main():
  cli()


if __name__ == '__main__':
  cli()
