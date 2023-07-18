import click

import yann


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
    yann.registry.print_tree()
  else:
    registry = yann.registry
    for n in names:
      registry = getattr(registry, n)
    registry.print_tree()


@cli.command()
def scaffold():
  """
  TODO: use cookiecutter to scaffold a new project
  https://github.com/drivendata/cookiecutter-data-science

  support different scaffolds

  data/
    raw/
    processed/
  train-runs/
  notebooks/
  docs/
  tests/
  {{project_name}}/
    models/
    datasets/
    cli.py
    train.py
    evaluate.py
    serve.py
  requirements.txt
  conda.yml
  setup.py
  dockerfile
  run
  README.md


  run
    prepare-data()
    test()
    train()
    evaluate()
    install-dependencies()
    save-dependencies()
    demo()
    deploy()
  """
  raise NotImplementedError()




@cli.command()
@click.argument('src')
@click.argument('dst')
def convert(src: str, dst: str):
  import yann
  data = yann.load(src)
  print(f'loaded {type(data)}')
  yann.save(data, dst)

def main():
  cli()


@cli.group()
def dataset():
  pass


@dataset.command()
@click.argument('name')
def preview(name: str):
  count = 10

  ds = yann.resolve.dataset(name)

  print(yann.utils.fully_qualified_name(ds))
  print('length:', len(ds))
  for i in range(count):
    x = ds[i]
    print(x)

@dataset.command()
def list():
  print(yann.registry.dataset.print_tree())

if __name__ == '__main__':
  cli()
