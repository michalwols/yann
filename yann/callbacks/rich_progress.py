import subprocess  # For git hash
import sys
import time

import torch  # For version/CUDA info

# Import necessary rich components
from rich import print

# from rich.live import Live # No longer needed
# from rich.console import Group # No longer needed
from rich.panel import Panel
from rich.progress import (
  BarColumn,
  MofNCompleteColumn,
  Progress,
  TextColumn,
  TimeElapsedColumn,
  TimeRemainingColumn,
)
from rich.table import Table

import yann.data  # Ensure yann.data is imported
import yann.utils
from yann.utils.tensor import describe  # For batch shapes

from .base import Callback


class RichProgress(Callback):
  """
  A rich-based logging callback for PyTorch training loops.

  Displays:
      - Progress bars for overall epochs and steps within the current epoch.
      - Initial summary panels (Parameters, Model, Dataset, DataLoader, Setup, Env).
      - Periodic log lines during training steps (Metrics, Optimizer state, Timing, GPU Mem).
      - Validation results summary after each validation phase.
      - Epoch duration summary.
      - Final summary (Total time, Best validation metrics).

  Args:
      log_interval_seconds (int): Frequency (in seconds) for printing periodic log lines
          during training steps. Defaults to 5.
      refresh_per_second (int): Refresh rate for the progress bars. Defaults to 10.
      disable (bool): If True, disable the progress bars entirely. Defaults to False.
      transient (bool): If True, progress bars will be removed after completion.
          Set to False to keep them visible. Defaults to False.
  """

  def __init__(
    self,
    log_interval_seconds: int = 5,
    refresh_per_second: int = 10,
    disable: bool = False,
    transient: bool = False,
  ):
    """Initializes the RichProgress callback and the underlying rich Progress object."""

    self.progress = Progress(
      TextColumn('[progress.description]{task.description}'),
      BarColumn(),
      MofNCompleteColumn(),
      TextColumn('•'),
      TimeElapsedColumn(),
      TextColumn('•'),
      TimeRemainingColumn(),
      TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
      refresh_per_second=refresh_per_second,
      disable=disable,
      transient=transient,
    )

    self.log_interval_seconds = log_interval_seconds
    self.last_log_time = 0.0
    self.tasks = {}
    self.last_update_time = time.time()  # For progress bar update throttling
    self.refresh_per_second = refresh_per_second
    self._current_epoch_steps = None
    self._epoch_start_step = 0
    self._step_start_time = None
    self._last_step_duration = None
    self.epoch_start_time = None
    self.train_start_time = None
    self.logged_batch_shapes = False

  # --- Callback Hooks ---

  def on_train_start(self, trainer=None):
    """Logs initial info panels and starts the progress display."""
    self.train_start_time = time.time()
    self.logged_batch_shapes = False  # Reset flag

    if trainer:
      self._print_parameter_table(trainer)
      self._print_model_summary(trainer)
      self._print_dataset_info(trainer)
      self._print_dataloader_info(trainer)
      self._print_training_setup(trainer)
      self._print_environment_info(trainer)
    else:
      print(
        '[yellow]Trainer object not available, cannot display start info.[/yellow]',
      )

    self.progress.start()
    self.last_log_time = time.time()
    self._epoch_start_step = trainer.num_steps if trainer else 0

  def on_epoch_start(self, epoch=None, trainer=None):
    """Adds or resets progress tasks for the new epoch."""
    self.epoch_start_time = time.time()
    try:
      self._current_epoch_steps = len(trainer.loader)
    except TypeError:
      self._current_epoch_steps = None

    self._epoch_start_step = trainer.num_steps
    run_total_epochs = trainer._epochs

    # Add/Update Epoch Task
    epoch_task_key = 'epoch'  # Use a consistent key for the main epoch bar
    if epoch_task_key not in self.tasks:
      self.tasks[epoch_task_key] = self.progress.add_task(
        '[bold cyan]Epochs',
        total=run_total_epochs,
      )
    else:
      if self.tasks[epoch_task_key] in self.progress.task_ids:
        self.progress.update(self.tasks[epoch_task_key], total=run_total_epochs)

    # Add/Reset Step Task
    step_task_key = 'step'  # Use a consistent key for the current step bar
    step_description = f'Epoch {epoch}'
    if (
      step_task_key not in self.tasks
      or self.tasks[step_task_key] not in self.progress.task_ids
    ):
      self.tasks[step_task_key] = self.progress.add_task(
        step_description,
        total=self._current_epoch_steps,
      )
    else:
      self.progress.reset(
        self.tasks[step_task_key],
        total=self._current_epoch_steps,
        description=step_description,
        visible=True,
      )

  def on_step_start(self, index=None, inputs=None, targets=None, trainer=None):
    """Record the start time of the step."""
    self._step_start_time = time.time()

  def on_step_end(
    self,
    index=None,
    inputs=None,
    targets=None,
    outputs=None,
    loss=None,
    trainer=None,
  ):
    """Updates progress bar, logs initial batch shapes, and logs periodic info."""
    self._log_batch_shapes_once(inputs, targets, outputs)
    self._calculate_step_duration()
    self._update_step_progress(trainer)
    self._log_periodic_info(trainer)

  def on_epoch_end(self, epoch=None, loss=None, metrics=None, trainer=None):
    """Completes progress tasks and logs epoch duration."""
    self._finalize_step_progress(epoch, trainer)
    self._log_epoch_duration(epoch, trainer)
    self._advance_epoch_progress(trainer)

  def on_validation_end(
    self,
    loss=None,
    outputs=None,
    targets=None,
    trainer=None,
  ):
    """Logs the validation metrics at the end of the validation phase."""
    if (
      trainer
      and hasattr(trainer, 'history')
      and hasattr(trainer.history, 'val_metrics')
    ):
      self._log_validation_results(trainer)

  def on_train_end(self, trainer=None):
    """Stops the progress display and logs final summary information."""
    if self.progress:
      self.progress.stop()
    self._log_final_summary(trainer)
    self.tasks = {}  # Reset tasks

  def log_checkpoint(self, path):
    """Logs a message indicating a checkpoint was saved.

    This method can be called by an external checkpointing callback.

    Args:
        path (str | Path): The path where the checkpoint was saved.
    """
    console = getattr(self.progress, 'console', print)
    log_message = f'Checkpoint saved: [green]{path}[/green]'
    console(log_message)

  # --- Helper Methods: Logging Start Info ---

  def _print_parameter_table(self, trainer):
    """Prints the training parameters table."""
    console = getattr(self.progress, 'console', None)
    print_func = console.print if console else print
    param_table = Table(
      title='[bold magenta]Training Parameters',
      show_header=True,
      header_style='bold blue',
    )
    param_table.add_column('Parameter', style='dim', width=30)
    param_table.add_column('Value')
    if hasattr(trainer, 'params') and trainer.params:
      for key, value in trainer.params.items():
        param_table.add_row(str(key), str(value))
    print_func(param_table)

  def _print_model_summary(self, trainer):
    """Prints the model summary panel."""
    console = getattr(self.progress, 'console', None)
    print_func = console.print if console else print
    if trainer.model:
      model_name = yann.get_model_name(trainer.model)
      total_params = yann.param_count(trainer.model)
      trainable_params = yann.param_count(
        yann.trainable(trainer.model.parameters()),
      )
      summary_text = (
        f'[bold]Model:[/bold] {model_name}\n'
        f'[bold]Total Params:[/bold] {total_params:,}\n'
        f'[bold]Trainable Params:[/bold] {trainable_params:,}'
      )
      model_panel = Panel(
        summary_text,
        title='[bold green]Model Summary',
        expand=False,
      )
      print_func(model_panel)
    else:
      print_func('[yellow]No model provided to trainer.[/yellow]')

  def _print_dataset_info(self, trainer):
    """Prints the dataset information panel."""
    console = getattr(self.progress, 'console', None)
    print_func = console.print if console else print
    dataset_info_base = ''
    train_ds_name = 'N/A'
    val_ds_name = 'N/A'
    if hasattr(trainer, 'dataset') and trainer.dataset:
      train_ds_name = yann.data.get_dataset_name(trainer.dataset)
      try:
        dataset_info_base += (
          f'\n[bold]Train Dataset:[/bold] {len(trainer.dataset):,} samples'
        )
      except TypeError:
        dataset_info_base += '\n[bold]Train Dataset:[/bold] Iterable'
    if hasattr(trainer, 'val_dataset') and trainer.val_dataset:
      val_ds_name = yann.data.get_dataset_name(trainer.val_dataset)
      try:
        dataset_info_base += (
          f'\n[bold]Val Dataset:[/bold] {len(trainer.val_dataset):,} samples'
        )
      except TypeError:
        dataset_info_base += '\n[bold]Val Dataset:[/bold] Iterable'
    if hasattr(trainer, 'classes') and trainer.classes:
      dataset_info_base += f'\n[bold]Classes:[/bold] {len(trainer.classes)}'

    if dataset_info_base:
      dataset_text = f'[bold]Train Dataset Name:[/bold] {train_ds_name}\n'
      dataset_text += f'[bold]Val Dataset Name:[/bold] {val_ds_name}\n'
      dataset_text += dataset_info_base.strip()
      dataset_panel = Panel(
        dataset_text,
        title='[bold purple]Dataset Info',
        expand=False,
      )
      print_func(dataset_panel)

  def _print_dataloader_info(self, trainer):
    """Prints the DataLoader settings panel."""
    console = getattr(self.progress, 'console', None)
    print_func = console.print if console else print
    loader_text = ''
    if hasattr(trainer, 'loader') and trainer.loader:
      loader_text += '[bold]Train Loader:[/bold]\n'
      loader_text += f'  Batch Size: {getattr(trainer.loader, "batch_size", "N/A")}\n'
      loader_text += f'  Num Workers: {getattr(trainer.loader, "num_workers", "N/A")}\n'
      loader_text += f'  Pin Memory: {getattr(trainer.loader, "pin_memory", "N/A")}\n'
      loader_text += (
        f'  Prefetch: {getattr(trainer.loader, "prefetch_factor", "N/A")}\n'
      )
      loader_text += f'  Persistent Workers: {getattr(trainer.loader, "persistent_workers", "N/A")}\n'
      if hasattr(trainer.loader, 'collate_fn') and trainer.loader.collate_fn:
        loader_text += f'  Collate Fn: {getattr(trainer.loader.collate_fn, "__name__", str(trainer.loader.collate_fn))}\n'
      if hasattr(trainer.loader, 'sampler') and trainer.loader.sampler:
        loader_text += f'  Sampler: {trainer.loader.sampler.__class__.__name__}\n'

    if hasattr(trainer, 'val_loader') and trainer.val_loader:
      loader_text += '\n[bold]Validation Loader:[/bold]\n'
      loader_text += (
        f'  Batch Size: {getattr(trainer.val_loader, "batch_size", "N/A")}\n'
      )
      loader_text += (
        f'  Num Workers: {getattr(trainer.val_loader, "num_workers", "N/A")}\n'
      )

    if loader_text:
      loader_panel = Panel(
        loader_text.strip(),
        title='[bold steel_blue]DataLoader Settings',
        expand=False,
      )
      print_func(loader_panel)

  def _print_training_setup(self, trainer):
    """Prints the training setup panel (Loss, Optimizer, AMP, Parallel)."""
    console = getattr(self.progress, 'console', None)
    print_func = console.print if console else print
    setup_text = ''
    if hasattr(trainer, 'loss') and trainer.loss:
      setup_text += f'[bold]Loss Fn:[/bold] {trainer.loss.__class__.__name__}\n'
    if hasattr(trainer, 'optimizer') and trainer.optimizer:
      setup_text += f'[bold]Optimizer:[/bold] {trainer.optimizer.__class__.__name__}\n'
    setup_text += f'[bold]AMP:[/bold] {getattr(trainer, "amp", "N/A")}\n'
    setup_text += f'[bold]Parallel:[/bold] {getattr(trainer, "parallel", "None")}\n'

    if setup_text:
      setup_panel = Panel(
        setup_text.strip(),
        title='[bold dark_orange]Training Setup',
        expand=False,
      )
      print_func(setup_panel)

  def _print_environment_info(self, trainer):
    """Prints the environment information panel (Git, Python, Torch, CUDA)."""
    console = getattr(self.progress, 'console', None)
    print_func = console.print if console else print
    git_hash = 'N/A'
    try:
      process = subprocess.run(
        ['git', 'rev-parse', '--short', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
        timeout=1,
      )
      git_hash = process.stdout.strip()
    except Exception:
      git_hash = 'N/A (git command failed or not in repo)'

    env_text = f'[bold]Git Commit:[/bold] {git_hash}\n'
    env_text += f'[bold]Python:[/bold] {sys.version.split()[0]}\n'
    env_text += f'[bold]PyTorch:[/bold] {torch.__version__}\n'
    if torch.cuda.is_available():
      env_text += f'[bold]CUDA:[/bold] {torch.version.cuda}\n'
      try:
        device_name = torch.cuda.get_device_name(
          trainer.device or torch.cuda.current_device(),
        )
        env_text += f'[bold]GPU:[/bold] {device_name}\n'
      except Exception:
        env_text += f'[bold]GPU:[/bold] Available (could not get name for device {trainer.device})\n'
    else:
      env_text += f'[bold]CUDA:[/bold] Not Available\n'
    env_panel = Panel(
      env_text.strip(),
      title='[bold yellow]Environment Info',
      expand=False,
    )
    print_func(env_panel)

  # --- Helper Methods: Step Processing ---

  def _log_batch_shapes_once(self, inputs, targets, outputs):
    """Logs the shapes of the first batch encountered, wrapped in panels."""
    if not self.logged_batch_shapes:
      console = getattr(self.progress, 'console', None)
      print_func = console.print if console else print
      try:
        print_func(
          Panel(describe(inputs), title='[cyan]inputs[/cyan]', expand=False),
        )
        print_func(
          Panel(describe(targets), title='[cyan]targets[/cyan]', expand=False),
        )
        print_func(
          Panel(describe(outputs), title='[cyan]outputs[/cyan]', expand=False),
        )
      except Exception as e:
        print_func(f'[yellow]Warning: Could not log batch shapes: {e}[/yellow]')
      self.logged_batch_shapes = True

  def _calculate_step_duration(self):
    """Calculates and stores the duration of the last step."""
    if self._step_start_time is not None:
      self._last_step_duration = time.time() - self._step_start_time
      self._step_start_time = None  # Reset for next step
    else:
      self._last_step_duration = None

  def _update_step_progress(self, trainer):
    """Updates the step progress bar description and completed count."""
    current_time = time.time()
    # Update description frequently, even if not logging periodically
    current_epoch_step = trainer.num_steps - self._epoch_start_step
    current_lr = 'N/A'
    if trainer and trainer.optimizer and trainer.optimizer.param_groups:
      try:
        current_lr = trainer.optimizer.param_groups[0]['lr']
      except (KeyError, IndexError):
        pass  # Keep N/A

    update_args = {
      'description': self._format_description(
        f'Epoch {trainer.num_epochs}',
        trainer.history.metrics,
        self._last_step_duration,
        current_lr,
      ),
    }

    if self._current_epoch_steps is not None:
      update_args['completed'] = current_epoch_step + 1
    else:
      update_args['advance'] = 1

    if 'step' in self.tasks and self.tasks['step'] in self.progress.task_ids:
      self.progress.update(self.tasks['step'], **update_args)

  def _log_periodic_info(self, trainer):
    """Constructs and prints a periodic log line if interval is met."""
    current_time = time.time()
    if current_time - self.last_log_time >= self.log_interval_seconds:
      log_items = []
      log_items.append(f'Step: {trainer.num_steps:<7}')
      log_items.append(f'Epoch: {trainer.num_epochs:<3}')
      # Metrics
      if trainer.history and trainer.history.metrics:
        metrics_log = []
        for name, vals in trainer.history.metrics.items():
          if vals:
            metrics_log.append(f'{name}: {vals[-1]:.4f}')
        if metrics_log:
          log_items.append('Metrics: (' + ' | '.join(metrics_log) + ')')
      # Optimizer Params
      if trainer.optimizer and trainer.optimizer.param_groups:
        group = trainer.optimizer.param_groups[0]
        opt_log = []
        for key in ('lr', 'momentum', 'weight_decay'):
          if key in group:
            opt_val = group[key]
            opt_log.append(
              f'{key}: {opt_val:.2e}'
              if isinstance(opt_val, float)
              else f'{key}: {opt_val}',
            )
        if opt_log:
          log_items.append('Opt: (' + ' | '.join(opt_log) + ')')
      # Step Time
      if self._last_step_duration is not None:
        log_items.append(f'Step Time: {self._last_step_duration:.3f}s')
      # GPU Memory
      if torch.cuda.is_available():
        try:
          mem_alloc_mb = torch.cuda.memory_allocated() / 1024**2
          mem_peak_mb = torch.cuda.max_memory_allocated() / 1024**2
          log_items.append(
            f'GPU Mem (MB): {mem_alloc_mb:.1f}/{mem_peak_mb:.1f}',
          )
        except Exception:
          log_items.append('GPU Mem: Error')

      console = getattr(self.progress, 'console', None)
      print_func = console.print if console else print
      print_func(' | '.join(log_items))
      self.last_log_time = current_time

  def _format_description(
    self,
    base_description,
    metrics,
    step_duration=None,
    current_lr=None,
  ):
    """Formats the description string for the step progress bar task."""
    desc = f'{base_description}'
    if metrics:
      metrics_str = ' | '.join(
        f'{name}: {vals[-1]:.4f}' for name, vals in metrics.items() if vals
      )
      if metrics_str:
        desc += f' ({metrics_str})'
    if step_duration is not None:
      desc += f' | Step: {step_duration:.3f}s'
    if current_lr is not None and current_lr != 'N/A':
      desc += f' | LR: {current_lr:.3e}'
    return desc

  # --- Helper Methods: Epoch/Validation End ---

  def _finalize_step_progress(self, epoch, trainer):
    """Marks the step progress bar as completed."""
    if 'step' in self.tasks:
      step_task_id = self.tasks['step']
      if step_task_id in self.progress.task_ids:
        step_task = self.progress.tasks[step_task_id]
        current_epoch_step = max(
          0,
          trainer.num_steps - self._epoch_start_step - 1,
        )
        self.progress.update(
          step_task_id,
          completed=step_task.total or current_epoch_step + 1,
          description=self._format_description(
            f'Epoch {epoch} (Completed)',
            trainer.history.metrics,
            self._last_step_duration,
            'N/A',
          ),
          visible=True,
        )

  def _log_epoch_duration(self, epoch, trainer):
    """Logs the duration of the completed epoch."""
    if self.epoch_start_time:
      epoch_duration = time.time() - self.epoch_start_time
      console = getattr(self.progress, 'console', None)
      print_func = console.print if console else print
      log_message = (
        f'Epoch {epoch} Duration: [bold cyan]{epoch_duration:.2f}s[/bold cyan]'
      )
      print_func(log_message)

  def _advance_epoch_progress(self, trainer):
    """Advances the main epoch progress bar."""
    if 'epoch' in self.tasks and self.tasks['epoch'] in self.progress.task_ids:
      self.progress.update(self.tasks['epoch'], advance=1)

  def _log_validation_results(self, trainer):
    """Logs validation results."""
    log_items = []
    log_items.append(f'Epoch: {trainer.num_epochs:<3}')
    log_items.append('[bold purple]Validation Results:[/bold purple]')
    val_metrics_log = []
    for name, vals in trainer.history.val_metrics.items():
      if vals:
        latest_val = vals[-1]
        val_metrics_log.append(f'{name}: {latest_val:.4f}')
    if val_metrics_log:
      log_items.append('(' + ' | '.join(val_metrics_log) + ')')
    console = getattr(self.progress, 'console', None)
    print_func = console.print if console else print
    log_message = ' | '.join(log_items)
    print_func(log_message)

  # --- Helper Methods: Train End ---

  def _log_final_summary(self, trainer):
    """Logs final summary information after training concludes."""
    console = getattr(self.progress, 'console', None)
    print_func = console.print if console else print
    print_func('-' * 30)  # Separator
    if self.train_start_time:
      total_duration = time.time() - self.train_start_time
      print_func(
        f'Total Training Time: [bold green]{total_duration:.2f}s[/bold green]',
      )
    if (
      trainer
      and hasattr(trainer, 'history')
      and hasattr(trainer.history, 'val_metrics')
      and trainer.history.val_metrics
    ):
      print_func('Best Validation Metrics:')
      try:
        if 'loss' in trainer.history.val_metrics:
          val_loss_data = trainer.history.val_metrics['loss']
          if (
            val_loss_data and isinstance(val_loss_data, list) and len(val_loss_data) > 0
          ):
            best_loss = min(val_loss_data)
            best_epoch = val_loss_data.index(best_loss)
            print_func(
              f'  Loss: [bold yellow]{best_loss:.4f}[/bold yellow] (Epoch {best_epoch})',
            )
        if 'accuracy' in trainer.history.val_metrics:
          val_acc_data = trainer.history.val_metrics['accuracy']
          if val_acc_data and isinstance(val_acc_data, list) and len(val_acc_data) > 0:
            best_acc = max(val_acc_data)
            best_acc_epoch = val_acc_data.index(best_acc)
            print_func(
              f'  Accuracy: [bold yellow]{best_acc:.4f}[/bold yellow] (Epoch {best_acc_epoch})',
            )
      except Exception as e:
        print_func(
          f'  [yellow]Could not determine best validation metrics: {e}[/yellow]',
        )
    print_func('-' * 30)

  # --- Serialization ---

  def __getstate__(self):
    """Ensure the Progress object itself is not pickled."""
    state = self.__dict__.copy()
    state['progress'] = None
    state['tasks'] = {}
    return state

  def __setstate__(self, state):
    """Reinitialize the Progress object after unpickling."""
    self.__dict__.update(state)
    self.progress = Progress(
      TextColumn('[progress.description]{task.description}'),
      BarColumn(),
      MofNCompleteColumn(),
      TextColumn('•'),
      TimeElapsedColumn(),
      TextColumn('•'),
      TimeRemainingColumn(),
      TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
      refresh_per_second=getattr(self, 'refresh_per_second', 10),
      disable=getattr(self, 'disable', False),
      transient=getattr(self, 'transient', False),
    )
    self.tasks = {}
    self.epoch_start_time = None
    self.train_start_time = None
    self.last_log_time = 0.0
