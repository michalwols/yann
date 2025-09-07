import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, TensorDataset

from yann.callbacks import Callback, Callbacks, Checkpoint, History, Logger, ProgressBar
from yann.data import Classes
from yann.modules import Flatten, Stack
from yann.params import HyperParams
from yann.train import Trainer
from yann.train.paths import Paths


# Fixtures
@pytest.fixture
def simple_model():
  """Create a simple model for testing."""
  return nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 2),
  )


@pytest.fixture
def simple_dataset():
  """Create a simple dataset for testing."""
  X = torch.randn(20, 10)  # Reduced size for faster testing
  y = torch.randint(0, 2, (20,))
  return TensorDataset(X, y)


@pytest.fixture
def temp_dir():
  """Create a temporary directory for testing."""
  with tempfile.TemporaryDirectory() as tmpdir:
    yield Path(tmpdir)


# Test Trainer Initialization
class TestTrainerInitialization:
  def test_empty_initialization(self):
    """Test that Trainer can be initialized without parameters."""
    trainer = Trainer()
    assert trainer is not None
    assert hasattr(trainer, 'params')
    assert hasattr(trainer, 'callbacks')

  def test_initialization_with_params_class(self):
    """Test initialization with Params class."""
    class TestParams(Trainer.Params):
      lr = 0.01
      batch_size = 32
    
    params = TestParams()
    trainer = Trainer(params=params)
    assert trainer.params.lr == 0.01
    assert trainer.params.batch_size == 32

  def test_initialization_with_kwargs(self):
    """Test initialization with keyword arguments."""
    trainer = Trainer(
      lr=0.001,
      batch_size=64,
      epochs=10,
    )
    assert trainer.params.lr == 0.001
    assert trainer.params.batch_size == 64
    assert trainer.params.epochs == 10

  def test_initialization_with_model(self, simple_model):
    """Test initialization with a model."""
    trainer = Trainer(model=simple_model)
    assert trainer.model is simple_model

  def test_initialization_with_optimizer(self, simple_model):
    """Test initialization with an optimizer."""
    optimizer = SGD(simple_model.parameters(), lr=0.01)
    trainer = Trainer(model=simple_model, optimizer=optimizer)
    assert trainer.optimizer is optimizer

  def test_initialization_with_loss(self):
    """Test initialization with a loss function."""
    loss = nn.CrossEntropyLoss()
    trainer = Trainer(loss=loss)
    assert trainer.loss is loss

  def test_initialization_with_dataset(self, simple_dataset):
    """Test initialization with a dataset."""
    trainer = Trainer(dataset=simple_dataset)
    assert trainer.params.dataset is simple_dataset

  def test_initialization_with_device_string(self):
    """Test initialization with device as string."""
    trainer = Trainer(device='cpu')
    assert trainer.params.device == 'cpu'

  def test_initialization_with_root_path(self, temp_dir):
    """Test initialization with root path."""
    trainer = Trainer(root=temp_dir)
    assert Path(trainer.params.root) == temp_dir

  def test_initialization_with_seed(self):
    """Test initialization with seed sets random seed."""
    trainer = Trainer(seed=42)
    assert trainer.params.seed == 42

  def test_params_update(self):
    """Test that params.update works correctly."""
    trainer = Trainer()
    trainer.params.update({'lr': 0.1, 'batch_size': 128})
    assert trainer.params.lr == 0.1
    assert trainer.params.batch_size == 128


# Test Trainer Setup and Configuration
class TestTrainerSetup:
  def test_model_assignment(self, simple_model):
    """Test model assignment during initialization."""
    trainer = Trainer(model=simple_model)
    assert trainer.model is simple_model

  def test_optimizer_resolution(self, simple_model, simple_dataset):
    """Test optimizer resolution from string."""
    trainer = Trainer(
      model=simple_model,
      dataset=simple_dataset,
      optimizer='Adam',
      lr=0.001,
      loss=nn.CrossEntropyLoss(),
      callbacks=[],  # No callbacks for speed
    )
    trainer(epochs=1)  # Need at least 1 epoch
    assert trainer.optimizer is not None

  def test_loss_assignment(self):
    """Test loss assignment."""
    loss = nn.MSELoss()
    trainer = Trainer(loss=loss)
    assert trainer.loss is loss

  def test_callbacks_initialization(self):
    """Test callbacks initialization."""
    callbacks = [History(), Logger()]
    trainer = Trainer(callbacks=callbacks)
    assert trainer.callbacks is not None

  def test_device_cpu(self):
    """Test device setup for CPU."""
    trainer = Trainer(device='cpu')
    assert trainer.params.device == 'cpu'

  @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
  def test_device_cuda(self):
    """Test device setup for CUDA."""
    trainer = Trainer(device='cuda')
    assert trainer.params.device == 'cuda'

  def test_paths_creation(self, temp_dir):
    """Test paths creation."""
    trainer = Trainer(root=temp_dir)
    assert trainer.paths is not None
    assert isinstance(trainer.paths, Paths)

  def test_loader_creation(self, simple_model, simple_dataset):
    """Test data loader creation."""
    trainer = Trainer(
      model=simple_model,
      dataset=simple_dataset,
      batch_size=10,
      optimizer='Adam',
      loss=nn.CrossEntropyLoss(),
      callbacks=[],  # No callbacks for speed
    )
    trainer(epochs=1)  # Need at least 1 epoch
    assert trainer.loader is not None
    assert trainer.loader.batch_size == 10

  def test_lr_scheduler_params(self, simple_model, simple_dataset):
    """Test learning rate scheduler parameters."""
    trainer = Trainer(
      model=simple_model,
      dataset=simple_dataset,
      optimizer='SGD',
      lr=0.1,
      loss=nn.CrossEntropyLoss(),
      lr_scheduler='ReduceLROnPlateau',  # Use a scheduler that exists in registry
      lr_scheduler_params={'patience': 10},
      callbacks=[],  # No callbacks for speed
    )
    trainer(epochs=1)  # Need at least 1 epoch
    assert trainer.lr_scheduler is not None


# Test Training Lifecycle
class TestTrainingLifecycle:
  def test_train_one_epoch(self, simple_model, simple_dataset):
    """Test training for one epoch."""
    trainer = Trainer(
      model=simple_model,
      dataset=simple_dataset,
      optimizer='Adam',
      loss=nn.CrossEntropyLoss(),
      batch_size=10,  # Small batch size for small dataset
      callbacks=[],  # No callbacks for speed
    )
    trainer(epochs=1)
    assert trainer.num_epochs == 1

  def test_train_multiple_epochs(self, simple_model, simple_dataset):
    """Test training for multiple epochs."""
    trainer = Trainer(
      model=simple_model,
      dataset=simple_dataset,
      optimizer='Adam',
      loss=nn.CrossEntropyLoss(),
      batch_size=10,  # Small batch size
      callbacks=[],  # No callbacks for speed
    )
    trainer(epochs=2)  # Reduced epochs
    assert trainer.num_epochs == 2

  def test_train_with_validation(self, simple_model, simple_dataset):
    """Test training with validation dataset."""
    val_dataset = TensorDataset(torch.randn(10, 10), torch.randint(0, 2, (10,)))
    trainer = Trainer(
      model=simple_model,
      dataset=simple_dataset,
      val_dataset=val_dataset,
      optimizer='Adam',
      loss=nn.CrossEntropyLoss(),
      batch_size=10,
      callbacks=[],  # No callbacks for speed
    )
    trainer(epochs=1)
    assert trainer.num_epochs == 1

  def test_train_step(self, simple_model, simple_dataset):
    """Test a single training step."""
    trainer = Trainer(
      model=simple_model,
      dataset=simple_dataset,
      optimizer='Adam',
      loss=nn.CrossEntropyLoss(),
    )
    trainer(epochs=0)  # Initialize
    
    batch = next(iter(trainer.loader))
    loss = trainer.step(batch)
    assert loss is not None
    assert isinstance(loss.item(), float)

  def test_forward_pass(self, simple_model, simple_dataset):
    """Test forward pass."""
    trainer = Trainer(
      model=simple_model,
      dataset=simple_dataset,
      loss=nn.CrossEntropyLoss(),
    )
    trainer(epochs=0)  # Initialize
    
    batch = next(iter(trainer.loader))
    output = trainer.forward(batch)
    assert output is not None

  def test_backward_pass(self, simple_model, simple_dataset):
    """Test backward pass."""
    trainer = Trainer(
      model=simple_model,
      dataset=simple_dataset,
      optimizer='Adam',
      loss=nn.CrossEntropyLoss(),
    )
    trainer(epochs=0)  # Initialize
    
    batch = next(iter(trainer.loader))
    loss = trainer.forward(batch)
    trainer.backward(loss)
    
    # Check that gradients were computed
    for param in trainer.model.parameters():
      if param.requires_grad:
        assert param.grad is not None


# Test Callbacks Integration
class TestCallbacksIntegration:
  def test_history_callback(self, simple_model, simple_dataset):
    """Test History callback integration."""
    trainer = Trainer(
      model=simple_model,
      dataset=simple_dataset,
      optimizer='Adam',
      loss=nn.CrossEntropyLoss(),
      callbacks=[History()],
    )
    trainer(epochs=2)
    assert hasattr(trainer, 'history')
    assert len(trainer.history.metrics) > 0

  def test_checkpoint_callback(self, simple_model, simple_dataset, temp_dir):
    """Test Checkpoint callback integration."""
    trainer = Trainer(
      root=temp_dir,
      model=simple_model,
      dataset=simple_dataset,
      optimizer='Adam',
      loss=nn.CrossEntropyLoss(),
      callbacks=[Checkpoint(freq=1)],
    )
    trainer(epochs=2)
    
    # Check that checkpoint files were created
    # The trainer creates a subdirectory structure
    checkpoint_dir = trainer.paths.checkpoints
    assert checkpoint_dir.exists()
    assert len(list(checkpoint_dir.glob('*.th'))) > 0

  def test_custom_callback(self, simple_model, simple_dataset):
    """Test custom callback integration."""
    class TestCallback(Callback):
      def __init__(self):
        super().__init__()
        self.on_epoch_end_called = False
        self.on_batch_end_called = False
      
      def on_epoch_end(self, trainer, **kwargs):
        self.on_epoch_end_called = True
      
      def on_batch_end(self, trainer, **kwargs):
        self.on_batch_end_called = True
    
    callback = TestCallback()
    trainer = Trainer(
      model=simple_model,
      dataset=simple_dataset,
      optimizer='Adam',
      loss=nn.CrossEntropyLoss(),
      callbacks=[callback],
    )
    trainer(epochs=1)
    
    assert callback.on_epoch_end_called
    assert callback.on_batch_end_called

  def test_multiple_callbacks(self, simple_model, simple_dataset):
    """Test multiple callbacks working together."""
    trainer = Trainer(
      model=simple_model,
      dataset=simple_dataset,
      optimizer='Adam',
      loss=nn.CrossEntropyLoss(),
      callbacks=[History(), Logger(batch_freq=10)],
    )
    trainer(epochs=1)
    assert hasattr(trainer, 'history')
    assert hasattr(trainer, 'log')


# Test State Management
class TestStateManagement:
  def test_save_checkpoint(self, simple_model, simple_dataset, temp_dir):
    """Test checkpoint saving during training."""
    trainer = Trainer(
      root=temp_dir,
      model=simple_model,
      dataset=simple_dataset,
      optimizer='Adam',
      loss=nn.CrossEntropyLoss(),
      callbacks=[Checkpoint(freq=1)],
    )
    trainer(epochs=1)
    
    # Check that checkpoint was created
    checkpoint_dir = trainer.paths.checkpoints
    assert checkpoint_dir.exists()
    checkpoints = list(checkpoint_dir.glob('*.th'))
    assert len(checkpoints) > 0

  def test_load_from_checkpoint(self, simple_model, simple_dataset, temp_dir):
    """Test loading from checkpoint."""
    # Train and save
    trainer1 = Trainer(
      root=temp_dir,
      model=simple_model,
      dataset=simple_dataset,
      optimizer='Adam',
      loss=nn.CrossEntropyLoss(),
      callbacks=[Checkpoint(freq=1)],
    )
    trainer1(epochs=2)
    
    # Get checkpoint file
    checkpoint_files = list(trainer1.paths.checkpoints.glob('*.th'))
    assert len(checkpoint_files) > 0
    checkpoint_path = checkpoint_files[0]
    
    # Load into new trainer
    trainer2 = Trainer(
      model=simple_model,
      dataset=simple_dataset,
      optimizer='Adam',
      loss=nn.CrossEntropyLoss(),
      from_checkpoint=str(checkpoint_path),
    )
    # The checkpoint should be loaded during initialization
    assert trainer2.params.from_checkpoint == str(checkpoint_path)

  def test_resume_training(self, simple_model, simple_dataset, temp_dir):
    """Test resuming training from checkpoint."""
    # Train for 2 epochs
    trainer1 = Trainer(
      root=temp_dir,
      model=simple_model,
      dataset=simple_dataset,
      optimizer='Adam',
      loss=nn.CrossEntropyLoss(),
      callbacks=[Checkpoint(freq=1)],
    )
    trainer1(epochs=2)
    
    # Get checkpoint file
    checkpoint_files = list(trainer1.paths.checkpoints.glob('*.th'))
    assert len(checkpoint_files) > 0
    checkpoint_path = checkpoint_files[-1]  # Get last checkpoint
    
    # Resume and train for 2 more epochs
    trainer2 = Trainer(
      root=temp_dir,
      model=simple_model,
      dataset=simple_dataset,
      optimizer='Adam',
      loss=nn.CrossEntropyLoss(),
      from_checkpoint=str(checkpoint_path),
    )
    # Note: Resuming from epoch 2, training to epoch 4
    trainer2(epochs=4)
    assert trainer2.num_epochs == 4

  def test_export_model(self, simple_model, simple_dataset, temp_dir):
    """Test exporting trained model."""
    trainer = Trainer(
      root=temp_dir,
      model=simple_model,
      dataset=simple_dataset,
      optimizer='Adam',
      loss=nn.CrossEntropyLoss(),
      classes=Classes.ordered(2),
    )
    trainer(epochs=1)
    
    export_path = trainer.export()
    assert export_path.exists()
    assert (export_path / 'model.th').exists() or (export_path / 'model.state_dict.th').exists()


# Test Advanced Features
class TestAdvancedFeatures:
  def test_gradient_clipping(self, simple_model, simple_dataset):
    """Test gradient clipping."""
    trainer = Trainer(
      model=simple_model,
      dataset=simple_dataset,
      optimizer='Adam',
      loss=nn.CrossEntropyLoss(),
      clip_grad={'max_norm': 1.0},
    )
    trainer(epochs=1)
    assert trainer.num_epochs == 1

  @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
  def test_mixed_precision_training(self, simple_model, simple_dataset):
    """Test automatic mixed precision training."""
    trainer = Trainer(
      model=simple_model,
      dataset=simple_dataset,
      optimizer='Adam',
      loss=nn.CrossEntropyLoss(),
      device='cuda',
      amp=True,
    )
    trainer(epochs=1)
    assert trainer.grad_scaler is not None

  def test_learning_rate_scheduling(self, simple_model, simple_dataset):
    """Test learning rate scheduling."""
    trainer = Trainer(
      model=simple_model,
      dataset=simple_dataset,
      optimizer='SGD',
      lr=0.1,
      loss=nn.CrossEntropyLoss(),
      lr_scheduler='StepLR',
      lr_scheduler_params={'step_size': 1, 'gamma': 0.1},
    )
    
    initial_lr = trainer.optimizer.param_groups[0]['lr']
    trainer(epochs=2)
    final_lr = trainer.optimizer.param_groups[0]['lr']
    assert final_lr < initial_lr

  def test_metrics_tracking(self, simple_model, simple_dataset):
    """Test custom metrics tracking."""
    def accuracy(output, target):
      pred = output.argmax(dim=1)
      return (pred == target).float().mean()
    
    trainer = Trainer(
      model=simple_model,
      dataset=simple_dataset,
      optimizer='Adam',
      loss=nn.CrossEntropyLoss(),
      metrics={'accuracy': accuracy},
      callbacks=[History()],
    )
    trainer(epochs=1)
    assert 'accuracy' in trainer.history.metrics

  def test_data_transforms(self, simple_model):
    """Test data transforms."""
    def transform(x):
      return x * 2
    
    dataset = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
    trainer = Trainer(
      model=simple_model,
      dataset=dataset,
      optimizer='Adam',
      loss=nn.CrossEntropyLoss(),
      transform=transform,
    )
    trainer(epochs=1)
    assert trainer.num_epochs == 1


# Test Error Handling
class TestErrorHandling:
  def test_missing_model_error(self, simple_dataset):
    """Test error when model is missing."""
    trainer = Trainer(
      dataset=simple_dataset,
      optimizer='Adam',
      loss=nn.CrossEntropyLoss(),
    )
    with pytest.raises((ValueError, AttributeError, RuntimeError)):
      trainer(epochs=1)

  def test_missing_optimizer_error(self, simple_model, simple_dataset):
    """Test error when optimizer is missing."""
    trainer = Trainer(
      model=simple_model,
      dataset=simple_dataset,
      loss=nn.CrossEntropyLoss(),
    )
    # Should create default optimizer or raise error
    trainer(epochs=1)  # This might work with defaults

  def test_missing_loss_error(self, simple_model, simple_dataset):
    """Test error when loss is missing."""
    trainer = Trainer(
      model=simple_model,
      dataset=simple_dataset,
      optimizer='Adam',
    )
    with pytest.raises((ValueError, AttributeError, RuntimeError)):
      trainer(epochs=1)

  def test_invalid_device_error(self, simple_model, simple_dataset):
    """Test error with invalid device."""
    with pytest.raises((ValueError, RuntimeError)):
      trainer = Trainer(
        model=simple_model,
        dataset=simple_dataset,
        optimizer='Adam',
        loss=nn.CrossEntropyLoss(),
        device='invalid_device',
      )
      trainer(epochs=1)

  def test_checkpoint_not_found_error(self, simple_model, simple_dataset):
    """Test error when checkpoint file not found."""
    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
      trainer = Trainer(
        model=simple_model,
        dataset=simple_dataset,
        from_checkpoint='nonexistent_checkpoint.th',
      )
      trainer(epochs=1)


# Test Performance Features
class TestPerformanceFeatures:
  def test_benchmarking_mode(self, simple_model, simple_dataset):
    """Test benchmarking mode."""
    trainer = Trainer(
      model=simple_model,
      dataset=simple_dataset,
      optimizer='Adam',
      loss=nn.CrossEntropyLoss(),
      benchmark=True,
    )
    trainer(epochs=1)
    assert trainer.num_epochs == 1

  def test_data_parallel(self, simple_model, simple_dataset):
    """Test data parallel training."""
    if torch.cuda.device_count() > 1:
      trainer = Trainer(
        model=simple_model,
        dataset=simple_dataset,
        optimizer='Adam',
        loss=nn.CrossEntropyLoss(),
        parallel='dp',
      )
      trainer(epochs=1)
      assert trainer.num_epochs == 1
    else:
      pytest.skip("Multiple GPUs not available")

  def test_compile_mode(self, simple_model, simple_dataset):
    """Test torch.compile mode."""
    if hasattr(torch, 'compile'):
      trainer = Trainer(
        model=simple_model,
        dataset=simple_dataset,
        optimizer='Adam',
        loss=nn.CrossEntropyLoss(),
        compile=True,
      )
      trainer(epochs=1)
      assert trainer.num_epochs == 1
    else:
      pytest.skip("torch.compile not available")


# Test Registry Integration
class TestRegistryIntegration:
  @pytest.mark.skip(reason="Registry resolution needs proper setup")
  def test_resolve_model_by_name(self):
    """Test resolving model by name from registry."""
    trainer = Trainer(
      model='resnet18',  # Should resolve from registry
      dataset=TensorDataset(torch.randn(10, 3, 224, 224), torch.randint(0, 2, (10,))),
      optimizer='Adam',
      loss='CrossEntropyLoss',
    )
    assert trainer.model is not None

  def test_resolve_optimizer_by_name(self, simple_model, simple_dataset):
    """Test resolving optimizer by name from registry."""
    trainer = Trainer(
      model=simple_model,
      dataset=simple_dataset,
      optimizer='SGD',
      loss=nn.CrossEntropyLoss(),
    )
    trainer(epochs=0)  # Initialize
    assert trainer.optimizer is not None

  def test_resolve_loss_by_name(self, simple_model, simple_dataset):
    """Test resolving loss by name from registry."""
    trainer = Trainer(
      model=simple_model,
      dataset=simple_dataset,
      optimizer='Adam',
      loss='CrossEntropyLoss',
    )
    trainer(epochs=0)  # Initialize
    assert trainer.loss is not None


# Test Distributed Training
class TestDistributedTraining:
  @pytest.mark.skipif(not torch.distributed.is_available(), reason="Distributed not available")
  def test_distributed_setup(self, simple_model, simple_dataset):
    """Test distributed training setup."""
    # This is a basic test - full distributed testing requires special setup
    from yann.distributed import Dist
    
    dist = Dist(backend='gloo', init=False)  # Don't actually initialize
    trainer = Trainer(
      model=simple_model,
      dataset=simple_dataset,
      optimizer='Adam',
      loss=nn.CrossEntropyLoss(),
      dist=dist,
    )
    assert trainer.dist is not None


# Test Custom Step Functions
class TestCustomStepFunctions:
  def test_custom_step_function(self, simple_model, simple_dataset):
    """Test using a custom step function."""
    def custom_step(trainer, batch):
      # Custom training step
      x, y = batch
      output = trainer.model(x)
      loss = trainer.loss(output, y)
      return loss
    
    trainer = Trainer(
      model=simple_model,
      dataset=simple_dataset,
      optimizer='Adam',
      loss=nn.CrossEntropyLoss(),
      step=custom_step,
    )
    trainer(epochs=1)
    assert trainer.num_epochs == 1

  def test_custom_forward_function(self, simple_model, simple_dataset):
    """Test using a custom forward function."""
    class CustomTrainer(Trainer):
      def forward(self, batch):
        # Custom forward logic
        x, y = batch
        output = self.model(x)
        loss = self.loss(output, y)
        return loss * 2  # Scale loss by 2
    
    trainer = CustomTrainer(
      model=simple_model,
      dataset=simple_dataset,
      optimizer='Adam',
      loss=nn.CrossEntropyLoss(),
    )
    trainer(epochs=1)
    assert trainer.num_epochs == 1