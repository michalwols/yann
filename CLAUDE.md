# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Yann (Yet Another Neural Network Library) is an extended version of torch.nn that adds powerful abstractions to make training models fast and easy. It's a PyTorch-based deep learning framework with automatic shape inference, comprehensive callbacks, and a registry system for component resolution.

## Commands

### Development
- **Format code**: `just format` (runs ruff check --fix and ruff format)
- **Run tests**: `uv run pytest` or `uv run pytest tests/path/to/specific_test.py::TestClass::test_method`
- **Install package**: `uv pip install -e .` for development install
- **CLI after install**: `yann` command becomes available globally

### Code Style
- Formatter: Ruff with 88 char lines, 2-space indentation, single quotes
- Always use trailing commas in multi-line structures (enforced by COM812)

## Architecture

### Core Components

1. **Training System** (`yann/train/`)
   - `Trainer` class orchestrates all training - the main entry point for model training
   - Functional utilities in `train/functional.py`
   - Training tracking in `train/track.py`

2. **Parameter System** (`yann/params.py`)
   - `HyperParams` base class for configuration
   - Automatically generates CLI arguments from class attributes
   - Use `Choice` and `Range` for constrained parameters
   - Call `Params.from_command()` to parse CLI args

3. **Module System** (`yann/modules/`)
   - `Stack`: Sequential container for layers (like nn.Sequential but more powerful)
   - `Infer`: Wrapper that automatically infers input dimensions for layers
   - Custom convolution modules: `MixConv`, `DepthwiseSeparableConv2d`
   - Shape manipulation: `Flatten`, `Reshape`, `Permute`, `Unsqueeze`

4. **Registry System** (`yann/config/registry.py`)
   - Components can be resolved by string names
   - Example: `optimizer = resolve('Adam', lr=0.001)`
   - Registry includes optimizers, schedulers, datasets, transforms, etc.

5. **Callback System** (`yann/callbacks/`)
   - Callbacks control training flow and monitoring
   - Key callbacks: History, Checkpoint, ProgressBar, WandbCallback, TensorBoardCallback
   - Rich-based progress bars in `rich_progress.py`

6. **Data Pipeline** (`yann/data/`)
   - Batch processing with custom collation
   - Storage backends: LMDB (`storage/lmdb.py`), Parquet (`storage/parquet.py`)
   - Search functionality: Annoy (`search/annoy.py`), inverted index

### Model Organization

- **Vision models** (`yann/models/vision/`): VGG, ViT, DeiT, FastViT, SFCNN
- **Base Model class** (`yann/models/model.py`): Common functionality for all models
- **Classifier** (`yann/models/classifier.py`): Standard classification model wrapper

### Typical Usage Pattern

```python
from yann import HyperParams, Trainer, Stack, Infer
import torch.nn as nn

class Params(HyperParams):
    dataset = 'MNIST'
    batch_size = 32
    optimizer = 'Adam'
    lr = 0.001

params = Params.from_command()

# Model with automatic shape inference
model = Stack(
    Infer(nn.Conv2d, 32, kernel_size=3),
    nn.ReLU(),
    Infer(nn.Linear, 10)
)

trainer = Trainer(
    model=model,
    optimizer=params.optimizer,
    callbacks=[History(), ProgressBar()]
)
trainer(epochs=10)
```

## Testing

- Test framework: pytest
- Test location: `/tests/` directory mirrors source structure
- Run specific test: `pytest tests/modules/conv/test_mixconv.py::TestMixConv::test_forward`
- Test files follow pattern: `test_*.py`

## Important Patterns

1. **Shape Inference**: Always wrap layers with `Infer()` when input dimensions are unknown
2. **Registry Resolution**: Use string names for components that get resolved via registry
3. **Callback-Driven**: Extend functionality through callbacks rather than modifying core training loop
4. **CLI Generation**: Define parameters as `HyperParams` subclass for automatic CLI
5. **Batch First**: All data operations assume batch dimension is first

## File Modifications

When modifying existing modules:
- Check imports to understand which frameworks/utilities are already in use
- Follow existing patterns in the file (especially for shape inference and registry usage)
- Maintain consistent indentation (2 spaces) and quote style (single quotes)
- Add trailing commas to multi-line structures

## Key Files to Understand

- `yann/__init__.py`: Main API exports and convenience functions
- `yann/params.py`: Parameter system implementation
- `yann/train/trainer.py`: Core training loop
- `yann/modules/__init__.py`: Module utilities and base classes
- `yann/config/registry.py`: Component resolution system