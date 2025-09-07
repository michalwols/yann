"""
Tests for HuggingFace model loading with hf:// URI scheme
"""

import pytest
import torch

import yann

# Check if transformers library is available
try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@pytest.mark.skipif(
    not HAS_TRANSFORMERS,
    reason="transformers library not installed"
)
class TestHuggingFaceModels:
    """Test HuggingFace model loading via hf:// URIs."""
    
    def test_basic_model_loading(self):
        """Test loading a basic HuggingFace model."""
        model = yann.resolve.model('hf://bert-base-uncased')
        
        assert model is not None
        assert hasattr(model, 'forward')
        assert isinstance(model, torch.nn.Module)
        
    def test_model_with_config(self):
        """Test loading a model with specific configuration."""
        model = yann.resolve.model('hf://distilbert-base-uncased')
        
        assert model is not None
        assert hasattr(model, 'forward')
        
    def test_model_forward_pass(self):
        """Test that loaded model can perform forward pass."""
        from transformers import AutoTokenizer
        
        model = yann.resolve.model('hf://distilbert-base-uncased')
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Test with simple input
        inputs = tokenizer("Hello, world!", return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        assert hasattr(outputs, 'last_hidden_state')
        assert outputs.last_hidden_state.shape[0] == 1  # batch size
        
    def test_model_with_additional_kwargs(self):
        """Test loading model with additional parameters."""
        # Load model with output_attentions=True
        model = yann.resolve.model('hf://distilbert-base-uncased', output_attentions=True)
        
        assert model is not None
        assert model.config.output_attentions is True
        
    def test_missing_transformers_library(self, monkeypatch):
        """Test error handling when transformers library is not installed."""
        # Patch the import in our ModelRegistry
        import yann.config.registry
        original_getitem = yann.config.registry.ModelRegistry.__getitem__
        
        def patched_getitem(self, item):
            if isinstance(item, str) and item.startswith('hf://'):
                raise ImportError(
                    "transformers library not installed. "
                    "Install with: pip install yann[transformers]"
                )
            return original_getitem(self, item)
        
        monkeypatch.setattr(yann.config.registry.ModelRegistry, '__getitem__', patched_getitem)
        
        with pytest.raises(ImportError) as exc_info:
            yann.resolve.model('hf://bert-base-uncased')
        
        assert "transformers library not installed" in str(exc_info.value)
        assert "pip install yann[transformers]" in str(exc_info.value)
    
    def test_invalid_model_name(self):
        """Test handling of invalid model names."""
        with pytest.raises(Exception):  # HuggingFace will raise an error
            yann.resolve.model('hf://this_model_does_not_exist')
    
    def test_model_in_trainer_params(self):
        """Test that hf:// URIs work with Trainer.Params."""
        from yann.train import Trainer
        
        class TestParams(Trainer.Params):
            model: str = 'hf://distilbert-base-uncased'
        
        params = TestParams()
        
        # This should resolve the model when trainer initializes
        model = yann.resolve.model(params.model)
        assert model is not None
        assert hasattr(model, 'forward')
    
    def test_different_model_types(self):
        """Test loading different types of HuggingFace models."""
        # Test a small model that should load quickly
        model = yann.resolve.model('hf://prajjwal1/bert-tiny')
        
        assert model is not None
        assert hasattr(model, 'forward')
        assert isinstance(model, torch.nn.Module)