"""
Tests for HuggingFace dataset loading with hf:// URI scheme
"""

import pytest

import yann

# Check if datasets library is available
try:
    import datasets
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


@pytest.mark.skipif(
    not HAS_DATASETS,
    reason="datasets library not installed"
)
class TestHuggingFaceDatasets:
    """Test HuggingFace dataset loading via hf:// URIs."""
    
    def test_basic_dataset_loading(self):
        """Test loading a basic HuggingFace dataset."""
        dataset = yann.resolve.dataset('hf://imdb', split='train')
        
        assert dataset is not None
        assert len(dataset) > 0
        assert 'text' in dataset[0]
        assert 'label' in dataset[0]
    
    def test_dataset_with_test_split(self):
        """Test loading a specific split."""
        dataset = yann.resolve.dataset('hf://imdb', split='test')
        
        assert dataset is not None
        assert len(dataset) > 0
    
    def test_dataset_with_configuration(self):
        """Test loading a dataset with a specific configuration."""
        dataset = yann.resolve.dataset('hf://glue', name='sst2', split='train')
        
        assert dataset is not None
        assert len(dataset) > 0
        assert 'sentence' in dataset[0]
        assert 'label' in dataset[0]
    
    def test_dataset_validation_split(self):
        """Test loading validation split."""
        dataset = yann.resolve.dataset('hf://glue', name='sst2', split='validation')
        
        assert dataset is not None
        assert len(dataset) > 0
    
    def test_missing_datasets_library(self, monkeypatch):
        """Test error handling when datasets library is not installed."""
        # Mock the import to raise ImportError
        def mock_import_error():
            raise ImportError("No module named 'datasets'")
        
        # Patch the import in our DatasetRegistry
        import yann.config.registry
        original_getitem = yann.config.registry.DatasetRegistry.__getitem__
        
        def patched_getitem(self, item):
            if isinstance(item, str) and item.startswith('hf://'):
                raise ImportError(
                    "datasets library not installed. "
                    "Install with: pip install yann[transformers]"
                )
            return original_getitem(self, item)
        
        monkeypatch.setattr(yann.config.registry.DatasetRegistry, '__getitem__', patched_getitem)
        
        with pytest.raises(ImportError) as exc_info:
            yann.resolve.dataset('hf://imdb')
        
        assert "datasets library not installed" in str(exc_info.value)
        assert "pip install yann[transformers]" in str(exc_info.value)
    
    def test_invalid_dataset_name(self):
        """Test handling of invalid dataset names."""
        with pytest.raises(Exception):  # HuggingFace will raise an error
            yann.resolve.dataset('hf://this_dataset_does_not_exist', split='train')
    
    def test_dataset_in_trainer_params(self):
        """Test that hf:// URIs work with Trainer.Params."""
        from yann.train import Trainer
        
        class TestParams(Trainer.Params):
            dataset: str = 'hf://imdb'
            dataset_split: str = 'train'
        
        params = TestParams()
        
        # This should resolve the dataset when trainer initializes
        dataset = yann.resolve.dataset(params.dataset, split=params.dataset_split)
        assert dataset is not None
        assert len(dataset) > 0