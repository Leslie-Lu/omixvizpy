import pytest
import pandas as pd
import tempfile
import os
from omixvizpy import plot_pca


class TestPCAPlotting:
    """Test cases for PCA plotting functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create sample eigenvec data
        self.sample_eigenvec_data = {
            'eid': [1, 2, 3, 4, 5],
            'PC1': [0.1, 0.2, 0.3, 0.4, 0.5],
            'PC2': [0.15, 0.25, 0.35, 0.45, 0.55],
            'PC3': [0.11, 0.21, 0.31, 0.41, 0.51],
            'PC4': [0.12, 0.22, 0.32, 0.42, 0.52],
            'PC5': [0.13, 0.23, 0.33, 0.43, 0.53]
        }
        
        # Create sample country data
        self.sample_country_data = {
            'eid': [1, 2, 3, 4, 5],
            'Country_of_birth': [1, 2, 3, 4, 1],
            'Ethnic_background': [1, 2, 3, 4, 5]
        }
        
        # Create temporary files
        self.temp_dir = tempfile.mkdtemp()
        self.eigenvec_file = os.path.join(self.temp_dir, 'test_eigenvec.txt')
        self.covar_file = os.path.join(self.temp_dir, 'test_country.txt')
        
        # Write test data to files
        pd.DataFrame(self.sample_eigenvec_data).to_csv(
            self.eigenvec_file, sep='\t', index=False
        )
        pd.DataFrame(self.sample_country_data).to_csv(
            self.covar_file, sep='\t', index=False
        )
    
    def test_plot_pca_basic(self):
        """Test basic functionality of plot_pca."""
        # This test checks if the function runs without errors
        # In a real test environment, you might want to check plot outputs
        try:
            plot_pca(
                eigenvec_file=self.eigenvec_file,
                covar_file=self.covar_file,
                cov1='Country_of_birth',
                cov2='Ethnic_background',
                legend_title_cov1='Country of Birth',
                legend_title_cov2='Ethnicity',
                cov1_levels=['England', 'Wales', 'Scotland', 'Others'],
                cov2_levels=['White', 'Mixed', 'Asian', 'Black', 'Chinese'],
                save_figs=False  # Don't save during testing
            )
            assert True  # If no exception, test passes
        except Exception as e:
            pytest.fail(f"plot_pca raised an exception: {e}")
    
    def test_file_loading(self):
        """Test if files are loaded correctly."""

        # Test eigenvec file loading
        eigenvec_df = pd.read_table(self.eigenvec_file, sep='\\s+', header=0)
        assert len(eigenvec_df) == 5
        assert 'eid' in eigenvec_df.columns
        assert 'PC1' in eigenvec_df.columns
        
        # Test country file loading
        country_df = pd.read_table(self.covar_file, sep='\\s+', header=0)
        assert len(country_df) == 5
        assert 'eid' in country_df.columns
        assert 'Country_of_birth' in country_df.columns
    
    def teardown_method(self):
        """Clean up after each test method."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


def test_import():
    """Test if the package can be imported successfully."""
    import omixvizpy
    assert hasattr(omixvizpy, 'plot_pca')
    assert omixvizpy.__version__


def test_version():
    """Test if version is defined."""
    import omixvizpy
    assert omixvizpy.__version__ == "0.1.0"
