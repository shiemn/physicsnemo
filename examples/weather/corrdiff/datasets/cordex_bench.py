import torch.utils
import xarray as xr
import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, IterableDataset, DataLoader

from tqdm import tqdm
from scipy import ndimage
import time
import h5py

import datetime
import cftime

from datasets.base import ChannelMetadata, DownscalingDataset


class CordexBenchDataset(DownscalingDataset):
    """Dataset class for CordexBench climate downscaling benchmark data."""
    
    def __init__(self, data_path: str, stage: str = "train", 
                 years: list = None, domain: str = 'NZ', evaluation_type: str = 'PP', task: str = "pseudo-reality",
                 gcm: str = 'EC-Earth3', standardize: bool = True, 
                 stats_path: str = None,
                 input_variables: list = None, 
                 output_variables: list = None,
                 invariant_variables: list = None,
                 bounds: dict = None):
        """
        Initialize CordexBench dataset.
        
        Args:
            data_path: Path to CordexBench domain directory (e.g., /path/to/CordexBench/NZ)
            stage: Data split - 'train', 'val', or 'test'
            static_variables: Dictionary of static variables (lat, lon, orog)
            years: List of years to include (not used for CordexBench as splits are predefined)
            domain: Domain name ('NZ', 'ALPS', or 'SouthAfrica')
            evaluation_type: Type of evaluation ('PP', 'imperfect', or 'transferability')
            task: Task type ('pseudo-reality' or 'emulator_hist_future')
            gcm: GCM model name ('EC-Earth3', 'ACCESS-CM2', etc.)
        """
        self.data_path = data_path
        self.stage = stage
        self.domain = domain
        self.evaluation_type = evaluation_type
        self.task = task
        self.gcm = gcm
        self.static_variables = invariant_variables or ['Orog']
        self.years = None
        self.standardize = standardize
        self.normalize = standardize  # Alias for compatibility

        self.chunks = {'time': '1'}  # Chunking for dask

        # CordexBench predictor variables - actual variables from the dataset
        self.predictor_vars = ["u_850", "u_700", "u_500", "v_850", "v_700", "v_500",
                               "q_850", "q_700", "q_500", "t_850", "t_700", "t_500",
                               "z_850", "z_700", "z_500"]
        self.target_vars = output_variables # e.g., ['pr', 'tasmax']
        
        # Store for compatibility with NorwayDatasetH5 interface
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.invariant_variables = self.static_variables

        # Set up file paths based on CordexBench directory structure and evaluation type
        self._setup_data_paths()

        # Load static data
        self.static_variables = self.prepare_cordexbench_static_data()
        self.static_data = self.static_variables['orog']
        
        # Load and prepare data
        self._load_data()
        
        # Load or calculate normalization statistics
        self._setup_normalization()
    
    def _setup_data_paths(self):
        """Setup file paths based on CordexBench structure and evaluation type."""
        self.target_path = None
        self.predictor_path = None

        if self.task == "pseudo-reality":

            if self.stage == 'train' or self.stage == 'val':
                    self.predictor_path = os.path.join(self.data_path, "train", "ESD_pseudo_reality", "predictors")
                    self.target_path = os.path.join(self.data_path, "train", "ESD_pseudo_reality", "target")

                    self.years = list(range(1961, 1979)) if self.stage == 'train' else list(range(1979, 1980+1))

            
            if self.evaluation_type == 'PP':
                # Perfect Prognosis cross-validation
                if self.stage == 'predict':  # test - predictors from test, no test targets available
                    self.predictor_path = os.path.join(self.data_path, "test", "historical", "predictors", "perfect")
                elif self.stage == 'test':
                    raise NotImplementedError("Test stage not implemented for PP evaluation in pseudo-reality task. No test targets available.")
                    #self.target_path = os.path.join(self.data_path, "train", "ESD_pseudo_reality", "target")
            
            elif self.evaluation_type == 'imperfect':
                # Imperfect model prognosis
                if self.stage == 'predict':
                    self.predictor_path = os.path.join(self.data_path, "test", "historical", "predictors", "imperfect")
                elif self.stage == 'test':
                    raise NotImplementedError("Test stage not implemented for imperfect evaluation in pseudo-reality task. No test targets available.")
                    # self.target_path = os.path.join(self.data_path, "train", "ESD_pseudo_reality", "target")
            
            elif self.evaluation_type == 'transferability-mid':
                # Temporal transferability testing
                if self.stage == 'predict':
                    self.predictor_path = os.path.join(self.data_path, "test", "mid_century", "predictors", "perfect")
                elif self.stage == 'test':
                    raise NotImplementedError("Test stage not implemented for transferability-mid evaluation in pseudo-reality task. No test targets available.")
                    #self.target_path = os.path.join(self.data_path, "train", "ESD_pseudo_reality", "target")

            elif self.evaluation_type == 'transferability-end':
                # Temporal transferability testing
                if self.stage == 'predict':
                    self.predictor_path = os.path.join(self.data_path, "test", "end_century", "predictors", "perfect")
                elif self.stage == 'test':
                    raise NotImplementedError("Test stage not implemented for transferability-end evaluation in pseudo-reality task. No test targets available.")
                    #self.target_path = os.path.join(self.data_path, "train", "ESD_pseudo_reality", "target")

        elif self.task == "emulator":

            if self.stage == 'train' or self.stage == 'val':
                    self.predictor_path = os.path.join(self.data_path, "train", "Emulator_hist_future", "predictors")
                    self.target_path = os.path.join(self.data_path, "train", "Emulator_hist_future", "target")

                    self.years = list(range(1961, 1979)) + list(range(2081, 2099)) if self.stage == 'train' else list(range(1979, 1980+1)) + list(range(2099, 2100+1))


            if self.evaluation_type == 'PP':
                # Perfect Prognosis cross-validation
                if self.stage == 'predict':  # test - predictors from test, no test targets available
                    self.predictor_path = os.path.join(self.data_path, "test", "historical", "predictors", "perfect")
                elif self.stage == 'test':
                    raise NotImplementedError("Test stage not implemented for PP evaluation in pseudo-reality task. No test targets available.")
                    #self.target_path = os.path.join(self.data_path, "train", "ESD_pseudo_reality", "target")
            
            elif self.evaluation_type == 'imperfect':
                # Imperfect model prognosis
                if self.stage == 'predict':
                    self.predictor_path = os.path.join(self.data_path, "test", "historical", "predictors", "imperfect")
                elif self.stage == 'test':
                    raise NotImplementedError("Test stage not implemented for imperfect evaluation in pseudo-reality task. No test targets available.")
                    # self.target_path = os.path.join(self.data_path, "train", "ESD_pseudo_reality", "target")
            
            elif self.evaluation_type == 'transferability':
                raise NotImplementedError("Transferability not implemented for emulator task yet.")

            elif self.evaluation_type == 'transferability-hard-perfect':
                raise NotImplementedError("Transferability not implemented for emulator task yet.")
            
            elif self.evaluation_type == 'transferability-hard-imperfect':
                raise NotImplementedError("Transferability not implemented for emulator task yet.")
                   
        
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def _load_data(self):
        """Load predictor and target data from CordexBench NetCDF files."""
        self.targets = None
        self.predictors = None

        try:
            # Find predictor and target files
            predictor_files = self._find_netcdf_files(self.predictor_path, self.predictor_vars)
            if not predictor_files:
                raise FileNotFoundError(f"No predictor files found in {self.predictor_path}")
            
            # Load predictor data
            if len(predictor_files) == 1:
                self.predictors = xr.open_dataset(predictor_files[0], chunks=self.chunks)
            else:
                # For multiple files, try concat along time dimension
                try:
                    datasets = [xr.open_dataset(f, chunks=self.chunks) for f in predictor_files]
                    self.predictors = xr.concat(datasets, dim='time', combine_attrs='no_conflicts')
                except Exception:
                    # Fallback: use only the first file if concatenation fails
                    print(f"Warning: Could not concatenate multiple predictor files, using only first file")
                    self.predictors = xr.open_dataset(predictor_files[0], chunks=self.chunks)
            
            # Ensure we have the expected variables
            missing_predictors = set(self.predictor_vars) - set(self.predictors.data_vars)
            if missing_predictors:
                print(f"Warning: Missing predictor variables: {missing_predictors}")
                # Use only available variables
                self.predictor_vars = [var for var in self.predictor_vars if var in self.predictors.data_vars]

            pred_times_pd = pd.Series(self._convert_time_to_pandas(self.predictors.time))
            
            if self.target_path is not None:
                target_files = self._find_netcdf_files(self.target_path, self.target_vars)
                if not target_files:
                    raise FileNotFoundError(f"No target files found in {self.target_path}")
                
                # Load target data  
                if len(target_files) == 1:
                    self.targets = xr.open_dataset(target_files[0], chunks=self.chunks)
                else:
                    try:
                        datasets = [xr.open_dataset(f, chunks=self.chunks) for f in target_files]
                        self.targets = xr.concat(datasets, dim='time', combine_attrs='no_conflicts')
                    except Exception:
                        print(f"Warning: Could not concatenate multiple target files, using only first file")
                        self.targets = xr.open_dataset(target_files[0], chunks=self.chunks)

                
                # Check target variables
                missing_targets = set(self.target_vars) - set(self.targets.data_vars)
                if missing_targets:
                    print(f"Warning: Missing target variables: {missing_targets}")
                    # Use only available variables
                    self.target_vars = [var for var in self.target_vars if var in self.targets.data_vars]
                
                if not self.target_vars:
                    raise ValueError("No target variables found in target files")
                
                # Handle different coordinate systems (lat/lon vs x/y)
                self._handle_coordinate_systems()

                target_times_pd = pd.Series(self._convert_time_to_pandas(self.targets.time))
            
            # Extract coordinate information for static variables if not provided
            if self.static_variables.get('lat') is None or self.static_variables.get('lon') is None:
                # Try to get lat/lon from coordinates or data variables
                lat_coord = None
                lon_coord = None
                
                # Check predictors first
                if 'lat' in self.predictors.coords:
                    lat_coord = self.predictors.lat
                elif 'latitude' in self.predictors.coords:
                    lat_coord = self.predictors.latitude
                elif 'lat' in self.predictors.data_vars:
                    lat_coord = self.predictors.lat
                
                if 'lon' in self.predictors.coords:
                    lon_coord = self.predictors.lon
                elif 'longitude' in self.predictors.coords:
                    lon_coord = self.predictors.longitude
                elif 'lon' in self.predictors.data_vars:
                    lon_coord = self.predictors.lon
                
                
                self.static_variables['lat'] = lat_coord
                self.static_variables['lon'] = lon_coord
                
            
            if self.stage in ['train', 'val']:
                # Create boolean masks for time selection
                # Filter times by year membership in self.years (compare only year component)
                if self.years is None:
                    raise ValueError("For training/validation, years must be specified for CordexBench dataset")
                years_set = set(self.years)

                # Ensure we have pandas datetime and extract years
                pred_years = pd.to_datetime(pred_times_pd).dt.year
                target_years = pd.to_datetime(target_times_pd).dt.year

                # Boolean masks selecting time steps whose year is in the requested years
                pred_mask = pred_years.isin(years_set)
                target_mask = target_years.isin(years_set)

                # For reporting/counting use the predictor times selected
                common_times_pd = pred_times_pd[pred_mask.values]
                
                # Select overlapping times from both datasets using original time coordinates
                pred_common_times = self.predictors.time.values[pred_mask.values]
                target_common_times = self.targets.time.values[target_mask.values]
                
                self.predictors = self.predictors.sel(time=pred_common_times)
                self.targets = self.targets.sel(time=target_common_times)
                time_steps = len(common_times_pd)

                print(f"Loaded {time_steps} time steps for {self.stage} split")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load CordexBench data: {str(e)}")
    
    def _find_netcdf_files(self, path, variables=None):
        """Find NetCDF files in the given path, filtering by GCM and excluding static files."""
        if not os.path.exists(path):
            return []
        
        files = []
        available_gcms = set()
        
        # First pass: identify all available GCMs
        for file in os.listdir(path):
            if file.endswith('.nc') and not file.startswith('.'):
                # Exclude static files which have different structure
                if not any(static_keyword in file.lower() for static_keyword in ['static', 'topo', 'elevation', 'orog']):
                    # Extract GCM name from filename (common patterns in CordexBench)
                    if 'EC-Earth3' in file or 'EC_Earth' in file:
                        available_gcms.add('EC-Earth3')
                    elif 'ACCESS-CM2' in file or 'ACCESS_CM2' in file:
                        available_gcms.add('ACCESS-CM2')
                    elif 'CNRM' in file:
                        available_gcms.add('CNRM-CM6-1')
                    elif 'MPI' in file:
                        available_gcms.add('MPI-ESM1-2-HR')
        
        print(f"Available GCMs in {path}: {sorted(available_gcms)}")
        
        # Second pass: select files for the specified GCM
        selected_gcm = self.gcm
        gcm_files = []
        
        for file in os.listdir(path):
            if file.endswith('.nc') and not file.startswith('.'):
                # Exclude static files
                if not any(static_keyword in file.lower() for static_keyword in ['static', 'topo', 'elevation', 'orog']):
                    # Check if file belongs to selected GCM
                    file_matches_gcm = False
                    
                    if selected_gcm == 'EC-Earth3' and ('EC-Earth3' in file or 'EC_Earth' in file):
                        file_matches_gcm = True
                    elif selected_gcm == 'ACCESS-CM2' and ('ACCESS-CM2' in file or 'ACCESS_CM2' in file):
                        file_matches_gcm = True
                    elif selected_gcm == 'CNRM-CM6-1' and 'CNRM' in file:
                        file_matches_gcm = True
                    elif selected_gcm == 'MPI-ESM1-2-HR' and 'MPI' in file:
                        file_matches_gcm = True
                    
                    if file_matches_gcm:
                        gcm_files.append(os.path.join(path, file))
        
        # If no files found for selected GCM, try to use any available GCM
        if not gcm_files and available_gcms:
            print(f"Warning: No files found for GCM '{selected_gcm}'. Available GCMs: {sorted(available_gcms)}")
            fallback_gcm = sorted(available_gcms)[0]  # Use first available GCM alphabetically
            print(f"Using fallback GCM: {fallback_gcm}")
            
            for file in os.listdir(path):
                if file.endswith('.nc') and not file.startswith('.'):
                    if not any(static_keyword in file.lower() for static_keyword in ['static', 'topo', 'elevation', 'orog']):
                        # Check if file belongs to fallback GCM
                        file_matches_fallback = False
                        
                        if fallback_gcm == 'EC-Earth3' and ('EC-Earth3' in file or 'EC_Earth' in file):
                            file_matches_fallback = True
                        elif fallback_gcm == 'ACCESS-CM2' and ('ACCESS-CM2' in file or 'ACCESS_CM2' in file):
                            file_matches_fallback = True
                        elif fallback_gcm == 'CNRM-CM6-1' and 'CNRM' in file:
                            file_matches_fallback = True
                        elif fallback_gcm == 'MPI-ESM1-2-HR' and 'MPI' in file:
                            file_matches_fallback = True
                        
                        if file_matches_fallback:
                            gcm_files.append(os.path.join(path, file))
        
        print(f"Selected {len(gcm_files)} files for GCM '{selected_gcm}': {[os.path.basename(f) for f in gcm_files]}")
        return sorted(gcm_files)  # Sort for consistent ordering
    
    
    def _convert_time_to_pandas(self, time_coord):
        """Convert time coordinate to pandas datetime for consistent comparison."""
        import cftime
        
        # Check what type of time coordinate we have
        time_values = time_coord.values
        
        # Debug: print type information
        if len(time_values) > 0:
            first_val = time_values[0]
            
            # Check for various cftime types
            if hasattr(first_val, 'year') and hasattr(first_val, 'month') and hasattr(first_val, 'day'):
                # This is a cftime object (DatetimeNoLeap, etc.)
                try:
                    dates = []
                    for dt in time_values:
                        # Create pandas Timestamp from cftime components
                        dates.append(pd.Timestamp(year=dt.year, month=dt.month, day=dt.day))
                    return dates
                except Exception as e:
                    print(f"Error converting cftime: {e}")
                    # Fallback: convert to ISO format string
                    return pd.to_datetime([dt.strftime('%Y-%m-%d') for dt in time_values])
        
        # Handle numpy datetime64 or regular datetime
        try:
            return pd.to_datetime(time_values)
        except Exception as e:
            print(f"Error converting with pd.to_datetime: {e}")
            # Last resort: string conversion
            time_strings = [str(t)[:10] for t in time_values]  # Take only YYYY-MM-DD
            return pd.to_datetime(time_strings)
    
    def _setup_normalization(self):
        """Setup normalization statistics for predictors and targets."""
       
        if self.standardize:
            if os.path.exists(os.path.join(self.data_path, "predictor_mean_std.pt")):
                print(f"Found normalization coefficients. Loading them from {os.path.join(self.data_path, 'predictor_mean_std.pt')}.")
                self.predictor_mean, self.predictor_std = torch.load(os.path.join(self.data_path, "predictor_mean_std.pt"), weights_only=False)
                self.predictor_mean = self.predictor_mean.reshape(-1, 1, 1).numpy()
                self.predictor_std = self.predictor_std.reshape(-1, 1, 1).numpy()

            if os.path.exists(os.path.join(self.data_path, "target_mean_std.pt")):
                print(f"Found normalization coefficients. Loading them from {os.path.join(self.data_path, 'target_mean_std.pt')}.")
                self.target_mean, self.target_std = torch.load(os.path.join(self.data_path, "target_mean_std.pt"), weights_only=False)
                self.target_mean = self.target_mean.reshape(-1, 1, 1).numpy()
                self.target_std = self.target_std.reshape(-1, 1, 1).numpy()

                if self.target_vars != ['pr', 'tasmax']:
                    if self.target_vars == ['pr']:
                        self.target_mean = self.target_mean[0:1, :, :]
                        self.target_std = self.target_std[0:1, :, :]
                    elif self.target_vars == ['tasmax']:    
                        self.target_mean = self.target_mean[1:2, :, :]
                        self.target_std = self.target_std[1:2, :, :]            
    
    def __len__(self):
        return len(self.predictors.time)
    
    def _handle_coordinate_systems(self):
        """Handle different coordinate systems (lat/lon vs x/y) and ensure consistency."""
        # Check what spatial dimensions we have
        pred_spatial_dims = [dim for dim in self.predictors.dims if dim not in ['time', 'variable']]
        target_spatial_dims = [dim for dim in self.targets.dims if dim not in ['time', 'variable']]
        
        print(f"Predictor spatial dimensions: {pred_spatial_dims}")
        print(f"Target spatial dimensions: {target_spatial_dims}")
        
        # Store the actual spatial dimension names for later use
        self.spatial_dims = pred_spatial_dims  # Use predictor dims as reference
        
        # Check if we need to align coordinate systems between predictors and targets
        if set(pred_spatial_dims) != set(target_spatial_dims):
            print(f"Warning: Spatial dimensions differ between predictors {pred_spatial_dims} and targets {target_spatial_dims}")
            
            # If one uses lat/lon and other uses x/y, we might need to handle this
            # For now, just proceed - the data loading will handle dimension matching
            
        # Verify data can be properly indexed with available dimensions
        try:
            # Test indexing with first time step
            pred_sample = self.predictors[self.predictor_vars].isel(time=0)
            target_sample = self.targets[self.target_vars].isel(time=0)
            
            pred_array = pred_sample.to_array()
            target_array = target_sample.to_array();
            
            print(f"Successfully indexed data:")
            print(f"  Predictor array shape: {pred_array.shape}")
            print(f"  Target array shape: {target_array.shape}")
            
        except Exception as e:
            print(f"Warning: Could not test data indexing: {e}")
    
    def __getitem__(self, idx):
        """Get a single sample - return (target, predictor) to match NorwayDatasetH5 pattern."""
        # Get predictor data for this time step
        predictor_sample = self.predictors[self.predictor_vars].isel(time=idx)
        predictor_array = predictor_sample.to_array().values.astype(np.float32)  # Shape: (variables, spatial_dim1, spatial_dim2)
        
        # Get target data for this time step - multiple target variables
        if self.targets is not None:
            target_sample = self.targets[self.target_vars].isel(time=idx)
            target_array = target_sample.to_array().values.astype(np.float32)  # Shape: (target_variables, spatial_dim1, spatial_dim2)
        else:
            # For prediction mode, raise error for now
            raise NotImplementedError("Prediction-only mode (no targets) is not yet implemented")
        
        # Normalize data
        if self.normalize and hasattr(self, 'predictor_mean'):
            # Normalize predictors
            predictor_array = (predictor_array - self.predictor_mean) / self.predictor_std
            
        if self.normalize and hasattr(self, 'target_mean') and self.targets is not None:
            # Normalize targets
            target_array = (target_array - self.target_mean) / self.target_std

        # Upsample predictor data to match target resolution (following NorwayDatasetH5 pattern)
        predictor_array = self.upsample(predictor_array)

        # Concatenate static variables if available (following NorwayDatasetH5 pattern)
        if self.static_data is not None:
            predictor_array = np.concatenate((predictor_array, self.static_data), axis=0)

        # Return in NorwayDatasetH5 format: (target, predictor)
        return target_array, predictor_array

    def upsample(self, x: np.ndarray) -> np.ndarray:
        """
        Upsample the input data to the target resolution.
        Following NorwayDatasetH5 pattern with 8x upsampling.
        """
        return x.repeat(8, axis=1).repeat(8, axis=2)

    def input_channels(self):
        """Metadata for the input channels. A list of ChannelMetadata, one for each channel"""
        inputs = [ChannelMetadata(name=v) for v in self.input_variables]
        invariants = [
            ChannelMetadata(name=v, auxiliary=True) for v in self.invariant_variables
        ]
        return inputs + invariants

    def output_channels(self):
        """Return output channels metadata."""
        return [ChannelMetadata(name=n) for n in self.output_variables]
    
    def image_shape(self) -> tuple:
        """Get the (height, width) of the data (same for input and output)."""
        # Get shape from first predictor sample
        # sample_data = self.predictors[self.predictor_vars[0]].isel(time=0)
        # shape = sample_data.values.shape
        shape = 128,128
        return shape  # Return (height, width)

    def info(self):
        """Return normalization info to match NorwayDatasetH5 interface."""
        info_dict = {}
        if hasattr(self, 'predictor_mean'):
            info_dict["input_normalization"] = (
                self.predictor_mean.squeeze(),
                self.predictor_std.squeeze(),
            )
        if hasattr(self, 'target_mean'):
            info_dict["target_normalization"] = (
                self.target_mean.squeeze(),
                self.target_std.squeeze(),
            )
        return info_dict

    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from normalized data to physical units."""
        if hasattr(self, 'target_mean') and hasattr(self, 'target_std'):
            return (x * self.target_std) + self.target_mean
        else:
            raise RuntimeError("Normalization statistics not available - cannot denormalize output")

    def latitude(self):
        return self.lat
    
    def longitude(self):
        return self.lon

    def time(self):
        """Get time values from the dataset."""
        from physicsnemo.utils.diffusion import convert_datetime_to_cftime
        return [convert_datetime_to_cftime(t) for t in self.predictors.time.values]
    
    def prepare_cordexbench_static_data(self):
        """Prepare static variables for CordexBench dataset."""
        # Look for Static_fields.nc in the domain directory
        static_file = None
        
        # Check different possible locations for static files
        possible_static_paths = [
            os.path.join(self.data_path, "train", "ESD_pseudo_reality", "predictors", "Static_fields.nc"),
            os.path.join(self.data_path, "Static_fields.nc"),
            os.path.join(self.data_path, "train", "Static_fields.nc")
        ]
        
        for path in possible_static_paths:
            if os.path.exists(path):
                static_file = path
                break
        
        if static_file:
            print(f"Loading static fields from {static_file}")
            try:
                static = xr.open_dataset(static_file)
                
                # Extract orography if available
                orog = None
                if 'orog' in static.data_vars:
                    orog_data = static['orog']
                    # Normalize orography to 0-1 range
                    orog_norm = (orog_data - orog_data.min()) / (orog_data.max() - orog_data.min())
                    orog = torch.tensor(orog_norm.values, dtype=torch.float32)[None, :, :]
                
                # Extract lat/lon coordinates
                lat = static.lat if 'lat' in static.coords else None
                lon = static.lon if 'lon' in static.coords else None
                
                print(f"Static fields loaded - Orography: {'✓' if orog is not None else '✗'}, "
                      f"Lat: {'✓' if lat is not None else '✗'}, Lon: {'✓' if lon is not None else '✗'}")
                
                return {'orog': orog, 'lat':  lat, 'lon': lon}
                
            except Exception as e:
                print(f"Warning: Could not load static fields from {static_file}: {e}")
                return None, None, None
        else:
            print("No static fields file found, will extract coordinates from data files")
            return None, None, None

def convert_datetime_to_cftime(time: datetime.datetime, cls=cftime.DatetimeGregorian) -> cftime.DatetimeGregorian:
    """Convert a datetime object to a cftime object of the specified class."""
    # Handle numpy.datetime64 by converting to pandas Timestamp first
    if isinstance(time, np.datetime64):
        time = pd.Timestamp(time).to_pydatetime()
    elif hasattr(time, 'to_pydatetime'):
        # Handle pandas Timestamp
        time = time.to_pydatetime()

    return cls(time.year, time.month, time.day, time.hour, time.minute, time.second)


if __name__ == '__main__':

    data_path = "/Users/simon/Datasets/CordexBench"
    domains = ['NZ', 'ALPS']
    evaluation_types = ['PP', 'imperfect', 'transferability']
    stages = ['train', 'val', 'test']
    gcms = ['EC-Earth3', 'ACCESS-CM2', 'CNRM', 'MPI']  # Test multiple GCMs

    

    #Example usage:
    dataset = CordexBenchDataset(
        data_path="/Users/simon/Datasets/CordexBench",
        domain="ALPS", 
        evaluation_type="PP",
        gcm="MPI",
    )

    print(f"Dataset length: {len(dataset)}")
    print(dataset[0])



    # calculate_statistics(datamodule=data_module, data_path=data_module.data_path_domain)





