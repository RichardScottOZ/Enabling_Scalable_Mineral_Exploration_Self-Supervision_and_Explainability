# Data Directory

This directory should contain your geospatial raster data files.

## Required Files

### 1. Explanatory Raster (`explanatory.tif`)
Multi-band GeoTIFF containing geological/geophysical features.

**Format:**
- File type: GeoTIFF (.tif or .tiff)
- Bands: Multiple channels (e.g., magnetic, gravity, radiometric data)
- Data type: Float32 or Float64
- Coordinate system: Any projected CRS (e.g., UTM)

**Example bands:**
- Band 1: Total Magnetic Intensity (TMI)
- Band 2: Bouguer Gravity Anomaly
- Band 3: Radiometric Potassium (K)
- Band 4: Radiometric Thorium (Th)
- Band 5: Radiometric Uranium (U)
- Additional: Elevation, slope, distance to faults, etc.

### 2. Label Raster (`label.tif`)
Binary label raster indicating mineral presence/absence.

**Format:**
- File type: GeoTIFF (.tif or .tiff)
- Bands: Single band
- Data type: Integer or Float (will be converted to binary)
- Values: 
  - 0 or NoData: Unknown/No mineral occurrence
  - 1: Known mineral occurrence
- Coordinate system: Must match explanatory raster

## Data Preparation Guidelines

### Coordinate Reference System (CRS)
- Both rasters MUST have the same CRS
- Use a projected coordinate system (e.g., UTM) for accurate distance measurements
- If rasters have different CRS, reproject them using GDAL or QGIS

### Spatial Extent and Resolution
- Both rasters MUST have the same spatial extent
- Both rasters MUST have the same pixel resolution
- Dimensions should be divisible by the patch window size (default: 64)

### Data Quality
- Remove or interpolate NoData values in the explanatory raster
- Ensure label data is accurately georeferenced
- Consider removing edge effects or border artifacts

## Example: Creating Rasters with GDAL

### Check raster information:
```bash
gdalinfo explanatory.tif
gdalinfo label.tif
```

### Reproject to match CRS:
```bash
gdalwarp -t_srs EPSG:32650 input.tif output.tif
```

### Resample to match resolution:
```bash
gdalwarp -tr 100 100 input.tif output.tif
```

### Clip to same extent:
```bash
gdalwarp -te xmin ymin xmax ymax input.tif output.tif
```

## Testing Without Real Data

If you don't have real geospatial data yet, you can:
1. Run the demo script: `python demo.py`
2. The demo uses synthetic data to test the pipeline

## File Structure

```
data/
├── README.md              # This file
├── explanatory.tif        # Your multi-band feature raster
├── label.tif             # Your binary label raster
└── (optional additional files)
```

## Troubleshooting

**"FileNotFoundError: data/explanatory.tif not found"**
- Ensure your raster file is in the data/ directory
- Update the file path in config.yaml if using a different location

**"Rasters have different dimensions"**
- Use GDAL/QGIS to ensure both rasters have identical dimensions
- Check that both have the same number of rows and columns

**"No patches created"**
- Check that raster dimensions are large enough
- Ensure dimensions are divisible by patch_window_size
- Try reducing patch_window_size in config.yaml

## Additional Resources

- GDAL Documentation: https://gdal.org/
- QGIS (GUI tool): https://qgis.org/
- Rasterio Python library: https://rasterio.readthedocs.io/
