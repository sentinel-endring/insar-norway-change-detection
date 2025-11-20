import pandas as pd
import json
from flask import Flask, render_template_string
import webbrowser
import threading
import time
import geopandas as gpd
from pathlib import Path
import warnings
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
import base64
from io import BytesIO
from PIL import Image
import tempfile
import os
warnings.filterwarnings('ignore')

# Flask app
app = Flask(__name__)

def load_and_filter_data(csv_file):
    """Load CSV and filter for significant changes only"""
    df = pd.read_csv(csv_file)
    
    # Filter for significant changes only
    significant_points = df[df['is_significant'] == True].copy()
    
    print(f"Total points: {len(df)}")
    print(f"Significant points: {len(significant_points)}")
    
    return significant_points

def load_geotiff_layer(tiff_path, target_crs='EPSG:3035', target_bounds=None):
    """Load GeoTIFF and convert to web-friendly format with coordinate transformation"""
    try:
        print(f"🗺️  Loading GeoTIFF from: {tiff_path}")
        
        # Check if file exists
        if not Path(tiff_path).exists():
            print(f"❌ GeoTIFF file not found: {tiff_path}")
            return None
        
        with rasterio.open(tiff_path) as src:
            print(f"📊 Original GeoTIFF info:")
            print(f"   - CRS: {src.crs}")
            print(f"   - Shape: {src.shape}")
            print(f"   - Bounds: {src.bounds}")
            print(f"   - Transform: {src.transform}")
            print(f"   - Data type: {src.dtypes[0]}")
            
            # Get source CRS
            src_crs = src.crs
            if src_crs is None:
                print("⚠️  No CRS found in GeoTIFF, assuming EPSG:32632")
                src_crs = 'EPSG:32632'
            
            # Read the full data first
            data = src.read(1)
            original_transform = src.transform
            original_bounds = src.bounds
            
            print(f"📍 Original bounds in {src_crs}: {original_bounds}")
            
            # Handle different data types and create binary mask for visualization
            if data.dtype == bool:
                binary_data = data.astype(np.uint8)
            else:
                # For non-boolean data, create binary mask (non-zero = 1, zero = 0)
                binary_data = (data != 0).astype(np.uint8)
            
            print(f"📈 Data statistics:")
            print(f"   - Unique values: {np.unique(binary_data)}")
            print(f"   - True pixels: {np.sum(binary_data)}")
            print(f"   - Total pixels: {binary_data.size}")
            
            # Skip transformation if already in target CRS
            if str(src_crs).upper() == target_crs.upper():
                print(f"✅ Already in target CRS {target_crs}")
                transformed_data = binary_data
                dst_transform = original_transform
                dst_bounds = original_bounds
            else:
                print(f"🔄 Transforming from {src_crs} to {target_crs}")
                
                # Calculate transform for reprojection - use original bounds
                dst_transform, width, height = calculate_default_transform(
                    src_crs, target_crs, 
                    binary_data.shape[1], binary_data.shape[0], 
                    *original_bounds
                )
                
                print(f"🎯 Calculated destination:")
                print(f"   - Width: {width}, Height: {height}")
                print(f"   - Transform: {dst_transform}")
                
                # Create destination array
                transformed_data = np.zeros((height, width), dtype=np.uint8)
                
                # Reproject
                reproject(
                    source=binary_data,
                    destination=transformed_data,
                    src_transform=original_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                )
                
                # Calculate bounds in target CRS using the transform
                dst_bounds = rasterio.transform.array_bounds(height, width, dst_transform)
                
                print(f"🎯 Reprojected bounds in {target_crs}: {dst_bounds}")
                
                # Alternative: also calculate bounds using transform_bounds for verification
                from rasterio.warp import transform_bounds
                verify_bounds = transform_bounds(src_crs, target_crs, *original_bounds)
                print(f"🔍 Verification bounds (transform_bounds): {verify_bounds}")
                
                # If there's a significant difference, use the transform_bounds result
                bounds_diff = max(abs(dst_bounds[i] - verify_bounds[i]) for i in range(4))
                if bounds_diff > 100:  # If difference > 100m, use verification bounds
                    print(f"⚠️  Large bounds difference ({bounds_diff:.1f}m), using transform_bounds result")
                    dst_bounds = verify_bounds
            
            print(f"🎯 Final bounds in {target_crs}: {dst_bounds}")
            
            # Convert to RGBA image for web display
            # Create colored image: transparent for 0, colored for 1
            rgba_image = np.zeros((transformed_data.shape[0], transformed_data.shape[1], 4), dtype=np.uint8)
            
            # Set color for positive values (orange/red)
            mask = transformed_data > 0
            rgba_image[mask] = [255, 165, 0, 180]  # Orange with transparency
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgba_image, 'RGBA')
            
            # Convert to base64 for web transmission
            buffer = BytesIO()
            pil_image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            result = {
                'image_data': f"data:image/png;base64,{img_str}",
                'bounds': [float(x) for x in dst_bounds],  # Convert to regular Python floats
                'crs': target_crs,
                'shape': [int(x) for x in transformed_data.shape],  # Convert to regular Python ints
                'pixel_count': int(np.sum(transformed_data)),  # Convert to regular Python int
                'total_pixels': int(transformed_data.size)  # Convert to regular Python int
            }
            
            print(f"✅ Successfully processed GeoTIFF: {result['shape']} pixels, {result['pixel_count']} positive pixels")
            return result
            
    except Exception as e:
        print(f"❌ Error loading GeoTIFF {tiff_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_historical_fgdb(fgdb_path, layer_name='fkb_tiltak_omrade', bbox_3035=None):
    """Load historical FGDB data within bounding box and convert to GeoJSON"""
    try:
        print(f"🗂️  Loading historical data from: {fgdb_path}")
        
        # Check if file exists
        if not Path(fgdb_path).exists():
            print(f"❌ FGDB file not found: {fgdb_path}")
            return None
        
        # If we have a bounding box, convert it to the FGDB's coordinate system for spatial filtering
        spatial_filter = None
        if bbox_3035:
            from shapely.geometry import box
            # Create a bounding box geometry in EPSG:3035
            bbox_geom_3035 = box(bbox_3035[0], bbox_3035[1], bbox_3035[2], bbox_3035[3])
            
            # Convert bbox to UTM 25833 (the FGDB's native CRS) for efficient spatial filtering
            import geopandas as gpd
            bbox_gdf = gpd.GeoDataFrame([1], geometry=[bbox_geom_3035], crs='EPSG:3035')
            bbox_gdf_utm = bbox_gdf.to_crs('EPSG:25833')
            spatial_filter = bbox_gdf_utm.geometry.iloc[0]
            
            print(f"📦 Spatial filter (EPSG:25833): {spatial_filter.bounds}")
        
        # Read the specific layer from FGDB with spatial filter
        if spatial_filter:
            gdf = gpd.read_file(fgdb_path, layer=layer_name, mask=spatial_filter)
            print(f"🎯 Spatial filtering applied - loaded {len(gdf)} features (instead of full dataset)")
        else:
            gdf = gpd.read_file(fgdb_path, layer=layer_name)
            print(f"⚠️  No spatial filter - loaded {len(gdf)} features")
        
        if len(gdf) == 0:
            print(f"📭 No features found in {layer_name} within the specified area")
            return []
        
        # Convert to EPSG:3035 if not already
        if gdf.crs and gdf.crs.to_string() != 'EPSG:3035':
            print(f"📍 Converting {len(gdf)} features from {gdf.crs} to EPSG:3035")
            gdf = gdf.to_crs('EPSG:3035')
        
        # Additional filtering in EPSG:3035 to ensure we only get features that intersect our AOI
        if bbox_3035:
            bbox_geom_3035 = box(bbox_3035[0], bbox_3035[1], bbox_3035[2], bbox_3035[3])
            # Filter features that intersect with our bounding box
            gdf = gdf[gdf.geometry.intersects(bbox_geom_3035)]
            print(f"🔍 After EPSG:3035 filtering: {len(gdf)} features remain")
        
        # Convert to GeoJSON format
        geojson_features = []
        for idx, row in gdf.iterrows():
            try:
                # Get geometry
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                
                # Convert geometry to GeoJSON format
                if hasattr(geom, '__geo_interface__'):
                    geom_dict = geom.__geo_interface__
                else:
                    # Fallback for older versions
                    from shapely.geometry import mapping
                    geom_dict = mapping(geom)
                
                # Create feature with properties (limit to key properties to reduce size)
                key_properties = {}
                for col in gdf.columns:
                    if col != 'geometry':
                        val = row[col]
                        if pd.notna(val):
                            key_properties[col] = str(val)
                        else:
                            key_properties[col] = ''
                
                feature = {
                    "type": "Feature",
                    "geometry": geom_dict,
                    "properties": key_properties
                }
                geojson_features.append(feature)
                
            except Exception as e:
                print(f"⚠️  Skipping feature {idx}: {e}")
                continue
        
        print(f"✅ Successfully processed {len(geojson_features)} features from {layer_name}")
        return geojson_features
        
    except Exception as e:
        print(f"❌ Error loading FGDB {fgdb_path}: {e}")
        return None

def create_geojson_features(df):
    """Convert dataframe to GeoJSON features for OpenLayers"""
    features = []
    
    for idx, row in df.iterrows():
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row['easting']), float(row['northing'])]  # X=easting, Y=northing
            },
            "properties": {
                "coherence": float(row['coherence']) if pd.notna(row['coherence']) else 0,
                "change_magnitude": float(row['change_magnitude']) if pd.notna(row['change_magnitude']) else 0,
                "track": str(row['track']),
                "confidence": float(row['confidence']) if pd.notna(row['confidence']) else 0,
                "trend_slope": float(row['trend_slope']) if pd.notna(row['trend_slope']) else 0,
                "change_date": str(row['change_date']) if pd.notna(row['change_date']) else 'Unknown'
            }
        }
        features.append(feature)
    
    print(f"Sample coordinates from CSV: easting={row['easting']}, northing={row['northing']}")
    print(f"Converted to GeoJSON: [{float(row['easting'])}, {float(row['northing'])}]")
    
    return features

@app.route('/')
def map_view():
    # Load and process data
    significant_points = load_and_filter_data('change_detection_results.csv')
    
    if len(significant_points) == 0:
        return "<h1>No significant changes found in the data!</h1>"
    
    # Create GeoJSON features for points
    features = create_geojson_features(significant_points)
    
    # Calculate extent with buffer for initial view AND spatial filtering
    min_easting = significant_points['easting'].min()
    max_easting = significant_points['easting'].max()
    min_northing = significant_points['northing'].min()
    max_northing = significant_points['northing'].max()
    
    # Calculate extent dimensions and add buffer (10% of range or minimum 1000m)
    easting_range = max_easting - min_easting
    northing_range = max_northing - min_northing
    buffer_easting = max(easting_range * 0.1, 1000)  # 10% buffer or 1km minimum
    buffer_northing = max(northing_range * 0.1, 1000)
    
    # Buffered extent for spatial filtering (add extra buffer for FGDB loading)
    fgdb_buffer_multiplier = 10.0  # Extra buffer for FGDB spatial filtering
    fgdb_extent = [
        min_easting - (buffer_easting * fgdb_buffer_multiplier),
        min_northing - (buffer_northing * fgdb_buffer_multiplier), 
        max_easting + (buffer_easting * fgdb_buffer_multiplier),
        max_northing + (buffer_northing * fgdb_buffer_multiplier)
    ]
    
    # Buffered extent for map display
    extent_min_easting = min_easting - buffer_easting
    extent_max_easting = max_easting + buffer_easting
    extent_min_northing = min_northing - buffer_northing
    extent_max_northing = max_northing + buffer_northing
    
    # Load GeoTIFF layer
    geotiff_data = None
    geotiff_path = "binary_detection.tif"
    if Path(geotiff_path).exists():
        print(f"🗺️  Loading Senbygg GeoTIFF layer...")
        geotiff_data = load_geotiff_layer(geotiff_path, target_crs='EPSG:3035', target_bounds=fgdb_extent)
        if geotiff_data:
            print(f"✅ Senbygg layer loaded: {geotiff_data['pixel_count']} positive pixels")
        else:
            print(f"❌ Failed to load Senbygg layer")
    else:
        print(f"⚠️  Senbygg GeoTIFF file not found: {geotiff_path}")
    
    # Load historical FGDB data with spatial filtering
    historical_paths = [
        "./Basisdata_0000_Norge_5973_FKB-Tiltak2023_FGDB/Basisdata_0000_Norge_5973_FKB-Tiltak_FGDB.gdb",
        "./Basisdata_0000_Norge_5973_FKB-Tiltak2024_FGDB/Basisdata_0000_Norge_5973_FKB-Tiltak_FGDB.gdb"
    ]
    
    historical_data = {}
    for i, path in enumerate(historical_paths):
        year = 2023 + i  # 2023, 2024
        print(f"🎯 Loading {year} FGDB with spatial filter: {fgdb_extent}")
        data = load_historical_fgdb(path, bbox_3035=fgdb_extent)
        if data:
            historical_data[f'tiltak_{year}'] = data
            print(f"📅 Historical data for {year}: {len(data)} features (spatially filtered)")
        else:
            print(f"❌ No data loaded for {year}")
    
    # Center point
    center_easting = (min_easting + max_easting) / 2
    center_northing = (min_northing + max_northing) / 2
    
    # Debug output
    print(f"Data extent:")
    print(f"  Easting: {min_easting:.1f} to {max_easting:.1f} (range: {easting_range:.1f}m)")
    print(f"  Northing: {min_northing:.1f} to {max_northing:.1f} (range: {northing_range:.1f}m)")
    print(f"  Center: {center_easting:.1f}, {center_northing:.1f}")
    print(f"  Buffer: {buffer_easting:.1f}m E/W, {buffer_northing:.1f}m N/S")

    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Change Detection Results - EPSG:3035 with Historical Data + Senbygg</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/ol@v7.5.2/ol.css" type="text/css">
    <script src="https://cdn.jsdelivr.net/npm/ol@v7.5.2/dist/ol.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/proj4@2.9.0/dist/proj4.js"></script>
    <style>
        body { margin: 0; font-family: Arial, sans-serif; }
        #map { 
            width: 100%; 
            height: 100vh; 
            background-color: #f0f8ff; /* Light blue background instead of blank */
        }
        .info-box {
            position: absolute;
            top: 10px;
            left: 10px;  /* CHANGE: from 'right: 10px;' to 'left: 10px;' */
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            max-width: 300px;
            z-index: 1000;
        }
        .popup {
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            max-width: 250px;
        }
        .popup h4 { margin-top: 0; color: #d73027; }
        .popup p { margin: 5px 0; font-size: 12px; }
        .debug-info {
            position: absolute;
            bottom: 10px;
            right: 10px;  /* CHANGE: from 'left: 10px;' to 'right: 10px;' */
            background: rgba(255,255,255,0.9);
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
            z-index: 1000;
        }
        .control-buttons {
            position: absolute;
            top: 10px;     /* CHANGE: from 'top: 140px;' to 'top: 10px;' */
            right: 10px;
            z-index: 1000;
        }
        .control-buttons button {
            display: block;
            margin-bottom: 5px;
            padding: 8px 12px;
            background: #007cba;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
            min-width: 140px;
        }
        .control-buttons button:hover {
            background: #005a87;
        }
        .control-buttons button.historical {
            background: #8e44ad;
        }
        .control-buttons button.historical:hover {
            background: #732d91;
        }
        .control-buttons button.geotiff {
            background: #e67e22;
        }
        .control-buttons button.geotiff:hover {
            background: #d35400;
        }
        .legend {
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(255,255,255,0.95);
            padding: 15px;
            border-radius: 5px;
            font-size: 12px;
            z-index: 1000;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            /* margin-bottom: 140px;  Move above debug info */
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }
        .legend-color {
            width: 20px;
            height: 15px;
            margin-right: 8px;
            border: 1px solid #ccc;
        }
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="info-box">
        <h3>🗺️ Change Detection Map</h3>
        <p><strong>Mode:</strong> Points + WMS + Historical + Senbygg</p>
        <p><strong>Coordinate System:</strong> EPSG:3035</p>
        <p><strong>Data Source:</strong> change_detection_results.csv</p>
        <p><strong>Historical:</strong> {{ historical_count }} datasets</p>
        <p><strong>Senbygg:</strong> {% if geotiff_data %}{{ geotiff_data.pixel_count }} pixels{% else %}Not loaded{% endif %}</p>
        <hr>
        <p><small>Click on red points for details</small></p>
        <p><small>Use mouse wheel to zoom, drag to pan</small></p>
    </div>
    
    <div class="control-buttons">
        <button onclick="zoomToPoints()">🔍 Zoom to Points</button>
        <button onclick="togglePointSize()">📍 Toggle Size (8/15px)</button>
        <button onclick="toggleLayer('osm')">🗺️ Toggle OSM</button>
        <button onclick="toggleLayer('fkb')">🏗️ Toggle WMS Tiltak</button>
        <button class="geotiff" onclick="toggleLayer('senbygg')">🏢 Toggle Senbygg</button>
        <button class="historical" onclick="toggleLayer('tiltak_2023')">📅 Toggle 2023 FGDB</button>
        <button class="historical" onclick="toggleLayer('tiltak_2024')">📅 Toggle 2024 FGDB</button>
    </div>
    
    <div class="legend">
        <h4 style="margin-top: 0;">Legend</h4>
        <div class="legend-item">
            <div class="legend-color" style="background-color: red;"></div>
            <span>Change Detection Points</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: rgba(255, 182, 193, 0.7);"></div>
            <span>WMS Tiltak (Current)</span>
        </div>
        {% if geotiff_data %}
        <div class="legend-item">
            <div class="legend-color" style="background-color: rgba(255, 165, 0, 0.7);"></div>
            <span>Senbygg (GeoTIFF)</span>
        </div>
        {% endif %}
        <div class="legend-item">
            <div class="legend-color" style="background-color: rgba(138, 43, 226, 0.5);"></div>
            <span>2023 Historical FGDB</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: rgba(75, 0, 130, 0.5);"></div>
            <span>2024 Historical FGDB</span>
        </div>
    </div>
    
    <div class="debug-info" id="debug-info">
        <strong>Debug Info:</strong><br>
        Loading...
    </div>

    <script>
        // Define EPSG:3035 projection
        proj4.defs('EPSG:3035', '+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs');
        
        // Define EPSG:25833 projection (UTM Zone 33N for Norway)
        proj4.defs('EPSG:25833', '+proj=utm +zone=33 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs');
        
        ol.proj.proj4.register(proj4);
        
        // Get projections
        const projection3035 = ol.proj.get('EPSG:3035');
        const projection25833 = ol.proj.get('EPSG:25833');
        
        // Set extent for EPSG:3035 (covering Europe)
        projection3035.setExtent([2000000, 1000000, 7000000, 5500000]);
        
        // Set extent for EPSG:25833 (UTM Zone 33N - Norway)
        projection25833.setExtent([-2000000, 3500000, 3500000, 9000000]);

        // Create EPSG:3035 compatible WMS layers
        
        // Norwegian FKB WMS - "Tiltak" layer (uses EPSG:25833, not 3035!)
        const fkbLayer = new ol.layer.Tile({
            source: new ol.source.TileWMS({
                url: 'https://wms.geonorge.no/skwms1/wms.fkb',
                params: {
                    'LAYERS': 'tiltak',
                    'FORMAT': 'image/png',
                    'TRANSPARENT': true,
                    'VERSION': '1.3.0',
                    'CRS': 'EPSG:25833'  // Changed from EPSG:3035 to EPSG:25833
                },
                projection: 'EPSG:25833',  // This layer uses UTM Zone 33N, not EPSG:3035
                crossOrigin: 'anonymous'
            }),
            opacity: 0.5,  // High transparency
            visible: true
        });

        // Simple background layer using transformed OSM
        const osmLayer = new ol.layer.Tile({
            source: new ol.source.OSM(),
            opacity: 0.6,
            visible: true  // Keep OSM visible as base layer
        });

        // Senbygg GeoTIFF layer
        let senbyggLayer = null;
        {% if geotiff_data %}
        const senbyggData = {{ geotiff_data | safe }};
        console.log('Senbygg data received:', senbyggData);
        
        if (senbyggData) {
            senbyggLayer = new ol.layer.Image({
                source: new ol.source.ImageStatic({
                    url: senbyggData.image_data,
                    projection: 'EPSG:3035',
                    imageExtent: senbyggData.bounds
                }),
                opacity: 0.7,
                visible: true
            });
            console.log('Senbygg layer created with extent:', senbyggData.bounds);
        }
        {% endif %}

        // Create vector source with significant points
        const pointsData = {{ features | safe }};
        console.log('Raw points data:', pointsData);
        
        const vectorSource = new ol.source.Vector();
        
        // Add points one by one to debug
        pointsData.forEach((pointData, index) => {
            const coords = pointData.geometry.coordinates;
            console.log(`Adding point ${index + 1}: [${coords[0]}, ${coords[1]}]`);
            
            const feature = new ol.Feature({
                geometry: new ol.geom.Point(coords)
            });
            
            // Copy properties
            Object.keys(pointData.properties).forEach(key => {
                feature.set(key, pointData.properties[key]);
            });
            
            vectorSource.addFeature(feature);
        });
        
        // Style function for clean red circles
        let currentPointSize = 8;  // Reduced from 25 to 8
        const styleFunction = function(feature) {
            // Clean red circles for all data points - no test point styling needed
            return new ol.style.Style({
                image: new ol.style.Circle({
                    radius: currentPointSize,
                    fill: new ol.style.Fill({
                        color: 'red'  // Pure red, no transparency
                    }),
                    stroke: new ol.style.Stroke({
                        color: 'white',
                        width: 1
                    })
                })
            });
        };

        const vectorLayer = new ol.layer.Vector({
            source: vectorSource,
            style: styleFunction
        });

        // Historical data layers
        const historicalData = {{ historical_data | safe }};
        const historicalLayers = {};
        
        console.log('Historical data received:', historicalData);
        
        // Create historical layers
        Object.keys(historicalData).forEach(layerName => {
            const data = historicalData[layerName];
            console.log(`Creating historical layer: ${layerName} with ${data.length} features`);
            
            const source = new ol.source.Vector();
            
            // Add features to source
            let featuresAdded = 0;
            data.forEach((featureData, index) => {
                try {
                    // Parse geometry using GeoJSON format
                    const format = new ol.format.GeoJSON();
                    const olFeature = format.readFeature(featureData, {
                        featureProjection: 'EPSG:3035',
                        dataProjection: 'EPSG:3035'
                    });
                    
                    source.addFeature(olFeature);
                    featuresAdded++;
                } catch (e) {
                    console.warn(`Error adding feature ${index} to ${layerName}:`, e);
                    console.log('Problematic feature:', featureData);
                }
            });
            
            console.log(`Successfully added ${featuresAdded} features to ${layerName}`);
            
            // Style for historical layers - different colors for different years
            const getHistoricalStyle = (layerName) => {
                let fillColor, strokeColor;
                if (layerName.includes('2023')) {
                    fillColor = 'rgba(138, 43, 226, 0.6)';  // BlueViolet with more opacity
                    strokeColor = 'rgba(138, 43, 226, 1.0)';  // Solid stroke
                } else if (layerName.includes('2024')) {
                    fillColor = 'rgba(75, 0, 130, 0.6)';   // Indigo with more opacity
                    strokeColor = 'rgba(75, 0, 130, 1.0)';   // Solid stroke
                } else {
                    fillColor = 'rgba(128, 128, 128, 0.6)'; // Default gray
                    strokeColor = 'rgba(128, 128, 128, 1.0)';
                }
                
                return new ol.style.Style({
                    fill: new ol.style.Fill({
                        color: fillColor
                    }),
                    stroke: new ol.style.Stroke({
                        color: strokeColor,
                        width: 2  // Thicker stroke for better visibility
                    })
                });
            };
            
            const layer = new ol.layer.Vector({
                source: source,
                style: getHistoricalStyle(layerName),
                visible: true,  // Start visible for debugging
                opacity: 1.0
            });
            
            historicalLayers[layerName] = layer;
            console.log(`Created layer ${layerName} with ${source.getFeatures().length} features, visible: ${layer.getVisible()}`);
        });

        // Create map with all layers
        const allLayers = [
            osmLayer,           // Base OSM layer
            fkbLayer,           // Norwegian FKB WMS (current)
        ];
        
        // Add Senbygg layer if available
        if (senbyggLayer) {
            allLayers.push(senbyggLayer);
        }
        
        // Add historical layers
        allLayers.push(...Object.values(historicalLayers));
        
        // Add change detection points on top
        allLayers.push(vectorLayer);

        const map = new ol.Map({
            target: 'map',
            layers: allLayers,
            view: new ol.View({
                projection: 'EPSG:3035',
                center: [{{ center_easting }}, {{ center_northing }}],
                zoom: 18,  // Start with very high zoom
                minZoom: 10,  // Allow higher minimum zoom
                maxZoom: 22   // Allow even higher maximum zoom
            })
        });

        // Popup for point details
        const popup = new ol.Overlay({
            element: document.createElement('div'),
            positioning: 'bottom-center',
            stopEvent: false,
            offset: [0, -10]
        });
        popup.getElement().className = 'popup';
        map.addOverlay(popup);

        // Click handler for points and historical features
        map.on('click', function(evt) {
            const features = [];
            map.forEachFeatureAtPixel(evt.pixel, function(feature, layer) {
                features.push({feature, layer});
            });

            if (features.length > 0) {
                const {feature, layer} = features[0]; // Get top feature
                const props = feature.getProperties();
                
                // Check if it's a change detection point or historical feature
                if (props.coherence !== undefined) {
                    // Change detection point
                    const coord = feature.getGeometry().getCoordinates();
                    popup.getElement().innerHTML = `
                        <h4>Significant Change Detected</h4>
                        <p><strong>Location:</strong> ${coord[0].toFixed(1)}, ${coord[1].toFixed(1)}</p>
                        <p><strong>Coherence:</strong> ${props.coherence.toFixed(3)}</p>
                        <p><strong>Change Magnitude:</strong> ${props.change_magnitude.toFixed(3)}</p>
                        <p><strong>Track:</strong> ${props.track}</p>
                        <p><strong>Confidence:</strong> ${props.confidence.toFixed(3)}</p>
                        <p><strong>Trend Slope:</strong> ${props.trend_slope.toFixed(6)}</p>
                        <p><strong>Change Date:</strong> ${props.change_date}</p>
                    `;
                } else {
                    // Historical FGDB feature
                    const coord = evt.coordinate;
                    const layerName = Object.keys(historicalLayers).find(name => 
                        historicalLayers[name] === layer
                    ) || 'Historical';
                    
                    popup.getElement().innerHTML = `
                        <h4>Historical Tiltak Feature</h4>
                        <p><strong>Layer:</strong> ${layerName.replace('tiltak_', '').toUpperCase()}</p>
                        <p><strong>Location:</strong> ${coord[0].toFixed(1)}, ${coord[1].toFixed(1)}</p>
                        ${Object.keys(props).filter(key => key !== 'geometry').slice(0, 5).map(key => 
                            `<p><strong>${key}:</strong> ${props[key]}</p>`
                        ).join('')}
                    `;
                }
                popup.setPosition(evt.coordinate);
            } else {
                popup.setPosition(undefined);
            }
        });

        // Fit view to calculated extent with buffer - force higher zoom
        const dataExtent = [{{ extent_min_easting }}, {{ extent_min_northing }}, {{ extent_max_easting }}, {{ extent_max_northing }}];
        
        // First, set the view to the data center with a high zoom
        map.getView().setCenter([{{ center_easting }}, {{ center_northing }}]);
        map.getView().setZoom(17); // Much higher zoom to see points clearly
        
        // Then fit to extent but with maximum zoom constraint
        setTimeout(() => {
            map.getView().fit(dataExtent, {
                size: map.getSize(),
                constrainResolution: false,
                padding: [100, 100, 100, 100],
                minResolution: 1.0  // Prevent zooming out too far (1m per pixel max)
            });
        }, 100);

        // Update debug info
        function updateDebugInfo() {
            const view = map.getView();
            const center = view.getCenter();
            const zoom = view.getZoom();
            const resolution = view.getResolution();
            
            // Count visible layers and features
            let visibleLayers = [];
            if (osmLayer.getVisible()) visibleLayers.push('OSM');
            if (fkbLayer.getVisible()) visibleLayers.push('WMS');
            if (senbyggLayer && senbyggLayer.getVisible()) visibleLayers.push('Senbygg');
            Object.keys(historicalLayers).forEach(name => {
                if (historicalLayers[name].getVisible()) {
                    const featureCount = historicalLayers[name].getSource().getFeatures().length;
                    visibleLayers.push(`${name}(${featureCount})`);
                }
            });
            
            document.getElementById('debug-info').innerHTML = `
                <strong>Debug Info:</strong><br>
                Points loaded: ${pointsData.length}<br>
                Features in source: ${vectorSource.getFeatures().length}<br>
                Historical layers: ${Object.keys(historicalLayers).length}<br>
                Senbygg layer: ${senbyggLayer ? 'Available' : 'Not loaded'}<br>
                Visible layers: ${visibleLayers.join(', ')}<br>
                Current center: ${center[0].toFixed(0)}, ${center[1].toFixed(0)}<br>
                Current zoom: ${zoom.toFixed(1)}<br>
                Resolution: ${resolution.toFixed(1)}m/px<br>
                Data extent: ${dataExtent[0].toFixed(0)}, ${dataExtent[1].toFixed(0)} to ${dataExtent[2].toFixed(0)}, ${dataExtent[3].toFixed(0)}
            `;
        }
        
        // Update debug info on view changes
        map.getView().on('change', updateDebugInfo);
        updateDebugInfo(); // Initial update

        // Debug information
        console.log('Map loaded with', pointsData.length, 'significant change points');
        console.log('Historical layers:', Object.keys(historicalLayers));
        console.log('Senbygg layer:', senbyggLayer ? 'Available' : 'Not loaded');
        console.log('Data extent with buffer:', dataExtent);
        console.log('Vector source features:', vectorSource.getFeatures().length);
        
        // Control functions
        window.zoomToPoints = function() {
            console.log('Zooming to points...');
            const dataExtent = [{{ extent_min_easting }}, {{ extent_min_northing }}, {{ extent_max_easting }}, {{ extent_max_northing }}];
            console.log('Fitting to extent:', dataExtent);
            
            map.getView().fit(dataExtent, {
                size: map.getSize(),
                padding: [50, 50, 50, 50],
                duration: 1000,
                maxZoom: 20
            });
        };
        
        window.togglePointSize = function() {
            currentPointSize = currentPointSize === 8 ? 15 : 8;  // Toggle between 8px and 15px
            console.log('Point size changed to:', currentPointSize);
            vectorLayer.getSource().changed();
        };
        
        window.toggleLayer = function(layerType) {
            switch(layerType) {
                case 'osm':
                    const osmVisible = osmLayer.getVisible();
                    osmLayer.setVisible(!osmVisible);
                    console.log('OSM layer:', !osmVisible ? 'shown' : 'hidden');
                    break;
                case 'fkb':
                    const fkbVisible = fkbLayer.getVisible();
                    fkbLayer.setVisible(!fkbVisible);
                    console.log('WMS Tiltak layer:', !fkbVisible ? 'shown' : 'hidden');
                    break;
                case 'senbygg':
                    if (senbyggLayer) {
                        const senbyggVisible = senbyggLayer.getVisible();
                        senbyggLayer.setVisible(!senbyggVisible);
                        console.log('Senbygg layer:', !senbyggVisible ? 'shown' : 'hidden');
                    } else {
                        console.warn('Senbygg layer not available');
                    }
                    break;
                case 'tiltak_2023':
                case 'tiltak_2024':
                    if (historicalLayers[layerType]) {
                        const layer = historicalLayers[layerType];
                        const layerVisible = layer.getVisible();
                        layer.setVisible(!layerVisible);
                        const featureCount = layer.getSource().getFeatures().length;
                        console.log(`Historical ${layerType} layer (${featureCount} features):`, !layerVisible ? 'shown' : 'hidden');
                        
                        // Force layer refresh
                        layer.getSource().changed();
                        map.render();
                    } else {
                        console.warn(`Historical layer ${layerType} not found`);
                    }
                    break;
            }
            
            // Update debug info after layer toggle
            updateDebugInfo();
        };

        // Force visibility - create extremely visible points
        setTimeout(() => {
            console.log('=== FORCING EXTREME VISIBILITY ===');
            
            // Clear existing features and add one by one with logging
            vectorSource.clear();
            
            pointsData.forEach((pointData, index) => {
                const coords = pointData.geometry.coordinates;
                console.log(`Re-adding point ${index + 1}: [${coords[0]}, ${coords[1]}]`);
                
                const feature = new ol.Feature({
                    geometry: new ol.geom.Point(coords)
                });
                
                // Copy properties
                Object.keys(pointData.properties).forEach(key => {
                    feature.set(key, pointData.properties[key]);
                });
                
                vectorSource.addFeature(feature);
            });
            
            // Force redraw
            vectorLayer.getSource().changed();
            map.render();
            
            console.log('Features after re-adding:', vectorSource.getFeatures().length);
            
            // Auto-zoom to points
            setTimeout(() => {
                window.zoomToPoints();
            }, 500);
            
        }, 1000);
    </script>
</body>
</html>
    """
    
    return render_template_string(
        html_template,
        features=json.dumps(features),
        historical_data=json.dumps(historical_data),
        geotiff_data=json.dumps(geotiff_data) if geotiff_data else None,
        historical_count=len(historical_data),
        point_count=len(significant_points),
        center_easting=center_easting,
        center_northing=center_northing,
        extent_min_easting=extent_min_easting,
        extent_max_easting=extent_max_easting,
        extent_min_northing=extent_min_northing,
        extent_max_northing=extent_max_northing
    )

def open_browser():
    """Open browser after a short delay"""
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    print("🗺️  Starting Enhanced Change Detection Map Viewer...")
    print("📊 Loading data from 'change_detection_results.csv'...")
    print("📂 Loading historical FGDB data...")
    print("🗺️  Loading Senbygg GeoTIFF layer...")
    
    # Start browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    print("🌐 Map will open at: http://127.0.0.1:5000")
    print("📍 Using coordinate system: EPSG:3035")
    print("🗂️ Historical FGDB layers: 2023, 2024")
    print("🏢 Senbygg GeoTIFF layer: binary_detection.tif")
    print("🎨 Different colors for each dataset")
    print("🛑 Press Ctrl+C to stop the server")
    
    # Run Flask app
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)