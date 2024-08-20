@fused.udf
def udf(bbox: fused.types.TileGDF=None):
    import geopandas as gpd
    import shapely
    import requests
    from PIL import Image
    from io import BytesIO

    def fetch_mapbox_image(access_token, style_id, bbox, width=1024, height=1024):
        minx, miny, maxx, maxy = bbox.bounds
        bbox_string = f"{minx},{miny},{maxx},{maxy}"
        url = f"https://api.mapbox.com/styles/v1/{style_id}/static/[{bbox_string}]/{width}x{height}?access_token={access_token}"
        
        response = requests.get(url)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            raise Exception(f"Error fetching image: {response.status_code} - {response.text}")

    def pixel_to_geo(x, y, bbox, img_width, img_height):
        minx, miny, maxx, maxy = bbox.bounds
        geox = minx + (x / img_width) * (maxx - minx)
        geoy = maxy - (y / img_height) * (maxy - miny)
        return geox, geoy

    def get_yolo_predictions(image):
        url = "https://5d54-122-177-98-118.ngrok-free.app/predict"
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        files = {"file": ("image.png", img_byte_arr, "image/png")}
        response = requests.post(url, files=files)
        
        if response.status_code == 200:
            return response.json()['detections']
        else:
            raise Exception(f"Error getting predictions: {response.status_code} - {response.text}")

    # Get the bbox or use a default if not provided
    bbox = bbox.geometry.iloc[0] if bbox is not None else shapely.box(-122.549, 37.681, -122.341, 37.818)
    
    print(f"Using bbox: {bbox.bounds}")
    
    # Fetch Mapbox image
    access_token = 'pk.eyJ1IjoibWlsc29uaTIwMSIsImEiOiJjbGVma203M2kwaHllM3JtdnNuODlsazY1In0.JBnsmC5BGpr6MzoYa_IJow'
    style_id = 'mapbox/satellite-v9'
    
    try:
        image = fetch_mapbox_image(access_token, style_id, bbox)
        print(f"Fetched image size: {image.size}")
    except Exception as e:
        print(f"Error fetching Mapbox image: {e}")
        return gpd.GeoDataFrame()  # Return empty GeoDataFrame on error

    # Get YOLO predictions
    try:
        predictions = get_yolo_predictions(image)
        print(f"Raw predictions: {predictions}")
    except Exception as e:
        print(f"Error getting YOLO predictions: {e}")
        return gpd.GeoDataFrame()  # Return empty GeoDataFrame on error

    # Convert predictions to georeferenced bounding boxes
    geometries = []
    labels = []
    confidences = []
    for pred in predictions:
        # Filter out predictions with confidence less than 60%
        if pred['confidence'] < 0.3:
            continue
        
        x1, y1, x2, y2 = pred['box']
        geo_x1, geo_y1 = pixel_to_geo(x1, y1, bbox, image.width, image.height)
        geo_x2, geo_y2 = pixel_to_geo(x2, y2, bbox, image.width, image.height)
        
        box = shapely.box(
            min(geo_x1, geo_x2),
            min(geo_y1, geo_y2),
            max(geo_x1, geo_x2),
            max(geo_y1, geo_y2)
        )
        
        geometries.append(box)
        labels.append(f"Class {pred['class']}")
        confidences.append(pred['confidence'])
        
        print(f"Included prediction - Class: {pred['class']}, Confidence: {pred['confidence']:.2f}")
        print(f"Original pixel bbox: ({x1}, {y1}, {x2}, {y2})")
        print(f"Converted geo bbox: {box.bounds}")

    # Create a GeoDataFrame with the georeferenced bounding boxes
    if geometries:
        gdf = gpd.GeoDataFrame(
            {"label": labels, "confidence": confidences},
            geometry=geometries,
            crs=bbox.crs if hasattr(bbox, 'crs') else "EPSG:4326"
        )
    else:
        gdf = gpd.GeoDataFrame(columns=["label", "confidence", "geometry"], crs=bbox.crs if hasattr(bbox, 'crs') else "EPSG:4326")

    print(f"Created GeoDataFrame with {len(gdf)} rows (confidence > 60%)")
    return gdf