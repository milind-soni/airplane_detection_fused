@fused.udf
def udf(
    bbox: fused.types.TileGDF,
):
    import geopandas as gpd
    import pandas as pd
    import shapely
    import requests
    from PIL import Image
    from io import BytesIO
    import time
    import concurrent.futures

    utils = fused.load(
        "https://github.com/fusedio/udfs/tree/f8f0c0f/public/common/"
    ).utils

    def fetch_mapbox_image(access_token, style_id, bbox, width=1024, height=1024):
        minx, miny, maxx, maxy = bbox.total_bounds
        bbox_string = f"{minx},{miny},{maxx},{maxy}"
        url = f"https://api.mapbox.com/styles/v1/{style_id}/static/[{bbox_string}]/{width}x{height}?access_token={access_token}"

        response = requests.get(url)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            raise Exception(f"Error fetching image: {response.status_code} - {response.text}")

    def pixel_to_geo(x, y, bbox, img_width, img_height):
        minx, miny, maxx, maxy = bbox.total_bounds
        geox = minx + (x / img_width) * (maxx - minx)
        geoy = maxy - (y / img_height) * (maxy - miny)
        return geox, geoy

    def get_yolo_predictions(image, max_retries=3, retry_delay=5):
        url = "https://1207-103-48-199-244.ngrok-free.app/predict"
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        files = {"file": ("image.png", img_byte_arr, "image/png")}
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, files=files, timeout=30)
                response.raise_for_status()
                return response.json()['detections']
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise Exception(f"Max retries exceeded. Last error: {e}")

    def process_tile(bbox):
        access_token = 'pk.eyJ1IjoibWlsc29uaTIwMSIsImEiOiJjbGVma203M2kwaHllM3JtdnNuODlsazY1In0.JBnsmC5BGpr6MzoYa_IJow'
        style_id = 'mapbox/satellite-v9'

        try:
            image = fetch_mapbox_image(access_token, style_id, bbox)
            print(f"Fetched image size: {image.size}")
        except Exception as e:
            print(f"Error fetching Mapbox image: {e}")
            return None

        try:
            predictions = get_yolo_predictions(image)
            print(f"Raw predictions: {predictions}")
        except Exception as e:
            print(f"Error getting YOLO predictions: {e}")
            return None

        geometries = []
        labels = []
        confidences = []

        for pred in predictions:
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
            labels.append(f"Class_{pred['class']}")
            confidences.append(pred['confidence'])

        if geometries:
            return gpd.GeoDataFrame(
                {"label": labels, "confidence": confidences},
                geometry=geometries,
                crs=bbox.crs
            )
        return None

    # Process the tile
    result_df = process_tile(bbox)

    if result_df is not None:
        # Ensure column names are strings and coerce non-geometry columns to string
        result_df.columns = result_df.columns.astype(str)
        for col in result_df.columns:
            if col != "geometry":
                result_df[col] = result_df[col].astype(str)

        print(f"Created GeoDataFrame with {len(result_df)} rows (confidence > 30%)")
        return result_df
    else:
        print("No valid predictions found")
        return None
