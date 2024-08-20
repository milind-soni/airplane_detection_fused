from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np

app = FastAPI()

# Load the YOLOv8 model
model = YOLO("yolov8s.pt")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Perform prediction
    results = model(image)

    # Process results
    detections = []
    for result in results:
        boxes = result.boxes.xyxy.tolist()  # Get bounding boxes
        classes = result.boxes.cls.tolist()  # Get class labels
        confs = result.boxes.conf.tolist()  # Get confidences

        for box, cls, conf in zip(boxes, classes, confs):
            detections.append(
                {"box": box, "class": int(cls), "confidence": float(conf)}
            )

    return JSONResponse(content={"detections": detections})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
