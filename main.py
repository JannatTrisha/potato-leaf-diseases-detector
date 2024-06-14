# from fastapi import FastAPI, File, UploadFile
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf

# app = FastAPI()

# # Load the model
# MODEL = tf.keras.models.load_model("../saved_models/1.keras")
# CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

# @app.get("/")
# async def root():
#     return {"message": "Hello, World!"}

# @app.get("/ping")
# async def ping():
#     return {"message": "Pong"}

# def read_file_as_image(data) -> np.ndarray:
#     try:
#         image = Image.open(BytesIO(data))
#         return np.array(image)
#     except Exception as e:
#         print(f"Error reading image: {e}")
#         raise

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         # Read the file content
#         file_content = await file.read()

#         # Convert the file content to an image
#         image = read_file_as_image(file_content)

#         # Ensure the image has 3 channels (RGB)
#         if image.ndim == 2:
#             image = np.stack((image,) * 3, axis=-1)
#         elif image.shape[2] == 4:
#             image = image[..., :3]

#         # Expand dimensions to create a batch
#         img_batch = np.expand_dims(image, 0)

#         # Make prediction
#         predictions = MODEL.predict(img_batch)

#         # Get the predicted class and confidence
#         predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#         confidence = float(np.max(predictions[0]))

#         # Return the prediction result
#         return {
#             "filename": file.filename,
#             "predicted_class": predicted_class,
#             "confidence": confidence
#         }
#     except Exception as e:
#         return {"error": str(e)}

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8001)
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../saved_models/1.keras")
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

@app.get("/ping")
async def ping():
    return {"message": "Pong"}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_content = await file.read()
    image = read_file_as_image(file_content)
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    return {"predicted_class": predicted_class, "confidence": confidence}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)

