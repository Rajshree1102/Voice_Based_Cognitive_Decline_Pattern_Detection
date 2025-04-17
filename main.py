from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
import numpy as np

from cognitive_decline_detection import detect_cognitive_decline  # Ensure this is imported correctly

app = FastAPI()

@app.post("/analyze-audio/")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        os.makedirs("temp_audio", exist_ok=True)
        file_path = os.path.join("temp_audio", file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run the cognitive decline detection function
        result = detect_cognitive_decline(file_path)  # Pass the file path

        # Extract data from the result
        text = result["audio_to_text"]
        score = float(result["risk_score"])  # Convert numpy.float32 to float
        features = result["features"]

        # Ensure all feature values are converted to native Python types (e.g., float, int, list)
        # If any values are numpy arrays, convert them to lists
        features = {key: (val.tolist() if isinstance(val, np.ndarray) else float(val)) 
                    for key, val in features.items()}

        # Return the result as a JSON response
        return {
            "filename": file.filename,
            "text" : text,
            "risk_score": score,
            "features": features
        }

    except Exception as e:
        print("Exception occurred:", e)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
