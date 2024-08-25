# gai_stt_router.py
from fastapi import APIRouter, File, UploadFile, HTTPException
from io import BytesIO
from pydub import AudioSegment
import tempfile
import numpy as np
import uuid

def create_router(app):

    router = APIRouter()

    @router.post("/gen/v1/audio/transcriptions")
    async def _speech_to_text(file: UploadFile = File(...)):
        host = app.state.host
        if not host:
            raise HTTPException(status_code=500, detail="Host not initialized yet. Please try again later.")
        try:
            print(f"Received file with filename: {file.filename} {file.content_type}")
            content = await file.read()
            
            # Convert webm file to wav if necessary
            if file.content_type == "audio/webm":
                audio = BytesIO(content)
                audio = AudioSegment.from_file(audio, format="webm")
                
                # Export audio to wav and get data
                with tempfile.NamedTemporaryFile(delete=True) as tmp:
                    audio.export(tmp.name, format="wav")
                    
                    # Read wav file data into numpy array
                    wav_file_data = np.memmap(tmp.name, dtype='h', mode='r')
                
            else:
                # If file is already in wav format, just read the data into numpy array
                wav_file_data = content

            return host.generator.create(file=wav_file_data)    

        except Exception as e:
            error_id = str(uuid.uuid4())
            raise HTTPException(status_code=500, detail=f"Failed to process audio file: {error_id}")
        
    return router

