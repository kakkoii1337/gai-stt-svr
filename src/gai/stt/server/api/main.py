import os
os.environ["LOG_LEVEL"]="DEBUG"
from gai.lib.common.logging import getLogger
logger = getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()

# GAI
from gai.lib.common.errors import *

# Router
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from io import BytesIO
from pydub import AudioSegment
import tempfile
import numpy as np
import uuid

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


# __main__
if __name__ == "__main__":
    # # Run self-test before anything else
    # import os
    # if os.environ.get("SELF_TEST",None):
    #     self_test_file=os.path.join(os.path.dirname(os.path.abspath(__file__)),"self-test.py")
    #     import subprocess,sys
    #     try:
    #         subprocess.run([f"python {self_test_file}"],shell=True,check=True)
    #     except subprocess.CalledProcessError as e:
    #         sys.exit(1)
    #     ## passed self-test

    import uvicorn
    pyproject_toml = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", "..", "..", "..", "..", "pyproject.toml")
    from gai.lib.server import api_factory
    app = api_factory.create_app(pyproject_toml, category="stt")
    app.include_router(router, dependencies=[Depends(lambda: app.state.host)])
    config = uvicorn.Config(
        app=app, 
        host="0.0.0.0", 
        port=12033, 
        timeout_keep_alive=180,
        timeout_notify=150,
        workers=1
    )
    server = uvicorn.Server(config=config)
    server.run()
