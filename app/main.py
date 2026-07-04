import subprocess

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class TriggerWakewordRequest(BaseModel):
    audio_file_path: str


def _file_exists_locally(path: str) -> bool:
    result = subprocess.run(["test", "-f", path], capture_output=True)
    return result.returncode == 0


@app.post("/trigger-wakeword")
def trigger_wakeword(request: TriggerWakewordRequest):
    if not _file_exists_locally(request.audio_file_path):
        raise HTTPException(
            status_code=404,
            detail=f"Audio file not found: {request.audio_file_path}",
        )

    # TTS/wakeword processing logic not implemented yet.
    return {
        "status": "success",
        "message": f"Wakeword trigger received for {request.audio_file_path}",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
