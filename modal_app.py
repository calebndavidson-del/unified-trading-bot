import modal
from fastapi import FastAPI

app = modal.App("unified-trading-bot")

image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

def get_fastapi_app():
    from backend.main import app as fastapi_app
    return fastapi_app

@app.function(image=image)
@modal.asgi_app()
def fastapi_entrypoint():
    return get_fastapi_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(get_fastapi_app(), host="0.0.0.0", port=8000)
