import modal
from fastapi import FastAPI

# Create Modal App instance
app = modal.App("unified-trading-bot")

# Define the container image with all required dependencies
image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

# Import your FastAPI app from backend
def get_fastapi_app():
    from backend.main import app as fastapi_app
    return fastapi_app

@modal.function(image=image)
@modal.asgi_app()
def fastapi_entrypoint():
    return get_fastapi_app()

# Example keepalive (optional, remove if not used)
@modal.function(
    image=image,
    schedule=modal.Cron("0 */6 * * *"),
    timeout=3600
)
def keepalive():
    print("Keeping Modal app alive...")

# Example daily market update (optional, remove if not used)
@modal.function(
    image=image,
    schedule=modal.Cron("0 9 * * 1-5"),
    timeout=1800
)
def daily_market_update():
    print("Running daily market data update...")

# NOTE:
# - Do NOT set app = fastapi_app() at the module level.
# - The entrypoint for Modal is the 'app' (Modal App), and the FastAPI is served via the @modal.asgi_app() function above.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(get_fastapi_app(), host="0.0.0.0", port=8000)