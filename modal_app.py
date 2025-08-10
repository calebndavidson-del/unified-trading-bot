#!/usr/bin/env python3
"""
Modal deployment wrapper for Unified Trading Bot
Deploys the FastAPI backend to Modal cloud platform
"""

import modal

# Create Modal app
app = modal.App("unified-trading-bot")

# Define the container image with required dependencies
image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

@app.function(
    image=image,
    schedule=modal.Cron("0 */6 * * *"),  # Run every 6 hours to keep warm
    timeout=3600  # 1 hour timeout
)
def keepalive():
    """Keep the Modal app warm by running periodically"""
    print("Keeping Modal app alive...")

@app.function(
    image=image,
    allow_concurrent_inputs=100,
    timeout=300,  # 5 minute timeout for API calls
)
@modal.asgi_app()
def fastapi_app():
    """Deploy FastAPI backend to Modal"""
    # Import the FastAPI app from backend
    from backend.main import app as fastapi_app
    return fastapi_app

@app.function(
    image=image,
    schedule=modal.Cron("0 9 * * 1-5"),  # Run weekdays at 9 AM UTC
    timeout=1800  # 30 minute timeout
)
def daily_market_update():
    """Run daily market data updates"""
    print("Running daily market data update...")
    # This can be expanded to include any scheduled market data processing

if __name__ == "__main__":
    # This allows the script to be run locally for testing
    print("Modal app configured for unified-trading-bot")
    print("FastAPI backend will be deployed to Modal cloud platform")