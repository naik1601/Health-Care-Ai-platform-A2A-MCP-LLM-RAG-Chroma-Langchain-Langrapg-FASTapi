from fastapi import FastAPI

app = FastAPI(
    title="Healthcare AI Platform",
    description="Backend for our healthcare assistant.",
    version="0.1.0",
)


@app.get("/")
async def home():
    return {"message": "Healthcare AI backend is running"}


@app.get("/health")
async def health_check():
    return {"status": "ok"}