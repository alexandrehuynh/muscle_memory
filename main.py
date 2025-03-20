import uvicorn
from fastapi import FastAPI
from api.routers import movement, feedback

app = FastAPI(
    title="Muscle Memory",
    description="AI-Enhanced Fitness Analysis Service",
    version="0.1.0"
)

# Include routers
app.include_router(movement.router, prefix="/api/v1", tags=["Movement Analysis"])
app.include_router(feedback.router, prefix="/api/v1", tags=["AI Feedback"])

@app.get("/")
def read_root():
    return {"message": "Welcome to Muscle Memory API"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)