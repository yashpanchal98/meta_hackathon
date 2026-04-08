from fastapi import FastAPI
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok", "version": "test"}

@app.post("/reset")
def reset():
    return {"observation": "test ok", "task": "email_triage", "step": 0, "done": False}
