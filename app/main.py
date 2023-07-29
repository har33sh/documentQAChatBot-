import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from app.chat import router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def main():
    content = """
            Hello World.. !
            """
    return HTMLResponse(content=content)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
