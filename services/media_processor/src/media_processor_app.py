from fastapi import FastAPI
from exports.src.db_clients.lifespan import lifespan
from exports.src.utils.logger import get_logger

app = FastAPI(lifespan=lifespan)
logger = get_logger()


# ---------------------- UPLOAD FLOW ----------------------
@app.post("/upload_media/")
async def upload_media():
    pass


# ---------------------- SEARCH FLOW ----------------------
@app.post("/search_media/")
async def search_media():
    pass
