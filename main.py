from fastapi import FastAPI
from nlp_service.nlp_api import app as nlp_app
from image_service.image_api import app as image_app

app = FastAPI()

app.mount("/nlp", nlp_app)
app.mount("/image", image_app)

@app.get("/")
def root():
    return {"status": "FYP Backend Running"}

