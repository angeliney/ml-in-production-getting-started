import os
import random
import uvicorn
from fastapi import FastAPI
from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse


app = FastAPI()
templates = Jinja2Templates(directory="templates")


# list of cat images
cat_images_url = "https://firebasestorage.googleapis.com/v0/b/docker-curriculum.appspot.com/o/catnip%2F"
images = [
    cat_images_url + "0.gif?alt=media&token=0fff4b31-b3d8-44fb-be39-723f040e57fb",
    cat_images_url + "1.gif?alt=media&token=2328c855-572f-4a10-af8c-23a6e1db574c",
    cat_images_url + "10.gif?alt=media&token=647fd422-c8d1-4879-af3e-fea695da79b2",
    cat_images_url + "11.gif?alt=media&token=900cce1f-55c0-4e02-80c6-ee587d1e9b6e",
    cat_images_url + "2.gif?alt=media&token=8a108bd4-8dfc-4dbc-9b8c-0db0e626f65b",
    cat_images_url + "3.gif?alt=media&token=4e270d85-0be3-4048-99bd-696ece8070ea",
    cat_images_url + "4.gif?alt=media&token=e7daf297-e615-4dfc-aa19-bee959204774",
    cat_images_url + "5.gif?alt=media&token=a8e472e6-94da-45f9-aab8-d51ec499e5ed",
    cat_images_url + "7.gif?alt=media&token=9e449089-9f94-4002-a92a-3e44c6bd18a9",
    cat_images_url + "8.gif?alt=media&token=80a48714-7aaa-45fa-a36b-a7653dc3292b",
    cat_images_url + "9.gif?alt=media&token=a57a1c71-a8af-4170-8fee-bfe11809f0b3",
]


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    url = random.choice(images)
    return templates.TemplateResponse("index.html", {"request": request, "url": url})


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
