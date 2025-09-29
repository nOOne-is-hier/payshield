
from fastapi import FastAPI
from src.sms.router import sms_router
from src.pdf.router import pdf_router

app = FastAPI()

app.include_router(sms_router, prefix="/sms")
app.include_router(pdf_router, prefix="/pdf")

@app.get("/")
def read_root():
    return {"message": "Fraud Detection API"}
