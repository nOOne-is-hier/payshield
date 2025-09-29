from fastapi import APIRouter
import json

sms_router = APIRouter()

@sms_router.get("/transactions")
def get_transactions():
    """Fetch a list of recent credit card transactions."""
    try:
        with open("data/sample_transactions.json", "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return {"error": "Sample transactions file not found."}
    except json.JSONDecodeError:
        return {"error": "Error decoding the transactions JSON file."}