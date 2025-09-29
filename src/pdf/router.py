from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from fpdf import FPDF
import json
import io

pdf_router = APIRouter()

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'Fraudulent Transaction Report', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

@pdf_router.get("/generate-report")
def generate_pdf_report():
    """Generate a PDF report of suspicious transactions."""
    try:
        with open("data/sample_transactions.json", "r") as f:
            transactions = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return {"error": str(e)}

    suspicious_transactions = [
        t for t in transactions if t['amount'] > 1000 or t['status'] == 'pending'
    ]

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size = 10)
    
    if not suspicious_transactions:
        pdf.cell(0, 10, "No suspicious transactions found.", 0, 1)
    else:
        for t in suspicious_transactions:
            pdf.cell(0, 10, f"ID: {t['transaction_id']}, User: {t['user_id']}, Amount: {t['amount']} {t['currency']}", 0, 1)
            pdf.cell(0, 5, f"  -> Merchant: {t['merchant']}, Location: {t['location']}, Status: {t['status']}", 0, 1)
            pdf.ln(5) # Add a little space

    pdf_output = pdf.output(dest='S')
    
    return StreamingResponse(io.BytesIO(pdf_output), media_type="application/pdf", headers={"Content-Disposition": "attachment;filename=fraud_report.pdf"})