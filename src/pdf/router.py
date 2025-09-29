from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from fpdf import FPDF
import json
import io
import os
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI

import httpx

# Load environment variables from .env file
load_dotenv()

pdf_router = APIRouter()

# Manually create an httpx client to bypass any proxy auto-detection issues.
http_client = httpx.AsyncClient()

# Initialize OpenAI client, passing the manually created http_client.
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=http_client,
)

async def generate_llm_analysis(t):
    """Generates a natural language analysis of a transaction by calling the OpenAI API."""
    prompt_text = f"""Analyze the following credit card transaction and provide a brief, professional fraud risk analysis in Korean. 
    Focus on why it might be suspicious. Be concise.

    Transaction data: 
    {json.dumps(t, indent=2, ensure_ascii=False)}
    """
    try:
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant acting as a fraud detection analyst.",
                },
                {
                    "role": "user",
                    "content": prompt_text,
                },
            ],
            model="gpt-3.5-turbo",
            temperature=0.3,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"LLM 분석 중 오류 발생: {str(e)}"

class PDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_font('NanumGothic', 'B', 'fonts/NanumGothic-Regular.ttf')
        self.add_font('NanumGothic', '', 'fonts/NanumGothic-Regular.ttf')

    def header(self):
        self.set_font('NanumGothic', 'B', 16)
        self.cell(0, 15, '이상 거래 탐지 보고서 (LLM 분석 포함)', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('NanumGothic', '', 8)
        today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cell(0, 10, f'보고서 생성 시각: {today} / Page {self.page_no()}', 0, 0, 'C')

@pdf_router.get("/generate-report")
async def generate_pdf_report():
    """Generate a PDF report of suspicious transactions with real LLM analysis."""
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
    pdf.add_font('NanumGothic', '', 'fonts/NanumGothic-Regular.ttf')
    pdf.set_font("NanumGothic", size = 11)
    
    if not suspicious_transactions:
        pdf.cell(0, 10, "탐지된 이상 거래가 없습니다.", 0, 1)
    else:
        pdf.write(8, f"총 {len(suspicious_transactions)}건의 이상 의심 거래가 탐지되었습니다. LLM 분석 결과는 아래와 같습니다.")
        pdf.ln(15)
        
        for t in suspicious_transactions:
            analysis_text = await generate_llm_analysis(t)
            pdf.set_font("NanumGothic", 'B', size=12)
            pdf.write(8, f"- 거래 ID {t['transaction_id']} 분석 결과:")
            pdf.ln(8)
            pdf.set_font("NanumGothic", '', size=11)
            pdf.write(8, analysis_text)
            pdf.ln(10)

    pdf_output = pdf.output()
    
    return StreamingResponse(io.BytesIO(pdf_output), media_type="application/pdf", headers={"Content-Disposition": "attachment;filename=fraud_report_real_llm.pdf"})