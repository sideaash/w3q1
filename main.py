"""
IITM Finance Cell - Invoice Extraction API
POST /extract  ->  {"invoice_no", "date", "vendor", "amount", "tax", "currency"}
"""

import re
from datetime import datetime
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dateutil import parser as dateparser

app = FastAPI(title="Invoice Extraction API")

# --- CORS: required so the Cloudflare Worker grader can call this endpoint ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InvoiceRequest(BaseModel):
    invoice_text: str


class InvoiceResponse(BaseModel):
    invoice_no: Optional[str] = None
    date: Optional[str] = None
    vendor: Optional[str] = None
    amount: Optional[float] = None
    tax: Optional[float] = None
    currency: Optional[str] = None


# Labels that can appear in invoices. Used as "stop words" so a value capture
# doesn't accidentally swallow the next field (important because some sample
# invoices have no newlines between fields).
TAX_LABEL = r"(?:I?C?S?GST|VAT|Tax)(?:\s*\([\d.]+%\))?"
LABELS = (
    r"Invoice\s*(?:No|Number|#)|Inv\.?\s*(?:No|#)|Ref(?:erence)?\.?\s*(?:No\.?|#)?|"
    r"(?:Invoice\s*)?Date|Issued|Issue\s*Date|"
    r"Vendor(?:\s*Name)?|Seller|Bill(?:ed)?\s*(?:From|By)|Client|"
    r"Sub\s*-?\s*Total|Amount|" + TAX_LABEL + r"|"
    r"Total\s*Due|Grand\s*Total|TOTAL|Total|Currency|Bill\s*To|Address|"
    r"Service|Description"
)
STOP = rf"(?={LABELS}|\Z)"


def _clean(s: str) -> str:
    return s.strip(" \t\r\n:-,")


def _to_number(s: str) -> Optional[float]:
    """Strip currency symbols/commas/labels, return float."""
    if not s:
        return None
    s = re.sub(r"(?i)rs\.?|inr|usd|eur|gbp|₹|\$|€|£", "", s)
    s = s.replace(",", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def extract_invoice_no(text: str) -> Optional[str]:
    m = re.search(
        rf"(?:Invoice\s*(?:No|Number|#)|Inv\.?\s*(?:No|#))\.?\s*[:\-]?\s*([A-Za-z0-9\-\/]+?)\s*{STOP}",
        text, re.IGNORECASE | re.DOTALL,
    )
    if m:
        return _clean(m.group(1))
    m = re.search(
        rf"Ref(?:erence)?\.?\s*(?:No\.?|#)?\s*[:\-]\s*([A-Za-z0-9\-\/]+?)\s*{STOP}",
        text, re.IGNORECASE | re.DOTALL,
    )
    if m:
        return _clean(m.group(1))
    return None


def extract_date(text: str) -> Optional[str]:
    m = re.search(
        rf"(?<!Invoice\s)(?<!Inv\s)\b(?:Date|Issued|Issue\s*Date)\s*[:\-]?\s*(.*?){STOP}",
        text, re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    raw = _clean(m.group(1))
    if not raw:
        return None
    try:
        dt = dateparser.parse(raw, dayfirst=True, fuzzy=True)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        return None


def extract_vendor(text: str) -> Optional[str]:
    m = re.search(
        rf"(?:Vendor(?:\s*Name)?|Seller|Bill(?:ed)?\s*(?:From|By))\s*[:\-]\s*(.*?){STOP}",
        text, re.IGNORECASE | re.DOTALL,
    )
    if m:
        val = _clean(m.group(1))
        if val:
            return val
    # Fallback: some invoices put the vendor as an unlabeled header line,
    # e.g. "NovaSoft Solutions — Tax Invoice" with no "Vendor:" field at all.
    first_line = text.strip().split("\n")[0].strip()
    if first_line and ":" not in first_line:
        hm = re.match(
            r"^(.*?)\s*[—\-]\s*(?:Tax\s+)?Invoice\s*$", first_line, re.IGNORECASE
        )
        candidate = hm.group(1).strip() if hm else first_line
        generic = re.fullmatch(
            r"(?:COMMERCIAL\s+)?INVOICE|TAX\s+INVOICE|RECEIPT|BILL|STATEMENT",
            candidate, re.IGNORECASE,
        )
        if candidate and not generic:
            return candidate
    return None


def extract_amount(text: str) -> Optional[float]:
    # Subtotal / Sub Total is the amount BEFORE tax.
    m = re.search(rf"Sub\s*-?\s*Total\s*[:\-]?\s*(.*?){STOP}", text, re.IGNORECASE | re.DOTALL)
    if m:
        val = _to_number(m.group(1))
        if val is not None:
            return val
    # Fallback: a standalone "Amount:" field (not "Total"/"Grand Total")
    m = re.search(rf"(?<!Sub)\bAmount\s*[:\-]?\s*(.*?){STOP}", text, re.IGNORECASE | re.DOTALL)
    if m:
        return _to_number(m.group(1))
    return None


def extract_tax(text: str) -> Optional[float]:
    m = re.search(rf"{TAX_LABEL}\s*[:\-]\s*(.*?){STOP}", text, re.IGNORECASE | re.DOTALL)
    if m:
        return _to_number(m.group(1))
    return None


def extract_currency(text: str) -> Optional[str]:
    m = re.search(r"Currency\s*[:\-]?\s*([A-Za-z]{3})", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    if re.search(r"₹|Rs\.?|INR", text, re.IGNORECASE):
        return "INR"
    if re.search(r"\$|USD", text):
        return "USD"
    if re.search(r"€|EUR", text):
        return "EUR"
    if re.search(r"£|GBP", text):
        return "GBP"
    return "INR"  # sensible default for this use-case


@app.post("/extract", response_model=InvoiceResponse)
def extract(req: InvoiceRequest):
    text = req.invoice_text or ""
    return InvoiceResponse(
        invoice_no=extract_invoice_no(text),
        date=extract_date(text),
        vendor=extract_vendor(text),
        amount=extract_amount(text),
        tax=extract_tax(text),
        currency=extract_currency(text),
    )


@app.get("/")
def health():
    return {"status": "ok", "usage": "POST /extract with {'invoice_text': '...'}"}
