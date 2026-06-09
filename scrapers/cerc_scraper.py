import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
import pdfplumber
from io import BytesIO
import hashlib
import os

BASE_URL = "http://cercind.gov.in"
ORDERS_URL = f"{BASE_URL}/recent_orders.html"
LOOKBACK_DAYS = 7

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive"
}

def parse_date(date_str):
    try:
        return datetime.strptime(date_str.strip(), "%d.%m.%Y")
    except:
        return None

def extract_pdf_text(pdf_url):
    try:
        r = requests.get(pdf_url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        with pdfplumber.open(BytesIO(r.content)) as pdf:
            text = ""
            for page in pdf.pages[:5]:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        return text[:10000]
    except Exception as e:
        print(f"  PDF error ({pdf_url}): {e}")
        return ""

def scrape():
    print("Fetching CERC orders page...")
    r = requests.get(ORDERS_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    cutoff = datetime.now() - timedelta(days=LOOKBACK_DAYS)
    items = []

    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 5:
                continue

            sl_no = cells[0].get_text(strip=True)
            if not sl_no.isdigit():
                continue

            petition_no   = cells[1].get_text(strip=True)
            subject       = cells[2].get_text(strip=True) if len(cells) > 2 else ""
            date_order_s  = cells[3].get_text(strip=True) if len(cells) > 3 else ""
            date_upload_s = cells[4].get_text(strip=True) if len(cells) > 4 else ""
            category      = cells[5].get_text(strip=True) if len(cells) > 5 else ""

            date_uploaded = parse_date(date_upload_s)
            date_of_order = parse_date(date_order_s)

            if not date_uploaded or date_uploaded < cutoff:
                continue

            pdf_url = None
            for cell in cells:
                for a in cell.find_all("a", href=True):
                    if ".pdf" in a["href"].lower():
                        pdf_url = a["href"]
                        break
                if pdf_url:
                    break

            if not pdf_url:
                print(f"  No PDF found for {petition_no}, skipping")
                continue

            if not pdf_url.startswith("http"):
                pdf_url = BASE_URL + "/" + pdf_url.lstrip("/")

            raw_id  = f"{petition_no}_{date_order_s}"
            item_id = hashlib.md5(raw_id.encode()).hexdigest()[:16]

            print(f"  Processing: {petition_no}")
            pdf_text = extract_pdf_text(pdf_url)

            items.append({
                "id":            item_id,
                "petition_no":   petition_no,
                "subject":       subject,
                "date_of_order": date_of_order.strftime("%Y-%m-%d") if date_of_order else date_order_s,
                "date_uploaded": date_uploaded.strftime("%Y-%m-%d"),
                "category":      category,
                "pdf_url":       pdf_url,
                "pdf_text":      pdf_text,
                "scraped_at":    datetime.utcnow().isoformat() + "Z"
            })

    return items

def main():
    items = scrape()
    os.makedirs("cerc", exist_ok=True)
    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "count":        len(items),
        "items":        items
    }
    with open("cerc/cerc_new.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nDone — {len(items)} orders written to cerc/cerc_new.json")

if __name__ == "__main__":
    main()
