import subprocess
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
import pdfplumber
from io import BytesIO
import hashlib
import os

BASE_URL = "https://cercind.gov.in"
ORDERS_URL = f"{BASE_URL}/recent_orders.html"
LOOKBACK_DAYS = 7


def fetch(url, binary=False):
    """Use curl to handle legacy TLS that Python's SSL rejects on cercind.gov.in."""
    result = subprocess.run(
        ['curl', '-s', '-S', '-L', '-k', '--ciphers', 'DEFAULT@SECLEVEL=1', '--max-time', '30', url],
        capture_output=True, timeout=35
    )
    if result.returncode != 0:
        raise Exception(f"curl failed (code {result.returncode}): {result.stderr.decode()}")
    return result.stdout if binary else result.stdout.decode('utf-8', errors='replace')


def parse_date(date_str):
    try:
        return datetime.strptime(date_str.strip(), "%d.%m.%Y")
    except Exception:
        return None


def extract_pdf_text(pdf_url):
    try:
        pdf_bytes = fetch(pdf_url, binary=True)
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
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
    html = fetch(ORDERS_URL)
    soup = BeautifulSoup(html, "html.parser")
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
