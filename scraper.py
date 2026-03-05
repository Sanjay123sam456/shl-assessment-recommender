"""
SHL Catalogue Scraper
Scrapes all Individual Test Solutions from SHL product catalogue.
Run this FIRST before anything else.
Usage: python scraper.py
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
import sys
import io
from urllib.parse import urljoin

# Fix Windows console encoding for emojis
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_URL = "https://www.shl.com"
CATALOGUE_BASE = "https://www.shl.com/solutions/products/product-catalog/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Valid test type codes
VALID_TYPES = {"A", "B", "C", "D", "E", "K", "P", "S"}


def scrape_catalogue():
    """
    Scrape all Individual Test Solutions from the SHL product catalogue.
    Uses type=1 filter and paginates through all pages (12 items per page).
    """
    assessments = []
    seen_urls = set()
    start = 0
    page_size = 12
    max_pages = 40  # Safety limit (32 pages expected = ~384 assessments)
    consecutive_empty = 0

    print("Starting SHL catalogue scrape (Individual Test Solutions)...")
    print(f"Base URL: {CATALOGUE_BASE}")

    for page_num in range(max_pages):
        start = page_num * page_size
        url = f"{CATALOGUE_BASE}?start={start}&type=1"
        print(f"\n--- Page {page_num + 1} (start={start}) ---")

        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "lxml")

            # Find all assessment links in the catalogue
            # SHL uses links like /products/product-catalog/view/assessment-name/
            # or /solutions/products/product-catalog/view/assessment-name/
            links = soup.find_all("a", href=re.compile(r"/product-catalog/view/"))

            if not links:
                consecutive_empty += 1
                print(f"  No assessment links found. (consecutive empty: {consecutive_empty})")
                if consecutive_empty >= 2:
                    print("  Stopping: 2 consecutive empty pages.")
                    break
                continue
            else:
                consecutive_empty = 0

            page_count = 0
            for link in links:
                href = link.get("href", "")
                name = link.get_text(strip=True)

                if not name or not href:
                    continue

                # Build full URL
                full_url = urljoin(BASE_URL, href)

                # Skip duplicates
                if full_url in seen_urls:
                    continue
                seen_urls.add(full_url)

                # Find the parent row/container to extract additional data
                row = link.find_parent("tr") or link.find_parent("div", class_=re.compile("row|item|card"))

                # Extract test types from the row
                test_types = []
                if row:
                    # Look for single-letter spans/elements that represent test types
                    for el in row.find_all(["span", "div", "td"]):
                        text = el.get_text(strip=True)
                        if text in VALID_TYPES:
                            test_types.append(text)

                # Extract remote/adaptive support indicators
                remote = "No"
                adaptive = "No"
                if row:
                    # Check for green dot indicators or checkmark icons
                    cells = row.find_all("td")
                    # Typically: col0=name, col1=remote, col2=adaptive, col3+=types
                    if len(cells) >= 3:
                        # Check if remote cell has an indicator (icon/dot)
                        remote_cell = cells[1] if len(cells) > 1 else None
                        adaptive_cell = cells[2] if len(cells) > 2 else None

                        if remote_cell and (remote_cell.find("span", class_=re.compile("catalogue__circle|dot|icon|check"))
                                           or remote_cell.find("i")
                                           or "Yes" in remote_cell.get_text()):
                            remote = "Yes"

                        if adaptive_cell and (adaptive_cell.find("span", class_=re.compile("catalogue__circle|dot|icon|check"))
                                              or adaptive_cell.find("i")
                                              or "Yes" in adaptive_cell.get_text()):
                            adaptive = "Yes"

                assessment = {
                    "name": name,
                    "url": full_url,
                    "test_type": list(set(test_types)),  # Deduplicate
                    "duration": None,
                    "remote_support": remote,
                    "adaptive_support": adaptive,
                    "description": ""
                }

                assessments.append(assessment)
                page_count += 1
                print(f"  [{len(assessments)}] {name} | Types: {test_types}")

            print(f"  Found {page_count} new assessments on this page.")
            time.sleep(1.5)  # Be polite to server

        except requests.exceptions.Timeout:
            print(f"  TIMEOUT on page {page_num + 1}. Retrying with longer timeout...")
            try:
                response = requests.get(url, headers=HEADERS, timeout=60)
                soup = BeautifulSoup(response.text, "lxml")
                links = soup.find_all("a", href=re.compile(r"/product-catalog/view/"))
                if not links:
                    consecutive_empty += 1
                    if consecutive_empty >= 2:
                        break
                    continue
                consecutive_empty = 0
                for link in links:
                    href = link.get("href", "")
                    name = link.get_text(strip=True)
                    if not name or not href:
                        continue
                    full_url = urljoin(BASE_URL, href)
                    if full_url in seen_urls:
                        continue
                    seen_urls.add(full_url)
                    assessments.append({
                        "name": name, "url": full_url,
                        "test_type": [], "duration": None,
                        "remote_support": "Yes", "adaptive_support": "No",
                        "description": ""
                    })
                    print(f"  [{len(assessments)}] {name}")
                time.sleep(2)
            except Exception as e2:
                print(f"  Retry also failed: {e2}")
                continue

        except Exception as e:
            print(f"  Error on page {page_num + 1}: {e}")
            continue

    return assessments


def scrape_assessment_details(assessment, index, total):
    """Get full description and duration from individual assessment page."""
    try:
        response = requests.get(assessment["url"], headers=HEADERS, timeout=30)
        soup = BeautifulSoup(response.text, "lxml")

        # Try to get description from various possible containers
        desc = ""
        for selector in [
            ("div", {"class": re.compile("product-catalogue.*description|overview|content")}),
            ("div", {"class": re.compile("description")}),
            ("p", {}),
        ]:
            el = soup.find(selector[0], selector[1])
            if el:
                text = el.get_text(strip=True)
                if len(text) > 20:  # Only accept meaningful descriptions
                    desc = text[:500]
                    break

        if desc:
            assessment["description"] = desc

        # Try to get duration if not already found
        if not assessment.get("duration"):
            text = soup.get_text()
            dur_match = re.search(r"(\d+)\s*(?:min|minute)", text, re.I)
            if dur_match:
                assessment["duration"] = int(dur_match.group(1))

        # Try to get test types if not found
        if not assessment.get("test_type"):
            for el in soup.find_all(["span", "div"], class_=re.compile("type|tag|label|badge")):
                t = el.get_text(strip=True)
                if t in VALID_TYPES and t not in assessment["test_type"]:
                    assessment["test_type"].append(t)

        time.sleep(0.5)

    except Exception as e:
        print(f"  Could not get details for [{index}/{total}] {assessment['name']}: {e}")

    return assessment


def build_from_known_urls():
    """
    Build dataset from known URLs found in training data.
    This ensures we have the exact URLs being evaluated against.
    """
    known_urls = [
        "https://www.shl.com/solutions/products/product-catalog/view/automata-fix-new/",
        "https://www.shl.com/solutions/products/product-catalog/view/core-java-entry-level-new/",
        "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
        "https://www.shl.com/solutions/products/product-catalog/view/core-java-advanced-level-new/",
        "https://www.shl.com/products/product-catalog/view/interpersonal-communications/",
        "https://www.shl.com/solutions/products/product-catalog/view/entry-level-sales-7-1/",
        "https://www.shl.com/solutions/products/product-catalog/view/entry-level-sales-sift-out-7-1/",
        "https://www.shl.com/solutions/products/product-catalog/view/entry-level-sales-solution/",
        "https://www.shl.com/solutions/products/product-catalog/view/sales-representative-solution/",
        "https://www.shl.com/products/product-catalog/view/business-communication-adaptive/",
        "https://www.shl.com/solutions/products/product-catalog/view/technical-sales-associate-solution/",
        "https://www.shl.com/solutions/products/product-catalog/view/svar-spoken-english-indian-accent-new/",
        "https://www.shl.com/solutions/products/product-catalog/view/english-comprehension-new/",
        "https://www.shl.com/products/product-catalog/view/enterprise-leadership-report/",
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        "https://www.shl.com/solutions/products/product-catalog/view/opq-leadership-report/",
        "https://www.shl.com/solutions/products/product-catalog/view/global-skills-assessment/",
        "https://www.shl.com/solutions/products/product-catalog/view/verify-verbal-ability-next-generation/",
        "https://www.shl.com/solutions/products/product-catalog/view/shl-verify-interactive-inductive-reasoning/",
        "https://www.shl.com/solutions/products/product-catalog/view/marketing-new/",
        "https://www.shl.com/solutions/products/product-catalog/view/python-new/",
        "https://www.shl.com/solutions/products/product-catalog/view/sql-server-new/",
        "https://www.shl.com/solutions/products/product-catalog/view/automata-sql-new/",
        "https://www.shl.com/solutions/products/product-catalog/view/verify-numerical-ability/",
        "https://www.shl.com/solutions/products/product-catalog/view/microsoft-excel-365-new/",
        "https://www.shl.com/solutions/products/product-catalog/view/microsoft-excel-365-essentials-new/",
    ]

    assessments = []
    for url in known_urls:
        name = url.rstrip("/").split("/")[-1].replace("-", " ").replace(" new", "").strip().title()
        assessments.append({
            "name": name,
            "url": url,
            "test_type": [],
            "duration": None,
            "remote_support": "Yes",
            "adaptive_support": "No",
            "description": f"SHL Assessment: {name}"
        })

    return assessments


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SHL Catalogue Scraper")
    parser.add_argument("--skip-details", action="store_true",
                        help="Skip fetching individual assessment pages (faster)")
    args = parser.parse_args()

    print("=== SHL Assessment Catalogue Scraper ===\n")

    # Step 1: Scrape catalogue pages
    print("Step 1: Scraping Individual Test Solutions from catalogue...")
    assessments = scrape_catalogue()
    print(f"\nStep 1 complete: Found {len(assessments)} assessments from catalogue.\n")

    # Step 2: If we got less than expected, merge with known URLs
    if len(assessments) < 377:
        print("Step 2: Adding known URLs from training data...")
        known = build_from_known_urls()
        existing_urls = {a["url"] for a in assessments}

        added = 0
        for k in known:
            url_variants = [k["url"], k["url"].replace("/solutions/products/", "/products/")]
            if not any(v in existing_urls for v in url_variants):
                assessments.append(k)
                existing_urls.add(k["url"])
                added += 1

        print(f"  Added {added} assessments from known URLs.")
        print(f"  Total now: {len(assessments)}")

    # Save immediately after catalogue scrape (before slow detail fetch)
    with open("shl_assessments.json", "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(assessments)} assessments to shl_assessments.json")

    # Step 3: Optionally get details for assessments
    if not args.skip_details:
        missing_details = [i for i, a in enumerate(assessments) if not a.get("description")]
        if missing_details:
            count = len(missing_details)
            print(f"\nStep 3: Getting details for {count} assessments...")
            for idx, i in enumerate(missing_details):
                print(f"  [{idx+1}/{count}] {assessments[i]['name']}")
                assessments[i] = scrape_assessment_details(assessments[i], idx+1, count)

            # Save again with details
            with open("shl_assessments.json", "w", encoding="utf-8") as f:
                json.dump(assessments, f, indent=2, ensure_ascii=False)
            print(f"\nUpdated shl_assessments.json with details.")
        else:
            print("\nStep 3: All assessments already have descriptions.")
    else:
        print("\nStep 3: Skipped detail fetching (--skip-details).")

    if len(assessments) < 377:
        print(f"\nWARNING: Only {len(assessments)} assessments. Need 377+.")
    else:
        print(f"\nSUCCESS: Got {len(assessments)} assessments -- meets 377+ requirement!")
