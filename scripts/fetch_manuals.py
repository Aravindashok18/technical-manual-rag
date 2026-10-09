"""Download the four Bureau of Reclamation FIST manuals used as the corpus into data/manuals.

FIST (Facilities Instructions, Standards, and Techniques) volumes are U.S. federal government
publications from the Bureau of Reclamation Power Resources Office:
    FIST 3-8   Operation, Maintenance, and Field Test Procedures for Protective Relays   125 pages
    FIST 3-16  Maintenance of Power Circuit Breakers                                       99 pages
    FIST 4-1A  Maintenance Scheduling for Mechanical Equipment                            222 pages
    FIST 4-1B  Maintenance Scheduling for Electrical Equipment                            162 pages
The PDFs are not stored in this repository.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://www.usbr.gov/power/data/fist/"
FILES = ["FIST_3-8_(9-2021).pdf", "FIST_3-16_(5-2024).pdf", "FIST_4-1A_(7-2020).pdf", "FIST_4-1B_(8-2022).pdf"]
DEST = Path(__file__).resolve().parent.parent / "data" / "manuals"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = DEST / name
        if target.exists():
            print("have", name)
            continue
        req = urllib.request.Request(BASE + urllib.request.quote(name), headers={"User-Agent": UA})
        blob = urllib.request.urlopen(req, timeout=120).read()
        if not blob.startswith(b"%PDF"):
            sys.exit(f"{name}: not a PDF (got {len(blob)} bytes); the file may have moved")
        target.write_bytes(blob)
        print(f"{name}: {len(blob) / 1e6:.1f} MB, sha256 {hashlib.sha256(blob).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
