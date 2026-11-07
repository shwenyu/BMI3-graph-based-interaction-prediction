"""Download original gold-standard inputs, never the paper's predicted edges."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

BASE = "http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = [name for code in ("e", "ic", "gpcr", "nr") for name in
             (f"{code}_admat_dgc.txt", f"{code}_simmat_dc.txt",
              f"{code}_simmat_dg.txt", f"bind_orfhsa_drug_{code}.txt")]

    def fetch(name):
        # A failed request raises; existing valid local files are not replaced by HTML.
        with urlopen(BASE + name, timeout=60) as response:
            raw = response.read()
            final_url = response.geturl()
        if not raw or b"<html" in raw[:500].lower():
            raise ValueError(f"Not a TXT dataset: {name}")
        raw.decode("utf-8-sig")
        temporary = args.output / (name + ".part")
        temporary.write_bytes(raw)
        os.replace(temporary, args.output / name)
        return dict(file=name, url=BASE + name, resolved_url=final_url,
                    bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                    retrieved_utc=datetime.now(timezone.utc).isoformat())

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        entries = list(pool.map(fetch, names))
    (args.output / "manifest.json").write_text(json.dumps(entries, indent=2) + "\n")
    print(f"Downloaded and hashed {len(entries)} files to {args.output}")


if __name__ == "__main__":
    main()
