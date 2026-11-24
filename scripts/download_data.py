"""Download the SISSLE DAS and seismic data (https://doi.org/10.25911/rnhb-vw10) from the ANU Data Commons.

The folder structure of the record is kept, so `--out data` yields
`data/distributed_acoustic_sensing_data/...` and `data/seismic_data/...`.
"""

import argparse
import fnmatch
import hashlib
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path

BASE_URL = "https://datacommons.anu.edu.au/DataCommons/rest/records/anudc:6317/data/"
USER_AGENT = "DAS-CN2S-downloader"


class ListingParser(HTMLParser):
    """Collects file names, sub folders and MD5 checksums from a Data Commons folder listing."""

    def __init__(self):
        super().__init__()
        self.entries = []  # (name, is_dir)
        self.md5 = []
        self._in_td = False
        self._expect_md5 = False
        self._in_th = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and "title" in attrs and attrs.get("href") and not attrs["href"].startswith(("/", "#", "http", "..", "javascript")):
            href = attrs["href"]
            self.entries.append((href.rstrip("/"), href.endswith("/")))
        self._in_th = tag == "th"
        self._in_td = tag == "td"

    def handle_data(self, data):
        if self._in_th and data.strip() == "MD5":
            self._expect_md5 = True
        elif self._in_td and self._expect_md5 and data.strip():
            self.md5.append(data.strip())
            self._expect_md5 = False

    def handle_endtag(self, tag):
        self._in_th = self._in_td = False


def fetch(url, retries=5):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"retrying {url}: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)


def crawl(rel=""):
    """Returns a list of (relative path, md5) for all files below `rel`."""
    parser = ListingParser()
    parser.feed(fetch(BASE_URL + rel))
    files = [(name, is_dir) for name, is_dir in parser.entries]
    dirs = [name for name, is_dir in files if is_dir]
    names = [name for name, is_dir in files if not is_dir]
    # every file row carries exactly one MD5 cell, in listing order
    if len(parser.md5) != len(names):
        raise RuntimeError(f"could not match MD5 checksums in listing of '{rel}' ({len(names)} files, {len(parser.md5)} checksums)")
    out = [(rel + n, m) for n, m in zip(names, parser.md5)]
    for d in dirs:
        out += crawl(rel + d + "/")
    return out


def md5sum(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def download(rel, md5, out, retries=5):
    dst = out / rel
    if dst.exists() and md5sum(dst) == md5:
        return rel, "skipped"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(BASE_URL + rel, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
                while b := r.read(1 << 24):
                    f.write(b)
            if md5sum(tmp) != md5:
                raise IOError("MD5 mismatch")
            os.replace(tmp, dst)
            return rel, "downloaded"
        except Exception as e:
            if attempt == retries - 1:
                return rel, f"FAILED ({e})"
            time.sleep(2 ** attempt)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data"), help="target directory (default: data)")
    p.add_argument("--include", nargs="*", default=["*"],
                   help="glob patterns on the relative path, e.g. 'seismic_data/*' 'distributed_acoustic_sensing_data/2023p181489/*'")
    p.add_argument("--workers", type=int, default=4, help="parallel downloads")
    p.add_argument("--list", action="store_true", help="only print the file list")
    args = p.parse_args()

    files = [(r, m) for r, m in crawl() if any(fnmatch.fnmatch(r, pat) for pat in args.include)]
    print(f"{len(files)} files")
    if args.list:
        for r, m in files:
            print(m, r)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "MD5SUMS", "w") as f:
        f.writelines(f"{m}  {r}\n" for r, m in files)

    failed = []
    with ThreadPoolExecutor(args.workers) as ex:
        futures = [ex.submit(download, r, m, args.out) for r, m in files]
        for i, fut in enumerate(as_completed(futures), 1):
            rel, status = fut.result()
            print(f"[{i}/{len(files)}] {status}: {rel}", flush=True)
            if status.startswith("FAILED"):
                failed.append(rel)
    if failed:
        sys.exit(f"{len(failed)} files failed, rerun the script to retry")


if __name__ == "__main__":
    main()
