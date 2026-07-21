#!/usr/bin/env python3
"""Range-capable static server for local PMTiles testing.

The pmtiles protocol fetches byte ranges from the archive; Python's stdlib
http.server ignores the Range header, so tiles never load locally. This adds
minimal HTTP Range (206) support. GitHub Pages / object storage serve Range
natively, so this is only for local dev/verification.

    python serve.py [port]   # default 8991, serves the prototype dir
"""
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            return super().send_head()
        m = re.match(r"bytes=(\d+)-(\d*)", rng)
        if not m:
            return super().send_head()
        size = os.path.getsize(path)
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else size - 1
        end = min(end, size - 1)
        if start > end:
            self.send_error(416)
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self._range = (f, end - start + 1)
        return f

    def copyfile(self, source, outputfile):
        rng = getattr(self, "_range", None)
        if not rng:
            return super().copyfile(source, outputfile)
        f, remaining = rng
        while remaining > 0:
            chunk = f.read(min(65536, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8991
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"serving {os.getcwd()} on http://127.0.0.1:{port} (Range-capable)")
    ThreadingHTTPServer(("127.0.0.1", port), RangeHandler).serve_forever()
