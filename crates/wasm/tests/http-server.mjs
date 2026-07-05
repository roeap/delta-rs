// Minimal static file server with Range-request support, for the FetchObjectStore
// smoke test (V7). Node's http module; no dependencies.
//
//   node tests/http-server.mjs <root-dir> [port]
//
// Then run the gated test:
//
//   WASM_SMOKE_HTTP_BASE=http://127.0.0.1:8917 wasm-pack test --node
//
// Sends `Accept-Ranges`/`Content-Range` (single ranges only) and permissive CORS
// headers so the same server works for `--headless --chrome` runs.

import { createServer } from 'node:http';
import { open, stat } from 'node:fs/promises';
import { join, normalize, resolve } from 'node:path';

const root = resolve(process.argv[2] ?? '.');
const port = Number(process.argv[3] ?? 8917);

createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://localhost');
    const path = normalize(join(root, decodeURIComponent(url.pathname)));
    if (!path.startsWith(root)) {
      res.writeHead(403).end();
      return;
    }
    const info = await stat(path).catch(() => null);
    if (!info?.isFile()) {
      res.writeHead(404).end();
      return;
    }

    const headers = {
      'accept-ranges': 'bytes',
      'access-control-allow-origin': '*',
      'access-control-allow-headers': 'Range',
      'access-control-expose-headers': 'Content-Range, Content-Length, ETag, Last-Modified',
      'last-modified': info.mtime.toUTCString(),
      etag: `"${info.size}-${Math.trunc(info.mtimeMs)}"`,
    };
    let [start, end, status] = [0, info.size - 1, 200];
    const range = req.headers.range?.match(/^bytes=(\d*)-(\d*)$/);
    if (range && (range[1] !== '' || range[2] !== '')) {
      if (range[1] === '') {
        start = Math.max(0, info.size - Number(range[2]));
      } else {
        start = Number(range[1]);
        if (range[2] !== '') end = Math.min(end, Number(range[2]));
      }
      if (start >= info.size || start > end) {
        res.writeHead(416, { 'content-range': `bytes */${info.size}` }).end();
        return;
      }
      status = 206;
      headers['content-range'] = `bytes ${start}-${end}/${info.size}`;
    }
    headers['content-length'] = end - start + 1;
    if (req.method === 'HEAD') {
      res.writeHead(status, headers).end();
      return;
    }
    const file = await open(path);
    res.writeHead(status, headers);
    file.createReadStream({ start, end, autoClose: true }).pipe(res);
  } catch {
    res.writeHead(500).end();
  }
}).listen(port, '127.0.0.1', () =>
  console.log(`serving ${root} at http://127.0.0.1:${port}`),
);
