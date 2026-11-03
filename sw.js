// 離線快取：先上網拿最新版，失敗才用快取
const cacheName = 'future-calculator-v1'
const shellFiles = ['./', 'index.html', 'manifest.webmanifest', 'data/margins.json', 'icons/icon-192.png', 'icons/apple-touch-icon.png']

// 安裝時先存一份基本檔案
self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(cacheName).then((cache) => cache.addAll(shellFiles)).then(() => self.skipWaiting()))
})

// 啟用時清掉舊版快取
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== cacheName).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  )
})

// 同網站的請求：網路優先，成功就更新快取
self.addEventListener('fetch', (event) => {
  const req = event.request
  if (req.method !== 'GET' || new URL(req.url).origin !== self.location.origin) return
  event.respondWith(
    fetch(req)
      .then((res) => {
        const copy = res.clone()
        caches.open(cacheName).then((cache) => cache.put(req, copy))
        return res
      })
      .catch(() => caches.match(req, { ignoreSearch: true }))
  )
})
