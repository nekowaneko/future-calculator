#從元大期貨抓國外期貨保證金，存成 data/margins.json 給計算機讀取
import json, re, sys, time, urllib.request
from datetime import datetime, timezone, timedelta

#常數定義
apiUrl = 'https://www.yuantafutures.com.tw/api/ProcessMargin?type=FR'
sourceUrl = 'https://www.yuantafutures.com.tw/margin_02'
outPath = 'data/margins.json'
minRows = 100  #筆數低於此值視為異常，不覆蓋舊資料
taipei = timezone(timedelta(hours=8))

#交易所代碼對應的顯示名稱
exchangeNames = {
    'CME': '芝商所 CME', 'SGX': '新加坡交易所 SGX', 'TCE': '東京商品交易所 TOCOM',
    'OSE': '日本交易所 JPX', 'KRX': '韓國交易所 KRX', 'HKF': '香港交易所 HKEx',
    'ICE-US': '洲際交易所 ICE 美國', 'ICE-EU': '洲際交易所 ICE 歐洲', 'ICE-UK': '洲際交易所 ICE 英國',
    'IFS': '洲際交易所 ICE 新加坡', 'EUX': '歐洲交易所 Eurex', 'LME': '倫敦金屬交易所 LME',
    'ASX': '澳洲交易所 ASX', 'CBOE': 'CBOE 期貨交易所 CFE', 'NIF': '印度國際交易所 NIF'
}

#呼叫元大資料介面；偶爾會回空白網頁，所以重試
def fetch_rows(retries=4):
    lastError = None
    for i in range(retries):
        url = f'{apiUrl}&_={int(time.time() * 1000)}'
        req = urllib.request.Request(url, headers={
            'X-Requested-With': 'XMLHttpRequest',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Referer': sourceUrl,
            'User-Agent': 'Mozilla/5.0 (future-calculator margin updater)'
        })
        try:
            with urllib.request.urlopen(req, timeout=30) as res:
                text = res.read().decode('utf-8')
            return json.loads(text)['result']
        except Exception as e:
            lastError = e
            print(f'第 {i + 1} 次失敗：{e}')
            time.sleep(5 * (i + 1))
    raise RuntimeError(f'抓取失敗：{lastError}')

#轉成計算機用的格式
def to_product(r):
    code = r['comno'].strip()
    tick = float(r['tickamt'])
    name = re.sub(r'\s*[\(（]?' + re.escape(code) + r'[\)）]?\s*$', '', r['comnm']).strip() or r['comnm']
    return {
        'group': exchangeNames.get(r['exh'], r['exh']), 'code': code, 'name': name, 'curr': r['curr'],
        'tick': int(tick) if tick.is_integer() else tick,
        'im': int(str(r['im']).replace(',', '')), 'mm': int(str(r['mm']).replace(',', ''))
    }

if __name__ == '__main__':
    rows = [r for r in fetch_rows() if r.get('comtype') == 'F']
    products = [to_product(r) for r in rows]
    if len(products) < minRows:
        sys.exit(f'只有 {len(products)} 筆，低於 {minRows}，保留舊資料')
    data = {
        'updated': datetime.now(taipei).strftime('%Y/%m/%d %H:%M'),
        'source': sourceUrl,
        'count': len(products),
        'products': products
    }
    with open(outPath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    mgc = next((p for p in products if p['code'] == 'MGC'), None)
    print(f'已更新 {len(products)} 檔，時間 {data["updated"]}，MGC：{mgc}')
