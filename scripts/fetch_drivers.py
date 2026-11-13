#抓黃金價格驅動因素資料（美元、利率、VIX、股市、地緣風險、COT、經濟數據日曆原始頁），存到 drivers/
#只寫入 drivers/ 資料夾，不碰計算機用到的任何檔案
import os, io, json, time
from datetime import datetime, timezone
import sys
import requests
import pandas as pd
import yfinance as yf

OUT = 'drivers'
RAW = os.path.join(OUT, 'raw')
os.makedirs(RAW, exist_ok=True)
UA = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36',
      'Accept-Language': 'en-US,en;q=0.9'}
START = '2024-06-01'
summary = []
SEC = sys.argv[1] if len(sys.argv) > 1 else 'all'
def on(x):
    return SEC in ('all', x)

def note(**kw):
    summary.append(kw)
    print(kw, flush=True)

#1. Yahoo 日線
if on('yahoo'):
    yahoo = {'DX-Y.NYB': 'dxy', '^VIX': 'vix', '^GSPC': 'spx', '^TNX': 'tnx', 'GC=F': 'gc',
             'SI=F': 'si', 'TIP': 'tip', 'UUP': 'uup', '^IRX': 'irx', 'CL=F': 'cl', 'BTC-USD': 'btc'}
    for sym, name in yahoo.items():
        try:
            df = yf.download(sym, start=START, interval='1d', auto_adjust=False, progress=False)
            if hasattr(df.columns, 'levels'):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df.index.name = 'date'
            df.to_csv(f'{OUT}/yahoo_{name}_1d.csv')
            note(src='yahoo', sym=sym, rows=len(df), first=str(df.index[0]), last=str(df.index[-1]))
        except Exception as e:
            note(src='yahoo', sym=sym, error=str(e))

#1b. Yahoo 美元指數與標普小時線（事件窗口可對照）
if on('yahoo1h'):
    for sym, name in {'DX-Y.NYB': 'dxy', '^TNX': 'tnx'}.items():
        try:
            df = yf.download(sym, period='730d', interval='1h', auto_adjust=False, progress=False)
            if hasattr(df.columns, 'levels'):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df.index = df.index.tz_convert('UTC') if df.index.tz is not None else df.index.tz_localize('UTC')
            df.index.name = 'time_utc'
            df.to_csv(f'{OUT}/yahoo_{name}_1h.csv')
            note(src='yahoo1h', sym=sym, rows=len(df))
        except Exception as e:
            note(src='yahoo1h', sym=sym, error=str(e))

#2. FRED（免金鑰 CSV）
if on('fred'):
    fred = ['DFII10', 'DGS10', 'DGS2', 'DTWEXBGS', 'T10YIE', 'VIXCLS', 'SP500', 'DFF', 'T5YIE', 'DFII5']
    for sid in fred:
        url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={START}'
        try:
            try:
                r = requests.get(url, headers=UA, timeout=(15, 45))
                r.raise_for_status()
            except Exception as e1:
                note(src='fred', id=sid, first_error=str(e1))
                r = requests.get(f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}', headers={'User-Agent': 'curl/8.5.0'}, timeout=(15, 45))
                r.raise_for_status()
            open(f'{OUT}/fred_{sid}.csv', 'w').write(r.text)
            note(src='fred', id=sid, lines=r.text.count('\n'))
        except Exception as e:
            note(src='fred', id=sid, error=str(e))
        time.sleep(1)

#3. 地緣政治風險指數（Caldara-Iacoviello 日資料）
if on('gpr'):
    for url in ['https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls',
                'https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xlsx',
                'https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls']:
        try:
            r = requests.get(url, headers=UA, timeout=(15, 45))
            r.raise_for_status()
            fn = url.rsplit('/', 1)[1]
            open(f'{RAW}/{fn}', 'wb').write(r.content)
            try:
                df = pd.read_excel(io.BytesIO(r.content))
                df.to_csv(f'{OUT}/gpr_{fn.split(".")[0]}.csv', index=False)
                note(src='gpr', url=url, rows=len(df), cols=list(map(str, df.columns))[:12])
            except Exception as e:
                note(src='gpr', url=url, bytes=len(r.content), parse_error=str(e))
        except Exception as e:
            note(src='gpr', url=url, error=str(e))

#4. CFTC COT 黃金（088691）分類報告 + 傳統報告
if on('cot'):
    for ds, name in [('72hh-3qpy', 'disagg'), ('6dca-aqww', 'legacy')]:
        url = (f'https://publicreporting.cftc.gov/resource/{ds}.csv?'
               f"$where=cftc_contract_market_code='088691' AND report_date_as_yyyy_mm_dd>'2024-01-01'&$limit=5000")
        try:
            r = requests.get(url, headers=UA, timeout=(15, 45))
            r.raise_for_status()
            open(f'{OUT}/cot_gold_{name}.csv', 'w').write(r.text)
            note(src='cot', ds=name, lines=r.text.count('\n'))
        except Exception as e:
            note(src='cot', ds=name, error=str(e))

#5. 經濟數據公布日曆原始頁（之後在本機解析）
if on('calendar'):
    pages = {
        'fomc_calendar.html': 'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm',
        'bls_sched_2024.html': 'https://www.bls.gov/schedule/2024/home.htm',
        'bls_sched_2025.html': 'https://www.bls.gov/schedule/2025/home.htm',
        'bls_sched_2026.html': 'https://www.bls.gov/schedule/2026/home.htm',
        'bls_empsit_sched.html': 'https://www.bls.gov/schedule/news_release/empsit.htm',
        'bls_cpi_sched.html': 'https://www.bls.gov/schedule/news_release/cpi.htm',
        'bls_ppi_sched.html': 'https://www.bls.gov/schedule/news_release/ppi.htm',
        'bls_ics.ics': 'https://www.bls.gov/schedule/news_release/bls.ics',
        'bea_schedule.html': 'https://www.bea.gov/news/schedule/full',
        'bea_schedule2.html': 'https://www.bea.gov/news/schedule',
        'census_retail.html': 'https://www.census.gov/retail/index.html',
        'census_sched.html': 'https://www.census.gov/economic-indicators/calendar-listview.html',
    }
    for fn, url in pages.items():
        try:
            r = requests.get(url, headers=UA, timeout=(15, 45))
            open(f'{RAW}/{fn}', 'wb').write(r.content)
            note(src='calendar', fn=fn, status=r.status_code, bytes=len(r.content))
        except Exception as e:
            note(src='calendar', fn=fn, error=str(e))

#6. ALFRED 資料版本日期 = 實際公布日（各指標的 vintage dates）
if on('alfred'):
    for sid in ['PAYEMS', 'CPIAUCSL', 'PPIFIS', 'PCEPI', 'GDP', 'RSAFS', 'ICSA']:
        url = f'https://alfred.stlouisfed.org/series/downloaddata?seid={sid}'
        for u in [f'https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={sid}&vintage_date_mode=all&cosd=2024-06-01',
                  f'https://alfred.stlouisfed.org/series?seid={sid}']:
            try:
                r = requests.get(u, headers=UA, timeout=(15, 45))
                fn = f'alfred_{sid}_{"csv" if "csv" in u else "page"}.txt'
                open(f'{RAW}/{fn}', 'wb').write(r.content)
                note(src='alfred', fn=fn, status=r.status_code, bytes=len(r.content))
            except Exception as e:
                note(src='alfred', sid=sid, error=str(e))
            time.sleep(1)

#7. 公布日期第二來源：FRED 發布日曆頁 + Wayback 存檔的 BLS 歷史新聞稿清單
if on('dates'):
    CURL = {'User-Agent': 'curl/8.5.0'}
    rids = {'empsit': 50, 'cpi': 10, 'ppi': 46, 'pio': 54, 'gdp': 53, 'retail': 9, 'claims': 180, 'fomc': 101}
    for name, rid in rids.items():
        for y in (2024, 2025, 2026):
            u = f'https://fred.stlouisfed.org/releases/calendar?rid={rid}&y={y}'
            try:
                r = requests.get(u, headers=CURL, timeout=(15, 40))
                open(f'{RAW}/fredcal_{name}_{y}.html', 'wb').write(r.content)
                note(src='fredcal', name=name, y=y, status=r.status_code, bytes=len(r.content))
            except Exception as e:
                note(src='fredcal', name=name, y=y, error=str(e))
    wb = {'bls_arch_empsit': 'https://www.bls.gov/bls/news-release/empsit.htm',
          'bls_arch_cpi': 'https://www.bls.gov/bls/news-release/cpi.htm',
          'bls_arch_ppi': 'https://www.bls.gov/bls/news-release/ppi.htm',
          'census_marts_sched': 'https://www.census.gov/retail/release_schedule.html',
          'bea_sched_2025': 'https://www.bea.gov/news/schedule/full'}
    for name, url in wb.items():
        for stamp in ('2026', '20251231', '20250115'):
            if name != 'bea_sched_2025' and stamp != '2026':
                continue
            if name == 'bea_sched_2025' and stamp == '2026':
                continue
            u = f'https://web.archive.org/web/{stamp}id_/{url}'
            try:
                r = requests.get(u, headers=UA, timeout=(20, 60))
                open(f'{RAW}/wb_{name}_{stamp}.html', 'wb').write(r.content)
                note(src='wayback', name=name, stamp=stamp, status=r.status_code, bytes=len(r.content))
            except Exception as e:
                note(src='wayback', name=name, stamp=stamp, error=str(e))

json.dump({'section': SEC, 'fetched_at': datetime.now(timezone.utc).isoformat(), 'items': summary},
          open(f'{OUT}/summary_{SEC}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
