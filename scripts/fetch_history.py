#從 Yahoo Finance 抓 MGC／GC 歷史 K 線，存成 history/ 底下的 CSV
import sys, json
from datetime import datetime, timezone
import yfinance as yf

#要抓的商品（代號：說明）
symbols = {
    'MGCZ26.CMX': '微黃金 2026 年 12 月合約（元大目前交易的月份）',
    'MGC=F': '微黃金連續近月合約',
    'GC=F': '黃金連續近月合約'
}
#K 線週期與 Yahoo 允許的最長期間
intervals = {'1d': '2y', '1h': '730d', '15m': '60d', '5m': '60d', '1m': '7d'}

summary = []
for sym, label in symbols.items():
    for itv, period in intervals.items():
        try:
            df = yf.download(sym, period=period, interval=itv, auto_adjust=False, progress=False, prepost=True)
        except Exception as e:
            summary.append({'symbol': sym, 'interval': itv, 'error': str(e)})
            continue
        if df is None or df.empty:
            summary.append({'symbol': sym, 'interval': itv, 'rows': 0})
            continue
        #yfinance 新版欄位是多層索引，攤平成單層
        if hasattr(df.columns, 'levels'):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.index = df.index.tz_convert('UTC') if df.index.tz is not None else df.index.tz_localize('UTC')
        df.index.name = 'time_utc'
        safe = sym.replace('=', '_').replace('.', '_')
        df.to_csv(f'history/{safe}_{itv}.csv')
        summary.append({'symbol': sym, 'label': label, 'interval': itv, 'rows': len(df),
                        'first': str(df.index[0]), 'last': str(df.index[-1]),
                        'last_close': float(df['Close'].iloc[-1])})
        print(sym, itv, len(df), df.index[0], df.index[-1])

json.dump({'fetched_at': datetime.now(timezone.utc).isoformat(), 'items': summary},
          open('history/summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
