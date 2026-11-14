#從 repo 的原始下載檔（history/、drivers/）整理出 repro/data/ 底下的乾淨 CSV
#一般使用者不需要執行這支；資料已經整理好放在 data/。保留它是為了說明 CSV 怎麼來的。
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = os.path.join(ROOT, 'history')
D = os.path.join(ROOT, 'drivers')
OUT = os.path.join(ROOT, 'repro', 'data')
KEEP_FROM = pd.Timestamp('2024-10-01')  #比分析起點 2024-11-13 早一些，讓第一天的差分與 COT 週變化算得出來
KEEP_TO = pd.Timestamp('2026-11-13')

#整理盤中 K 棒（Yahoo，UTC 時間）
def build_intraday(sym, interval):
    df = pd.read_csv(os.path.join(H, f'{sym}_{interval}.csv'))
    t = pd.to_datetime(df['time_utc'], utc=True)
    df = df[(t >= KEEP_FROM.tz_localize('UTC')) & (t < (KEEP_TO + pd.Timedelta(days=1)).tz_localize('UTC'))]
    df = df[['time_utc', 'Open', 'High', 'Low', 'Close', 'Volume']]
    name = f'gold_{interval}_{sym.replace("_F", "")}.csv'
    df.to_csv(os.path.join(OUT, name), index=False)
    return name, len(df)

#讀 Yahoo 日收盤
def yclose(name):
    df = pd.read_csv(os.path.join(D, f'yahoo_{name}_1d.csv'), parse_dates=['date']).set_index('date')
    return df['Close'].astype(float)

#讀 FRED 序列（缺值 '.' 轉成 NaN）
def fred(sid):
    df = pd.read_csv(os.path.join(D, f'fred_{sid}.csv'))
    df.columns = ['date', sid]
    df['date'] = pd.to_datetime(df['date'])
    return pd.to_numeric(df[sid], errors='coerce').set_axis(df['date'])

#合併日資料：以黃金期貨交易日為列，其他市場沒開盤的日子留空白
def build_daily():
    gpr = pd.read_csv(os.path.join(D, 'gpr_data_gpr_daily_recent.csv'))
    gpr['date'] = pd.to_datetime(gpr['date'], errors='coerce')
    gpr = gpr.dropna(subset=['date']).set_index('date')['GPRD'].astype(float)
    lvl = pd.DataFrame({'gold': yclose('gc'), 'dxy': yclose('dxy'), 'broad_usd': fred('DTWEXBGS'),
                        'dfii10': fred('DFII10'), 'dgs10': fred('DGS10'), 'dgs2': fred('DGS2'),
                        't10yie': fred('T10YIE'), 'vix': yclose('vix'), 'spx': yclose('spx'),
                        'oil': yclose('cl'), 'silver': yclose('si'), 'gpr': gpr})
    lvl = lvl[lvl['gold'].notna()]
    lvl = lvl[(lvl.index >= KEEP_FROM) & (lvl.index <= KEEP_TO)]
    lvl.index.name = 'date'
    lvl.to_csv(os.path.join(OUT, 'daily_levels.csv'), date_format='%Y-%m-%d')
    return len(lvl)

#COT：資產管理人（managed money）多單與空單
def build_cot():
    cot = pd.read_csv(os.path.join(D, 'cot_gold_disagg.csv'))
    cot['report_date'] = pd.to_datetime(cot['report_date_as_yyyy_mm_dd']).dt.strftime('%Y-%m-%d')
    cot = cot.sort_values('report_date').drop_duplicates('report_date')
    cot = cot[(cot['report_date'] >= '2024-09-01') & (cot['report_date'] <= str(KEEP_TO.date()))]
    out = cot[['report_date', 'open_interest_all', 'm_money_positions_long_all', 'm_money_positions_short_all']]
    out.columns = ['report_date', 'open_interest', 'mm_long', 'mm_short']
    out.to_csv(os.path.join(OUT, 'cot_gold_managed_money.csv'), index=False)
    return len(out)

#公布日曆：只保留分析期間，欄位不變
def build_calendar():
    cal = pd.read_csv(os.path.join(D, 'release_calendar.csv'))
    cal = cal[(cal['date'] >= str(KEEP_FROM.date())) & (cal['date'] <= str(KEEP_TO.date()))]
    cal.to_csv(os.path.join(OUT, 'release_calendar.csv'), index=False)
    return len(cal)

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for sym in ['GC_F', 'MGC_F']:
        for itv in ['1h', '5m']:
            print(*build_intraday(sym, itv))
    print('daily_levels.csv', build_daily())
    print('cot_gold_managed_money.csv', build_cot())
    print('release_calendar.csv', build_calendar())
