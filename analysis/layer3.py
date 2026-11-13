#計算 MGC 12 月合約的價格特性：日波動、各時段波動、先碰停利或停損的比例
import sys, json
import numpy as np
import pandas as pd

histDir = sys.argv[1]
out = {}

#讀取 K 線並轉成紐約時間（CME 的作息依紐約時間）
def load(name):
    df = pd.read_csv(f'{histDir}/{name}.csv', parse_dates=['time_utc'], index_col='time_utc')
    df = df[['Open', 'High', 'Low', 'Close']].dropna()
    df = df[(df['High'] >= df['Low']) & (df['Close'] > 0)]
    df.index = df.index.tz_convert('America/New_York')
    return df

#一、日波動（每日最高減最低）
d1 = load('MGCZ26_CMX_1d')
d1['range'] = d1['High'] - d1['Low']
d1 = d1[d1['range'] > 0]
out['daily'] = {}
for n in (20, 60, 120):
    r = d1['range'].tail(n)
    out['daily'][n] = {'mean': round(r.mean(), 1), 'median': round(r.median(), 1), 'p10': round(r.quantile(0.1), 1), 'p90': round(r.quantile(0.9), 1), 'from': str(r.index[0].date()), 'to': str(r.index[-1].date())}
out['daily_last10'] = [(str(i.date()), round(v, 1)) for i, v in d1['range'].tail(10).items()]

#二、各時段波動：用 5 分 K 加總成每小時的高低差（紐約時間分組，再換成冬令時的臺灣時間）
m5 = load('MGCZ26_CMX_5m')
m5['hourKey'] = m5.index.floor('h')
hourly = m5.groupby('hourKey').agg(High=('High', 'max'), Low=('Low', 'min'), n=('Close', 'size'))
hourly = hourly[hourly['n'] >= 6]
hourly['range'] = hourly['High'] - hourly['Low']
hourly['nyHour'] = hourly.index.hour
byHour = hourly.groupby('nyHour')['range'].agg(['mean', 'median', 'count'])
out['hourly'] = []
for h, row in byHour.iterrows():
    twHour = (h + 13) % 24  #美國冬令時間：臺灣 = 紐約 + 13 小時
    out['hourly'].append({'ny': h, 'tw_winter': twHour, 'mean': round(row['mean'], 1), 'median': round(row['median'], 1), 'n': int(row['count'])})
out['hourly_period'] = [str(hourly.index[0].date()), str(hourly.index[-1].date())]

#三、8:30（臺灣冬令 21:30）那根 5 分 K 和其他時間比較
m5['range'] = m5['High'] - m5['Low']
at830 = m5[(m5.index.hour == 8) & (m5.index.minute == 30)]['range']
other = m5[~((m5.index.hour == 8) & (m5.index.minute == 30))]['range']
out['bar830'] = {'mean_830': round(at830.mean(), 2), 'median_830': round(at830.median(), 2), 'mean_other': round(other.mean(), 2), 'median_other': round(other.median(), 2), 'days': int(at830.size)}

#四、先碰停利還是先碰停損：每根 5 分 K 收盤當作進場，往後逐根檢查
def first_touch(df, tp, sl, maxBars=288 * 3):
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    idx = df.index
    res = {'long': [0, 0, 0, 0], 'short': [0, 0, 0, 0]}  #停利、停損、同根無法判斷、期限內都沒碰到
    waits = {'long': [], 'short': []}
    gaps = np.diff(idx.asi8) / 6e10  #相鄰 K 線間隔（分鐘），用來排除跨休市
    for i in range(len(c) - 1):
        e = c[i]
        for side in ('long', 'short'):
            up, dn = (e + tp, e - sl) if side == 'long' else (e + sl, e - tp)
            done = False
            for j in range(i + 1, min(i + 1 + maxBars, len(c))):
                hitUp, hitDn = h[j] >= up, l[j] <= dn
                if hitUp and hitDn:
                    res[side][2] += 1; done = True; break
                if hitUp or hitDn:
                    win = hitUp if side == 'long' else hitDn
                    res[side][0 if win else 1] += 1
                    waits[side].append((idx[j] - idx[i]).total_seconds() / 60)
                    done = True; break
            if not done:
                res[side][3] += 1
    summary = {}
    for side in ('long', 'short'):
        tpN, slN, amb, none = res[side]
        decided = tpN + slN
        summary[side] = {'tp': tpN, 'sl': slN, 'ambiguous': amb, 'unresolved': none, 'tp_rate': round(tpN / decided, 4) if decided else None,
                         'wait_median_min': round(float(np.median(waits[side])), 0) if waits[side] else None}
    summary['random_baseline'] = round(sl / (tp + sl), 4)
    summary['breakeven_with_fee'] = round((sl + 0.5) / (tp - 0.5 + sl + 0.5), 4)
    return summary

#用 5 分 K（約 60 天），每 3 根取一次進場點以節省計算
sample = m5[['High', 'Low', 'Close']]
out['first_touch'] = {}
combos = [(4, 24), (4, 23.5), (4, 12), (4, 4), (10, 10), (2.5, 2)]
for tp, sl in combos:
    s = sample.iloc[::1]
    out['first_touch'][f'{tp}/{sl}'] = first_touch(s, tp, sl)
    print(tp, sl, out['first_touch'][f'{tp}/{sl}'], flush=True)

json.dump(out, open(sys.argv[2], 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: out[k] for k in ('daily', 'bar830', 'hourly_period')}, ensure_ascii=False, indent=1))
