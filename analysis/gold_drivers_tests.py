#黃金（GC/MGC）價格驅動因素統計檢定
#輸入：history/（Yahoo 黃金 K 線）、drivers/（因素資料與公布日曆 release_calendar.csv）
#輸出：analysis/out/own_tests_results.json
#用法：python3 analysis/gold_drivers_tests.py [輸出路徑]
import sys, json, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.multitest import multipletests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = os.path.join(ROOT, 'history')
D = os.path.join(ROOT, 'drivers')
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'analysis', 'out', 'own_tests_results.json')
START, END = pd.Timestamp('2024-11-13'), pd.Timestamp('2026-11-13')
NY = 'America/New_York'
rng = np.random.default_rng(20261113)

def grade(p):
    if p is None or not np.isfinite(p):
        return 'N/A'
    return '非常強' if p < 0.001 else '強' if p < 0.01 else '中等' if p < 0.05 else '弱' if p < 0.10 else '無證據'

def tw_time(et_hhmm):
    h, m = map(int, et_hhmm.split(':'))
    w = (h + 13) % 24; s = (h + 12) % 24
    return f'台灣 {w:02d}:{m:02d}（美國冬令時間）／{s:02d}:{m:02d}（美國夏令時間）'

def load_intraday(path):
    df = pd.read_csv(path, parse_dates=['time_utc'])
    df['time_utc'] = pd.to_datetime(df['time_utc'], utc=True)
    df = df.set_index('time_utc').sort_index()
    df = df[(df['Open'] > 0) & (df['High'] >= df['Low'])]
    df.index = df.index.tz_convert(NY)
    return df

def boot_ratio_p(a, b, n=5000):
    #置換檢定：事件平均/對照平均的比值，雙尾
    a, b = np.asarray(a), np.asarray(b)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b]); na = len(a); cnt = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(pool[:na].mean() - pool[na:].mean()) >= abs(obs):
            cnt += 1
    return (cnt + 1) / (n + 1)

# ---------- 公布日曆 ----------
cal = pd.read_csv(os.path.join(D, 'release_calendar.csv'), parse_dates=['date'])
cal = cal[(cal['date'] >= START) & (cal['date'] <= END)]
MAJOR830 = ['NFP', 'CPI', 'PPI', 'PCE', 'GDP_advance', 'Retail_sales']
major_days = set(cal[cal['release'].isin(MAJOR830 + ['GDP'])]['date'].dt.date)  #任何時間公布都排除在對照組外
at830 = cal['time_et'] == '08:30'
fomc_days = set(cal[cal['release'] == 'FOMC']['date'].dt.date)
claims_days = set(cal[cal['release'] == 'Jobless_claims']['date'].dt.date)

event_tests = []
event_detail = {}

def window_metrics(df, start_hm, end_hm):
    #每天在 [start, end) 美東時間窗口內：|收盤/開盤 對數報酬|（%）與 (高-低)/開盤（%）
    t = df.index
    hm = t.hour * 60 + t.minute
    s = int(start_hm[:2]) * 60 + int(start_hm[3:]); e = int(end_hm[:2]) * 60 + int(end_hm[3:])
    sub = df[(hm >= s) & (hm < e) & (t.dayofweek < 5)]
    g = sub.groupby(sub.index.date)
    expected = max(1, (e - s) // int(pd.Series(df.index).diff().dt.total_seconds().mode()[0] / 60))
    agg = pd.DataFrame({'o': g['Open'].first(), 'c': g['Close'].last(), 'h': g['High'].max(),
                        'l': g['Low'].min(), 'n': g.size()})
    agg = agg[agg['n'] >= expected]  #窗口內 K 棒要齊全
    agg['absret'] = np.abs(np.log(agg['c'] / agg['o'])) * 100
    agg['range'] = (agg['h'] - agg['l']) / agg['o'] * 100
    agg.index = pd.to_datetime(agg.index)
    return agg[(agg.index >= START) & (agg.index <= END)]

def compare(name, metric_df, ev_days, ctl_days, window, src, et_time, metric='range'):
    ev = metric_df[metric_df.index.map(lambda d: d.date() in ev_days)][metric]
    ct = metric_df[metric_df.index.map(lambda d: d.date() in ctl_days)][metric]
    rec = {'release': name, 'window': window, 'data': src, 'metric': metric,
           'n_event': int(len(ev)), 'n_control': int(len(ct))}
    if len(ev) >= 2 and len(ct) >= 2:
        rec.update(mean_event=round(ev.mean(), 4), mean_control=round(ct.mean(), 4),
                   median_event=round(ev.median(), 4), median_control=round(ct.median(), 4),
                   ratio=round(ev.mean() / ct.mean(), 3), median_ratio=round(ev.median() / ct.median(), 3),
                   p_welch=float(stats.ttest_ind(ev, ct, equal_var=False).pvalue),
                   p_mwu=float(stats.mannwhitneyu(ev, ct, alternative='two-sided').pvalue),
                   p_perm=float(boot_ratio_p(ev.values, ct.values, 3000)))
    else:
        rec.update(mean_event=None, mean_control=None, ratio=None, p_welch=None, p_mwu=None, p_perm=None)
    rec['taiwan_time'] = tw_time(et_time)
    rec['event_dates_used'] = [d.strftime('%Y-%m-%d') for d in ev.index]
    return rec

all_days = pd.bdate_range(START, END)
all_days_set = set(all_days.date)
ctl_830 = all_days_set - major_days - fomc_days  #對照組：當天沒有任何主要 8:30 數據、也不是 FOMC
ctl_830_noclaims = ctl_830 - claims_days

for sym in ['GC_F', 'MGC_F']:
    h1 = load_intraday(os.path.join(H, f'{sym}_1h.csv'))
    m830 = window_metrics(h1, '08:00', '09:00')   #含 8:30 公布的那根 1 小時 K 棒
    m830b = window_metrics(h1, '08:00', '10:00')  #延長到 10:00
    m14 = window_metrics(h1, '14:00', '15:00')    #FOMC 聲明 14:00、記者會 14:30
    for rel in MAJOR830 + ['GDP']:
        days = set(cal[(cal['release'] == rel) & at830]['date'].dt.date)
        for met in ['range', 'absret']:
            event_tests.append(compare(rel, m830, days, ctl_830, '08:00-09:00 ET (1h bar containing 08:30)', sym + ' 1h', '08:30', met))
        event_tests.append(compare(rel, m830b, days, ctl_830, '08:00-10:00 ET (2 x 1h bars)', sym + ' 1h', '08:30', 'range'))
    #只有初領失業金、沒有其他主要數據的星期四
    cl_only = claims_days - major_days - fomc_days
    for met in ['range', 'absret']:
        event_tests.append(compare('Jobless_claims_only', m830, cl_only, ctl_830_noclaims, '08:00-09:00 ET', sym + ' 1h', '08:30', met))
        event_tests.append(compare('FOMC', m14, fomc_days, all_days_set - fomc_days, '14:00-15:00 ET (statement 14:00, presser 14:30)', sym + ' 1h', '14:00', met))
        event_tests.append(compare('Any_major_830', m830, set(cal[cal['release'].isin(MAJOR830 + ['GDP']) & at830]['date'].dt.date), ctl_830, '08:00-09:00 ET', sym + ' 1h', '08:30', met))

#5 分鐘：只有 60 天，合併所有主要 8:30 數據
for sym in ['GC_F', 'MGC_F']:
    m5 = load_intraday(os.path.join(H, f'{sym}_5m.csv'))
    w5 = window_metrics(m5, '08:30', '08:35')
    w5b = window_metrics(m5, '08:30', '09:00')
    for met in ['range', 'absret']:
        event_tests.append(compare('Any_major_830', w5, set(cal[cal['release'].isin(MAJOR830 + ['GDP']) & at830]['date'].dt.date), ctl_830, '08:30-08:35 ET (first 5m bar)', sym + ' 5m', '08:30', met))
        event_tests.append(compare('Jobless_claims_only', w5, claims_days - major_days, ctl_830_noclaims, '08:30-08:35 ET', sym + ' 5m', '08:30', met))
    event_tests.append(compare('Any_major_830', w5b, set(cal[cal['release'].isin(MAJOR830 + ['GDP']) & at830]['date'].dt.date), ctl_830, '08:30-09:00 ET (6 x 5m)', sym + ' 5m', '08:30', 'range'))
    w5f = window_metrics(m5, '14:00', '14:05')
    event_tests.append(compare('FOMC', w5f, fomc_days, all_days_set - fomc_days, '14:00-14:05 ET', sym + ' 5m', '14:00', 'range'))

# ---------- 日資料迴歸 ----------
def yclose(name):
    df = pd.read_csv(os.path.join(D, f'yahoo_{name}_1d.csv'), parse_dates=['date']).set_index('date')
    return df['Close'].astype(float)

def fred(sid):
    df = pd.read_csv(os.path.join(D, f'fred_{sid}.csv'))
    df.columns = ['date', sid]
    df['date'] = pd.to_datetime(df['date'])
    df[sid] = pd.to_numeric(df[sid], errors='coerce')
    return df.set_index('date')[sid]

gc = yclose('gc')
gpr = pd.read_csv(os.path.join(D, 'gpr_data_gpr_daily_recent.csv'))
gpr['date'] = pd.to_datetime(gpr['date'] if 'date' in gpr else gpr['DAY'].astype(str), errors='coerce')
gpr = gpr.dropna(subset=['date']).set_index('date')['GPRD'].astype(float)

lvl = pd.DataFrame({'gold': gc, 'dxy': yclose('dxy'), 'vix': yclose('vix'), 'spx': yclose('spx'),
                    'tnx': yclose('tnx'), 'oil': yclose('cl'), 'silver': yclose('si'),
                    'dfii10': fred('DFII10'), 'dgs10': fred('DGS10'), 'dgs2': fred('DGS2'),
                    'broad_usd': fred('DTWEXBGS'), 't10yie': fred('T10YIE'), 'gpr': gpr})
lvl = lvl[lvl['gold'].notna()]  #以黃金交易日為準
lvl = lvl[(lvl.index >= START - pd.Timedelta(days=10)) & (lvl.index <= END)]
chg = pd.DataFrame(index=lvl.index)
chg['gold'] = np.log(lvl['gold']).diff() * 100
#只用「當天有新數值」的變化，前值補齊會把變化擠到下一天，所以先 dropna 再算差分再對回
def diff_valid(s, how):
    s = s.dropna()
    d = (np.log(s).diff() * 100) if how == 'pct' else (s.diff() * (100 if how == 'bp' else 1))
    return d
chg['dxy'] = diff_valid(lvl['dxy'], 'pct')
chg['broad_usd'] = diff_valid(lvl['broad_usd'], 'pct')
chg['dfii10'] = diff_valid(lvl['dfii10'], 'bp')
chg['dgs10'] = diff_valid(lvl['dgs10'], 'bp')
chg['dgs2'] = diff_valid(lvl['dgs2'], 'bp')
chg['t10yie'] = diff_valid(lvl['t10yie'], 'bp')
chg['vix'] = diff_valid(lvl['vix'], 'lvl')
chg['spx'] = diff_valid(lvl['spx'], 'pct')
chg['oil'] = diff_valid(lvl['oil'], 'pct')
chg['silver'] = diff_valid(lvl['silver'], 'pct')
chg['gpr'] = diff_valid(lvl['gpr'], 'pct')
chg = chg[chg.index >= START]
#剔除報酬極端值造成的數字問題（展期換月跳空）：|gold|>8% 標記
chg = chg.replace([np.inf, -np.inf], np.nan)

FACT = {
    'dxy': ('美元指數 DXY 日報酬 %', '每 +1% 美元指數，黃金同日 % 變化'),
    'broad_usd': ('Fed 廣義美元指數 DTWEXBGS 日報酬 %', '每 +1% 廣義美元，黃金同日 % 變化'),
    'dfii10': ('10 年期 TIPS 實質利率 DFII10 日變化（基點）', '實質利率每 +1bp，黃金同日 % 變化'),
    'dgs10': ('10 年期名目公債殖利率 DGS10 日變化（基點）', '名目利率每 +1bp，黃金同日 % 變化'),
    'dgs2': ('2 年期公債殖利率 DGS2 日變化（基點）', '2 年利率每 +1bp，黃金同日 % 變化'),
    't10yie': ('10 年期損益平衡通膨 T10YIE 日變化（基點）', '通膨預期每 +1bp，黃金同日 % 變化'),
    'vix': ('VIX 日變化（點）', 'VIX 每 +1 點，黃金同日 % 變化'),
    'spx': ('S&P 500 日報酬 %', 'S&P 每 +1%，黃金同日 % 變化'),
    'oil': ('WTI 原油期貨日報酬 %', '原油每 +1%，黃金同日 % 變化'),
    'gpr': ('地緣政治風險指數 GPRD 日變化 %', 'GPR 每 +1%，黃金同日 % 變化'),
    'silver': ('白銀期貨日報酬 %（對照，非驅動因素）', '白銀每 +1%，黃金同日 % 變化'),
}

def hac_ols(y, X, lags=5):
    X = sm.add_constant(X)
    return sm.OLS(y, X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': lags})

daily = []
for f, (label, meaning) in FACT.items():
    d = chg[['gold', f]].dropna()
    m = hac_ols(d['gold'], d[[f]])
    r, pr = stats.pearsonr(d['gold'], d[f])
    daily.append({'factor': f, 'label': label, 'coef': round(float(m.params[f]), 5),
                  't': round(float(m.tvalues[f]), 3), 'p': float(m.pvalues[f]),
                  'r2': round(float(m.rsquared), 4), 'corr': round(float(r), 3), 'n': int(m.nobs),
                  'grade': grade(m.pvalues[f]), 'sign_meaning': meaning,
                  'period': f'{d.index[0].date()} ~ {d.index[-1].date()}', 'se': 'Newey-West HAC, 5 lags'})

#穩健性：黃金日報酬 1%/99% 縮尾（避免 2026-01-30 -11% 等極端日主導）
lo, hi = chg['gold'].quantile([0.01, 0.99])
robust = []
for f in FACT:
    d = chg[['gold', f]].dropna().copy()
    d['gold'] = d['gold'].clip(lo, hi)
    m = hac_ols(d['gold'], d[[f]])
    robust.append({'factor': f, 'coef': round(float(m.params[f]), 5), 't': round(float(m.tvalues[f]), 3),
                   'p': float(m.pvalues[f]), 'r2': round(float(m.rsquared), 4), 'n': int(m.nobs), 'grade': grade(m.pvalues[f]),
                   'note': f'gold returns winsorized at {lo:.2f}% / {hi:.2f}%'})

#多變量（同時放入）：美元、實質利率、損益平衡通膨、VIX、S&P、GPR
multi_cols = ['dxy', 'dfii10', 't10yie', 'vix', 'spx', 'gpr']
dm = chg[['gold'] + multi_cols].dropna()
mm = hac_ols(dm['gold'], dm[multi_cols])
multivariate = {'factors': multi_cols, 'n': int(mm.nobs), 'r2': round(float(mm.rsquared), 4),
                'adj_r2': round(float(mm.rsquared_adj), 4),
                'coefs': {c: {'coef': round(float(mm.params[c]), 5), 't': round(float(mm.tvalues[c]), 3),
                              'p': float(mm.pvalues[c]), 'grade': grade(mm.pvalues[c])} for c in multi_cols},
                'note': '名目利率 = 實質利率 + 損益平衡通膨，所以 dgs10 不放入以免共線'}

#上下半段穩定性
halves = []
mid = chg.index[len(chg) // 2]
for f in FACT:
    for nm, part in [('first_half', chg[chg.index < mid]), ('second_half', chg[chg.index >= mid])]:
        d = part[['gold', f]].dropna()
        if len(d) > 30:
            m = hac_ols(d['gold'], d[[f]])
            halves.append({'factor': f, 'half': nm, 'period': f'{d.index[0].date()} ~ {d.index[-1].date()}',
                           'n': int(m.nobs), 'coef': round(float(m.params[f]), 5), 'p': float(m.pvalues[f]),
                           'grade': grade(m.pvalues[f])})

#滾動 60 日相關
rolling = []
for f in FACT:
    d = chg[['gold', f]].dropna()
    cs, ps = [], []
    for i in range(60, len(d) + 1):
        w = d.iloc[i - 60:i]
        r, p = stats.pearsonr(w['gold'], w[f])
        cs.append(r); ps.append(p)
    cs, ps = np.array(cs), np.array(ps)
    rolling.append({'factor': f, 'n_windows': int(len(cs)), 'window_days': 60,
                    'share_windows_p_lt_0_05': round(float((ps < 0.05).mean()), 3),
                    'median_corr': round(float(np.median(cs)), 3),
                    'min_corr': round(float(cs.min()), 3), 'max_corr': round(float(cs.max()), 3),
                    'share_negative': round(float((cs < 0).mean()), 3),
                    'latest_corr': round(float(cs[-1]), 3),
                    'latest_window_end': str(d.index[-1].date())})

#領先-落後：昨天因素變化 → 今天黃金報酬
leadlag = []
for f in [k for k in FACT if k != 'silver'] + ['gold_self']:
    if f == 'gold_self':
        d = pd.DataFrame({'y': chg['gold'], 'x': chg['gold'].shift(1)}).dropna()
        label = '黃金自己昨天的報酬（動能/反轉）'
    else:
        d = pd.DataFrame({'y': chg['gold'], 'x': chg[f].shift(1)}).dropna()
        label = FACT[f][0]
    m = hac_ols(d['y'], d[['x']])
    rec = {'factor': f, 'label': label, 'test': 'gold_t ~ factor_(t-1), HAC',
           'coef': round(float(m.params['x']), 5), 't': round(float(m.tvalues['x']), 3),
           'p': float(m.pvalues['x']), 'r2': round(float(m.rsquared), 4), 'n': int(m.nobs),
           'grade': grade(m.pvalues['x'])}
    if f != 'gold_self':
        g = chg[['gold', f]].dropna()
        try:
            gt = grangercausalitytests(g[['gold', f]].values, maxlag=5)
            rec['granger_factor_to_gold_p_by_lag'] = {str(int(L)): round(float(gt[L][0]['ssr_ftest'][1]), 4) for L in gt}
            gt2 = grangercausalitytests(g[[f, 'gold']].values, maxlag=5)
            rec['granger_gold_to_factor_p_by_lag'] = {str(int(L)): round(float(gt2[L][0]['ssr_ftest'][1]), 4) for L in gt2}
        except Exception as e:
            rec['granger_error'] = str(e)
    leadlag.append(rec)

#日線層級：公布日 vs 非公布日的黃金 |日報酬|
daily_event = []
absg = chg['gold'].abs().dropna()
for rel in MAJOR830 + ['GDP', 'FOMC']:
    days = set(cal[cal['release'] == rel]['date'].dt.date)
    nxt = days if rel != 'FOMC' else days | {(pd.Timestamp(x) + pd.offsets.BDay(1)).date() for x in days}
    ev = absg[absg.index.map(lambda d: d.date() in days)]
    ct = absg[absg.index.map(lambda d: d.date() not in (major_days | fomc_days))]
    if len(ev) > 2:
        daily_event.append({'release': rel, 'metric': '|gold daily log return| %', 'n_event': int(len(ev)),
                            'n_control': int(len(ct)), 'mean_event': round(ev.mean(), 4), 'mean_control': round(ct.mean(), 4),
                            'ratio': round(ev.mean() / ct.mean(), 3),
                            'p_welch': float(stats.ttest_ind(ev, ct, equal_var=False).pvalue),
                            'p_mwu': float(stats.mannwhitneyu(ev, ct, alternative='two-sided').pvalue)})

#COT 週資料：資產管理人（managed money）淨部位變化 vs 同週黃金報酬
cot_res = None
try:
    cot = pd.read_csv(os.path.join(D, 'cot_gold_disagg.csv'))
    cot['d'] = pd.to_datetime(cot['report_date_as_yyyy_mm_dd'])
    cot = cot.sort_values('d').drop_duplicates('d').set_index('d')
    net = (cot['m_money_positions_long_all'] - cot['m_money_positions_short_all']).astype(float)
    #COT 以週二為基準；黃金週報酬也取週二收盤到週二收盤
    gtue = gc.reindex(pd.date_range(gc.index.min(), gc.index.max())).ffill()
    wk = pd.DataFrame({'net': net, 'gold': gtue.reindex(net.index)}).dropna()
    wk['dnet_k'] = wk['net'].diff() / 1000
    wk['gret'] = np.log(wk['gold']).diff() * 100
    wk['gret_next'] = wk['gret'].shift(-1)
    wk = wk[(wk.index >= START) & (wk.index <= END)]
    w1 = wk[['gret', 'dnet_k']].dropna()
    m1 = hac_ols(w1['gret'], w1[['dnet_k']], lags=2)
    w2 = wk[['gret_next', 'dnet_k']].dropna()
    m2 = hac_ols(w2['gret_next'], w2[['dnet_k']], lags=2)
    cot_res = {'contemporaneous': {'test': 'weekly gold return (Tue-Tue) ~ change in managed-money net (thousand contracts)',
                                   'coef': round(float(m1.params['dnet_k']), 5), 't': round(float(m1.tvalues['dnet_k']), 3),
                                   'p': float(m1.pvalues['dnet_k']), 'r2': round(float(m1.rsquared), 4), 'n': int(m1.nobs),
                                   'grade': grade(m1.pvalues['dnet_k'])},
               'predictive': {'test': 'next-week gold return ~ this week change in managed-money net',
                              'coef': round(float(m2.params['dnet_k']), 5), 't': round(float(m2.tvalues['dnet_k']), 3),
                              'p': float(m2.pvalues['dnet_k']), 'r2': round(float(m2.rsquared), 4), 'n': int(m2.nobs),
                              'grade': grade(m2.pvalues['dnet_k'])},
               'period': f'{wk.index[0].date()} ~ {wk.index[-1].date()}',
               'latest_net_long_contracts': int(net.iloc[-1]), 'latest_report': str(net.index[-1].date())}
except Exception as e:
    cot_res = {'error': str(e)}

LIMITATIONS = [
    '沒有市場共識預期（consensus）資料，無法計算「實際值 - 預期值」的驚奇幅度，所以只檢定公布時段的波動大小，不能檢定方向（數據好/壞時黃金漲或跌）。',
    'Yahoo 1 小時 K 棒以整點切分，8:30 公布落在 8:00-9:00 那根，窗口含公布前 30 分鐘，會稀釋事件效果（偏向低估）。',
    '5 分鐘資料只有 2026-09-04 ~ 2026-11-13（Yahoo 60 天上限），事件數只有 11 個（合併所有主要數據）、FOMC 只有 2 次，不能分數據類型排名。',
    'GC=F / MGC=F 是連續近月合約，換月時可能出現價差跳空；日報酬有一天 -11.4%（2026-01-30），已用 1%/99% 縮尾做穩健性檢查，結論不變。',
    '日資料各市場收盤時間不同（黃金期貨、美元指數 ~17:00 ET；S&P/VIX 16:00；FRED 公債殖利率為收盤報價），同日相關可能部分是時間錯位造成。',
    '相關/迴歸只代表同時變動，不代表因果；美元與黃金可能同時被第三個因素（例如聯準會預期）推動。',
    '2024-11-13 ~ 2024-12-31 的經濟數據公布日期是依標準時程回憶填入（FRED 2024 日曆頁抓不到），信心較低；2025-01 之後來自 FRED 發布日曆（含政府停擺後的實際改期）。',
    '初領失業金每週四公布，對照組只能用其他平日，星期幾效應無法完全排除；2025-10 停擺期間全國週報未公布，已依 FRED 日曆略過。',
    '多次檢定：主要家族 33 個檢定做 BH 與 Bonferroni 校正；穩健性與分段檢定未納入校正，應視為輔助證據。',
    '只有約兩年（約 500 個交易日）資料，期間黃金經歷強勢多頭與劇烈波動，關係可能隨市場階段改變（上下半段比較顯示利率、S&P、VIX、GPR 的關係只在後半段顯著）。',
    'COT 為週資料且每週二結算、週五公布，同期相關不等於可用來交易；預測性檢定不顯著。',
    '統計顯著不等於可交易獲利：未考慮滑價、手續費、保證金，也未做樣本外驗證。',
]

# ---------- 多重檢定校正 ----------
#主要家族：GC_F 1h 的每種數據 range 檢定（p_welch）、FOMC、日迴歸單變量、領先落後、COT
family = []
for i, r in enumerate(event_tests):
    if r['data'] == 'GC_F 1h' and r['metric'] == 'range' and r['window'].startswith(('08:00-09:00', '14:00-15:00')) and r.get('p_welch') is not None:
        family.append(('event', i, r['p_welch']))
for i, r in enumerate(daily):
    if r['factor'] != 'silver':  #白銀只是對照組，不算驅動因素檢定
        family.append(('daily', i, r['p']))
for i, r in enumerate(leadlag):
    family.append(('leadlag', i, r['p']))
if cot_res and 'contemporaneous' in cot_res:
    family.append(('cot_c', 0, cot_res['contemporaneous']['p']))
    family.append(('cot_p', 0, cot_res['predictive']['p']))
pv = np.array([x[2] for x in family])
bh = multipletests(pv, alpha=0.05, method='fdr_bh')
bf = multipletests(pv, alpha=0.05, method='bonferroni')
for (kind, i, p), padj_bh, rej_bh, padj_bf, rej_bf in zip(family, bh[1], bh[0], bf[1], bf[0]):
    tgt = {'event': event_tests, 'daily': daily, 'leadlag': leadlag}.get(kind)
    rec = tgt[i] if tgt is not None else cot_res['contemporaneous' if kind == 'cot_c' else 'predictive']
    rec['p_adj_bh'] = float(padj_bh); rec['survives_bh_0_05'] = bool(rej_bh)
    rec['p_adj_bonferroni'] = float(padj_bf); rec['survives_bonferroni_0_05'] = bool(rej_bf)
for r in event_tests:
    r['p_adj'] = r.get('p_adj_bh')
    r['grade'] = grade(r.get('p_welch'))
    r['grade_mwu'] = grade(r.get('p_mwu'))

n_all = sum(1 for r in event_tests if r.get('p_welch') is not None) * 3 + len(daily) + len(leadlag) * 1 + len(halves) + len(daily_event) * 2 + len(multi_cols) + 2

meta = {
    'generated': pd.Timestamp.now('UTC').isoformat(),
    'periods': {'target': f'{START.date()} ~ {END.date()}',
                'daily_regression': daily[0]['period'],
                'intraday_1h': 'GC=F / MGC=F 1h, 2024-11-13 ~ 2026-11-13',
                'intraday_5m': 'GC=F / MGC=F 5m, 2026-09-04 ~ 2026-11-13 only (Yahoo 60-day limit)'},
    'data_sources': {
        'gold': 'Yahoo Finance GC=F / MGC=F（連續近月），1d/1h/5m',
        'dxy, vix, spx, tnx, oil, silver': 'Yahoo Finance DX-Y.NYB, ^VIX, ^GSPC, ^TNX, CL=F, SI=F',
        'rates': 'FRED DFII10, DGS10, DGS2, T10YIE, DTWEXBGS（fredgraph.csv）',
        'gpr': 'Caldara & Iacoviello daily GPR index (matteoiacoviello.com data_gpr_daily_recent.xls)',
        'cot': 'CFTC Public Reporting, Disaggregated Futures Only, gold 088691',
        'calendar': 'drivers/release_calendar.csv（Fed FOMC 行事曆、BEA 公布時程、FRED 發布日曆等，見 source 欄）'},
    'n_tests_primary_family': int(len(family)),
    'n_tests_all_including_robustness': int(n_all),
    'multiple_testing': 'Benjamini-Hochberg FDR and Bonferroni applied to primary family (GC_F 1h range event tests, univariate daily regressions, lead-lag tests, COT)',
    'grading': 'p<0.001 非常強; p<0.01 強; p<0.05 中等; p<0.10 弱; else 無證據（統計顯著不等於可交易獲利）',
}

rank = sorted([r for r in event_tests if r['data'] == 'GC_F 1h' and r['metric'] == 'range' and r['window'].startswith(('08:00-09:00', '14:00-15:00')) and r['release'] != 'Any_major_830' and r.get('ratio')],
              key=lambda r: -r['ratio'])
meta['event_ranking_gc_1h_range'] = [{'release': r['release'], 'ratio': r['ratio'], 'p_welch': r['p_welch'], 'p_mwu': r['p_mwu'], 'grade': r['grade'], 'survives_bh': r.get('survives_bh_0_05')} for r in rank]
res = {'meta': meta, 'event_tests': event_tests, 'daily_event_absret': daily_event,
       'daily_regressions': daily, 'multivariate': multivariate, 'daily_regressions_winsorized': robust, 'half_sample_stability': halves,
       'rolling': rolling, 'leadlag': leadlag, 'cot': cot_res, 'limitations': LIMITATIONS}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print('saved', OUT, 'primary family', len(family))
