#黃金（GC / MGC）價格驅動因素統計檢定：重現用程式
#輸入：同資料夾 data/ 底下的 CSV（說明見 README.md）
#輸出：results/results.json（全部數字）＋ results/*.csv（各檢定表）＋ 終端機摘要
#用法：python gold_stats.py            → 跑全部檢定
#      python gold_stats.py --quick    → 跳過置換檢定（快很多，其他數字不變）
import os, sys, json, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.multitest import multipletests

#常數定義
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
OUT = os.path.join(HERE, 'results')
START, END = pd.Timestamp('2024-11-13'), pd.Timestamp('2026-11-13')  #分析期間（兩年）
NY = 'America/New_York'
SEED = 20261113  #置換檢定的亂數種子，固定才會得到一樣的 p_perm
MAJOR830 = ['NFP', 'CPI', 'PPI', 'PCE', 'GDP_advance', 'Retail_sales']  #美東 08:30 公布的主要數據

#每個日資料因素：(變化計算方式, 說明)；pct = 對數報酬 %，bp = 基點，lvl = 原始點數差
FACTORS = {
    'dxy': ('pct', '美元指數 DXY 日報酬 %'),
    'broad_usd': ('pct', 'Fed 廣義美元指數 DTWEXBGS 日報酬 %'),
    'dfii10': ('bp', '10 年期 TIPS 實質利率 DFII10 日變化（基點）'),
    'dgs10': ('bp', '10 年期名目公債殖利率 DGS10 日變化（基點）'),
    'dgs2': ('bp', '2 年期公債殖利率 DGS2 日變化（基點）'),
    't10yie': ('bp', '10 年期損益平衡通膨 T10YIE 日變化（基點）'),
    'vix': ('lvl', 'VIX 日變化（點）'),
    'spx': ('pct', 'S&P 500 日報酬 %'),
    'oil': ('pct', 'WTI 原油期貨日報酬 %'),
    'gpr': ('pct', '地緣政治風險指數 GPRD 日變化 %'),
    'silver': ('pct', '白銀期貨日報酬 %（對照組，不算驅動因素）'),
}
MULTI_COLS = ['dxy', 'dfii10', 't10yie', 'vix', 'spx', 'gpr']  #多變量迴歸同時放入的因素

rng = np.random.default_rng(SEED)


# ====================================================================
# 一、通用統計函數
# ====================================================================

#證據力分級：p 值 → 中文描述
def grade(p):
    if p is None or not np.isfinite(p):
        return 'N/A'
    return '非常強' if p < 0.001 else '強' if p < 0.01 else '中等' if p < 0.05 else '弱' if p < 0.10 else '無證據'

#美東時間換成台灣時間（冬令 +13、夏令 +12）
def tw_time(et_hhmm):
    h, m = map(int, et_hhmm.split(':'))
    return f'台灣 {(h + 13) % 24:02d}:{m:02d}（美國冬令時間）／{(h + 12) % 24:02d}:{m:02d}（美國夏令時間）'

#置換檢定：把兩組混在一起隨機重分，看「平均差」至少這麼大的機率（雙尾）
def permutation_p(a, b, n=3000):
    a, b = np.asarray(a), np.asarray(b)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b]); na = len(a); cnt = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(pool[:na].mean() - pool[na:].mean()) >= abs(obs):
            cnt += 1
    return (cnt + 1) / (n + 1)

#OLS 迴歸＋Newey-West HAC 標準誤（處理自我相關與異質變異）
def hac_ols(y, X, lags=5):
    X = sm.add_constant(X)
    return sm.OLS(y, X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': lags})

#單一因素迴歸：y ~ 常數 + x，回傳係數、t、p、R²
def regress_one(y, x, lags=5):
    d = pd.DataFrame({'y': y, 'x': x}).dropna()
    m = hac_ols(d['y'], d[['x']], lags)
    return {'coef': round(float(m.params['x']), 5), 't': round(float(m.tvalues['x']), 3),
            'p': float(m.pvalues['x']), 'r2': round(float(m.rsquared), 4), 'n': int(m.nobs),
            'grade': grade(m.pvalues['x']), 'period': f'{d.index[0].date()} ~ {d.index[-1].date()}'}

#多重檢定校正：Benjamini-Hochberg（控制錯誤發現率）與 Bonferroni（最保守）
def adjust_pvalues(pvalues, alpha=0.05):
    pv = np.asarray(pvalues, dtype=float)
    bh = multipletests(pv, alpha=alpha, method='fdr_bh')
    bf = multipletests(pv, alpha=alpha, method='bonferroni')
    return pd.DataFrame({'p': pv, 'p_adj_bh': bh[1], 'survives_bh': bh[0],
                         'p_adj_bonferroni': bf[1], 'survives_bonferroni': bf[0]})


# ====================================================================
# 二、讀資料
# ====================================================================

#讀盤中 K 棒，轉成美東時間；剔除開盤價為 0 或高低顛倒的壞資料
def load_intraday(name):
    df = pd.read_csv(os.path.join(DATA, name))
    df['time_utc'] = pd.to_datetime(df['time_utc'], utc=True)
    df = df.set_index('time_utc').sort_index()
    df = df[(df['Open'] > 0) & (df['High'] >= df['Low'])]
    df.index = df.index.tz_convert(NY)
    return df

#讀公布日曆，整理出各類日期集合
def load_calendar():
    cal = pd.read_csv(os.path.join(DATA, 'release_calendar.csv'), parse_dates=['date'])
    cal = cal[(cal['date'] >= START) & (cal['date'] <= END)]
    days = {
        'major': set(cal[cal['release'].isin(MAJOR830 + ['GDP'])]['date'].dt.date),  #任何時段公布都排除在對照組外
        'fomc': set(cal[cal['release'] == 'FOMC']['date'].dt.date),
        'claims': set(cal[cal['release'] == 'Jobless_claims']['date'].dt.date),
    }
    all_days = set(pd.bdate_range(START, END).date)
    days['all'] = all_days
    days['ctl_830'] = all_days - days['major'] - days['fomc']  #對照組：沒有主要數據、也不是 FOMC 的平日
    days['ctl_830_noclaims'] = days['ctl_830'] - days['claims']
    return cal, days

#某類數據在 08:30 公布的日期
def release_days(cal, releases):
    releases = [releases] if isinstance(releases, str) else releases
    return set(cal[cal['release'].isin(releases) & (cal['time_et'] == '08:30')]['date'].dt.date)

#讀日資料水準值，並算出每日變化（只用「當天真的有新數值」的日子做差分）
def load_daily_changes():
    lvl = pd.read_csv(os.path.join(DATA, 'daily_levels.csv'), parse_dates=['date']).set_index('date')
    lvl = lvl[(lvl.index >= START - pd.Timedelta(days=10)) & (lvl.index <= END)]
    chg = pd.DataFrame(index=lvl.index)
    chg['gold'] = np.log(lvl['gold']).diff() * 100
    for f, (how, _) in FACTORS.items():
        s = lvl[f].dropna()
        chg[f] = (np.log(s).diff() * 100) if how == 'pct' else (s.diff() * (100 if how == 'bp' else 1))
    chg = chg[chg.index >= START].replace([np.inf, -np.inf], np.nan)
    return lvl, chg


# ====================================================================
# 三、事件檢定（公布時段的波動是否比平常大）
# ====================================================================

#每天在美東 [start, end) 時段內算兩個指標：|收盤/開盤 對數報酬| % 與（高-低）/開盤 %
def window_metrics(df, start_hm, end_hm):
    t = df.index
    hm = t.hour * 60 + t.minute
    s = int(start_hm[:2]) * 60 + int(start_hm[3:]); e = int(end_hm[:2]) * 60 + int(end_hm[3:])
    sub = df[(hm >= s) & (hm < e) & (t.dayofweek < 5)]
    g = sub.groupby(sub.index.date)
    bar_min = int(pd.Series(df.index).diff().dt.total_seconds().mode()[0] / 60)  #K 棒間隔（分鐘）
    expected = max(1, (e - s) // bar_min)
    agg = pd.DataFrame({'o': g['Open'].first(), 'c': g['Close'].last(), 'h': g['High'].max(),
                        'l': g['Low'].min(), 'n': g.size()})
    agg = agg[agg['n'] >= expected]  #窗口內 K 棒要齊全
    agg['absret'] = np.abs(np.log(agg['c'] / agg['o'])) * 100
    agg['range'] = (agg['h'] - agg['l']) / agg['o'] * 100
    agg.index = pd.to_datetime(agg.index)
    return agg[(agg.index >= START) & (agg.index <= END)]

#事件日 vs 對照日：平均比值、Welch t 檢定、Mann-Whitney U、置換檢定
def compare_event(name, metric_df, ev_days, ctl_days, window, src, et_time, metric='range', do_perm=True):
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
                   p_perm=float(permutation_p(ev.values, ct.values, 3000)) if do_perm else None)
    else:
        rec.update(mean_event=None, mean_control=None, ratio=None, p_welch=None, p_mwu=None, p_perm=None)
    rec['taiwan_time'] = tw_time(et_time)
    rec['event_dates_used'] = [d.strftime('%Y-%m-%d') for d in ev.index]
    return rec

#跑全部事件檢定（順序固定，置換檢定的亂數才會一致）
def run_event_tests(cal, days, do_perm=True):
    out = []
    any830 = release_days(cal, MAJOR830 + ['GDP'])
    #1 小時 K 棒：GC（主要）與 MGC（穩健性）
    for sym in ['GC', 'MGC']:
        h1 = load_intraday(f'gold_1h_{sym}.csv')
        src = f'{sym}_F 1h'
        m830 = window_metrics(h1, '08:00', '09:00')   #含 08:30 公布的那根 1 小時 K 棒
        m830b = window_metrics(h1, '08:00', '10:00')  #延長到 10:00
        m14 = window_metrics(h1, '14:00', '15:00')    #FOMC 聲明 14:00、記者會 14:30
        for rel in MAJOR830 + ['GDP']:
            ev = release_days(cal, rel)
            for met in ['range', 'absret']:
                out.append(compare_event(rel, m830, ev, days['ctl_830'], '08:00-09:00 ET (1h bar containing 08:30)', src, '08:30', met, do_perm))
            out.append(compare_event(rel, m830b, ev, days['ctl_830'], '08:00-10:00 ET (2 x 1h bars)', src, '08:30', 'range', do_perm))
        cl_only = days['claims'] - days['major'] - days['fomc']  #只有初領失業金的星期四
        for met in ['range', 'absret']:
            out.append(compare_event('Jobless_claims_only', m830, cl_only, days['ctl_830_noclaims'], '08:00-09:00 ET', src, '08:30', met, do_perm))
            out.append(compare_event('FOMC', m14, days['fomc'], days['all'] - days['fomc'], '14:00-15:00 ET (statement 14:00, presser 14:30)', src, '14:00', met, do_perm))
            out.append(compare_event('Any_major_830', m830, any830, days['ctl_830'], '08:00-09:00 ET', src, '08:30', met, do_perm))
    #5 分鐘 K 棒：只有最近 60 天，合併所有主要數據
    for sym in ['GC', 'MGC']:
        m5 = load_intraday(f'gold_5m_{sym}.csv')
        src = f'{sym}_F 5m'
        w5 = window_metrics(m5, '08:30', '08:35')
        w5b = window_metrics(m5, '08:30', '09:00')
        for met in ['range', 'absret']:
            out.append(compare_event('Any_major_830', w5, any830, days['ctl_830'], '08:30-08:35 ET (first 5m bar)', src, '08:30', met, do_perm))
            out.append(compare_event('Jobless_claims_only', w5, days['claims'] - days['major'], days['ctl_830_noclaims'], '08:30-08:35 ET', src, '08:30', met, do_perm))
        out.append(compare_event('Any_major_830', w5b, any830, days['ctl_830'], '08:30-09:00 ET (6 x 5m)', src, '08:30', 'range', do_perm))
        w5f = window_metrics(m5, '14:00', '14:05')
        out.append(compare_event('FOMC', w5f, days['fomc'], days['all'] - days['fomc'], '14:00-14:05 ET', src, '14:00', 'range', do_perm))
    return out

#日線層級：公布日 vs 非公布日的黃金 |日報酬|
def daily_event_absret(chg, cal, days):
    out = []
    absg = chg['gold'].abs().dropna()
    for rel in MAJOR830 + ['GDP', 'FOMC']:
        ev_days = set(cal[cal['release'] == rel]['date'].dt.date)
        ev = absg[absg.index.map(lambda d: d.date() in ev_days)]
        ct = absg[absg.index.map(lambda d: d.date() not in (days['major'] | days['fomc']))]
        if len(ev) > 2:
            out.append({'release': rel, 'metric': '|gold daily log return| %', 'n_event': int(len(ev)),
                        'n_control': int(len(ct)), 'mean_event': round(ev.mean(), 4), 'mean_control': round(ct.mean(), 4),
                        'ratio': round(ev.mean() / ct.mean(), 3),
                        'p_welch': float(stats.ttest_ind(ev, ct, equal_var=False).pvalue),
                        'p_mwu': float(stats.mannwhitneyu(ev, ct, alternative='two-sided').pvalue)})
    return out


# ====================================================================
# 四、日資料關係（黃金日報酬 vs 各因素日變化）
# ====================================================================

#單變量 HAC 迴歸：黃金同日報酬 ~ 因素同日變化
def daily_regressions(chg):
    out = []
    for f, (_, label) in FACTORS.items():
        d = chg[['gold', f]].dropna()
        rec = {'factor': f, 'label': label, **regress_one(d['gold'], d[f])}
        rec['corr'] = round(float(stats.pearsonr(d['gold'], d[f])[0]), 3)
        out.append(rec)
    return out

#穩健性：黃金日報酬 1% / 99% 縮尾，避免少數極端日主導結果
def winsorized_regressions(chg):
    lo, hi = chg['gold'].quantile([0.01, 0.99])
    out = []
    for f in FACTORS:
        d = chg[['gold', f]].dropna().copy()
        d['gold'] = d['gold'].clip(lo, hi)
        out.append({'factor': f, **regress_one(d['gold'], d[f]), 'note': f'gold winsorized at {lo:.2f}% / {hi:.2f}%'})
    return out

#多變量 HAC 迴歸：同時放入多個因素，看誰在控制其他因素後還顯著
def multivariate(chg, cols=MULTI_COLS):
    d = chg[['gold'] + cols].dropna()
    m = hac_ols(d['gold'], d[cols])
    return {'factors': cols, 'n': int(m.nobs), 'r2': round(float(m.rsquared), 4), 'adj_r2': round(float(m.rsquared_adj), 4),
            'coefs': {c: {'coef': round(float(m.params[c]), 5), 't': round(float(m.tvalues[c]), 3),
                          'p': float(m.pvalues[c]), 'grade': grade(m.pvalues[c])} for c in cols}}

#前後半段穩定性：關係是否只在某一段期間成立
def half_sample(chg):
    out = []
    mid = chg.index[len(chg) // 2]
    for f in FACTORS:
        for nm, part in [('first_half', chg[chg.index < mid]), ('second_half', chg[chg.index >= mid])]:
            d = part[['gold', f]].dropna()
            if len(d) > 30:
                r = regress_one(d['gold'], d[f])
                out.append({'factor': f, 'half': nm, 'period': r['period'], 'n': r['n'],
                            'coef': r['coef'], 'p': r['p'], 'grade': r['grade']})
    return out

#滾動相關：每 60 個交易日算一次相關係數，看關係多穩定
def rolling_corr(chg, window=60):
    out = []
    for f in FACTORS:
        d = chg[['gold', f]].dropna()
        cs, ps = [], []
        for i in range(window, len(d) + 1):
            w = d.iloc[i - window:i]
            r, p = stats.pearsonr(w['gold'], w[f])
            cs.append(r); ps.append(p)
        cs, ps = np.array(cs), np.array(ps)
        out.append({'factor': f, 'n_windows': int(len(cs)), 'window_days': window,
                    'share_windows_p_lt_0_05': round(float((ps < 0.05).mean()), 3),
                    'median_corr': round(float(np.median(cs)), 3),
                    'min_corr': round(float(cs.min()), 3), 'max_corr': round(float(cs.max()), 3),
                    'share_negative': round(float((cs < 0).mean()), 3),
                    'latest_corr': round(float(cs[-1]), 3), 'latest_window_end': str(d.index[-1].date())})
    return out

#領先-落後：昨天的因素變化能不能預測今天的黃金報酬（另附 Granger 因果檢定 1~5 期）
def lead_lag(chg, maxlag=5):
    out = []
    for f in [k for k in FACTORS if k != 'silver'] + ['gold_self']:
        x = chg['gold'].shift(1) if f == 'gold_self' else chg[f].shift(1)
        label = '黃金自己昨天的報酬（動能/反轉）' if f == 'gold_self' else FACTORS[f][1]
        rec = {'factor': f, 'label': label, 'test': 'gold_t ~ factor_(t-1), HAC', **regress_one(chg['gold'], x)}
        if f != 'gold_self':
            g = chg[['gold', f]].dropna()
            gt = granger(g[['gold', f]].values, maxlag)
            rec['granger_factor_to_gold_p_by_lag'] = {str(int(L)): round(float(gt[L][0]['ssr_ftest'][1]), 4) for L in gt}
            gt2 = granger(g[[f, 'gold']].values, maxlag)
            rec['granger_gold_to_factor_p_by_lag'] = {str(int(L)): round(float(gt2[L][0]['ssr_ftest'][1]), 4) for L in gt2}
        out.append(rec)
    return out

#Granger 因果檢定：第二欄是否幫助預測第一欄；舊版 statsmodels 預設會印一大堆表格，所以有 verbose 參數就關掉
def granger(values, maxlag):
    import inspect
    if 'verbose' in inspect.signature(grangercausalitytests).parameters:
        return grangercausalitytests(values, maxlag=maxlag, verbose=False)
    return grangercausalitytests(values, maxlag=maxlag)


# ====================================================================
# 五、COT 持倉（每週）
# ====================================================================

#資產管理人淨多單週變化 vs 同週／下週黃金報酬（週二收盤對週二收盤）
def cot_tests(lvl_full):
    cot = pd.read_csv(os.path.join(DATA, 'cot_gold_managed_money.csv'), parse_dates=['report_date']).set_index('report_date')
    net = (cot['mm_long'] - cot['mm_short']).astype(float)
    gold = lvl_full['gold'].dropna()
    gold_daily = gold.reindex(pd.date_range(gold.index.min(), gold.index.max())).ffill()  #週二休市就用前一天收盤
    wk = pd.DataFrame({'net': net, 'gold': gold_daily.reindex(net.index)}).dropna()
    wk['dnet_k'] = wk['net'].diff() / 1000  #淨部位變化（千口）
    wk['gret'] = np.log(wk['gold']).diff() * 100
    wk['gret_next'] = wk['gret'].shift(-1)
    wk = wk[(wk.index >= START) & (wk.index <= END)]
    same = regress_one(wk['gret'], wk['dnet_k'], lags=2)
    nxt = regress_one(wk['gret_next'], wk['dnet_k'], lags=2)
    same['test'] = 'weekly gold return (Tue-Tue) ~ change in managed-money net (thousand contracts)'
    nxt['test'] = 'next-week gold return ~ this week change in managed-money net'
    return {'contemporaneous': same, 'predictive': nxt, 'period': f'{wk.index[0].date()} ~ {wk.index[-1].date()}',
            'latest_net_long_contracts': int(net.iloc[-1]), 'latest_report': str(net.index[-1].date())}


# ====================================================================
# 六、主流程
# ====================================================================

#主要檢定家族（33 個）做多重檢定校正，結果寫回各筆紀錄
def apply_family_correction(res):
    family = []
    for r in res['event_tests']:
        if r['data'] == 'GC_F 1h' and r['metric'] == 'range' and r['window'].startswith(('08:00-09:00', '14:00-15:00')) and r.get('p_welch') is not None:
            family.append((r, r['p_welch']))
    for r in res['daily_regressions']:
        if r['factor'] != 'silver':
            family.append((r, r['p']))
    for r in res['leadlag']:
        family.append((r, r['p']))
    family.append((res['cot']['contemporaneous'], res['cot']['contemporaneous']['p']))
    family.append((res['cot']['predictive'], res['cot']['predictive']['p']))
    adj = adjust_pvalues([p for _, p in family])
    for (rec, _), row in zip(family, adj.itertuples()):
        rec['p_adj_bh'] = float(row.p_adj_bh); rec['survives_bh_0_05'] = bool(row.survives_bh)
        rec['p_adj_bonferroni'] = float(row.p_adj_bonferroni); rec['survives_bonferroni_0_05'] = bool(row.survives_bonferroni)
    for r in res['event_tests']:
        r['grade'] = grade(r.get('p_welch'))
        r['grade_mwu'] = grade(r.get('p_mwu'))
    return len(family)

#跑全部檢定
def run_all(do_perm=True):
    cal, days = load_calendar()
    lvl, chg = load_daily_changes()
    lvl_full = pd.read_csv(os.path.join(DATA, 'daily_levels.csv'), parse_dates=['date']).set_index('date')
    res = {}
    res['event_tests'] = run_event_tests(cal, days, do_perm)
    res['daily_event_absret'] = daily_event_absret(chg, cal, days)
    res['daily_regressions'] = daily_regressions(chg)
    res['multivariate'] = multivariate(chg)
    res['daily_regressions_winsorized'] = winsorized_regressions(chg)
    res['half_sample_stability'] = half_sample(chg)
    res['rolling'] = rolling_corr(chg)
    res['leadlag'] = lead_lag(chg)
    res['cot'] = cot_tests(lvl_full)
    n_family = apply_family_correction(res)
    res['meta'] = {'period': f'{START.date()} ~ {END.date()}', 'seed': SEED, 'permutation': do_perm,
                   'n_tests_primary_family': n_family,
                   'grading': 'p<0.001 非常強; p<0.01 強; p<0.05 中等; p<0.10 弱; else 無證據'}
    return res

#把結果存成 JSON 與幾張 CSV 表
def save_results(res):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    ev = pd.DataFrame(res['event_tests']).drop(columns=['event_dates_used'])
    ev.to_csv(os.path.join(OUT, 'event_tests.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(res['daily_regressions']).to_csv(os.path.join(OUT, 'daily_regressions.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(res['daily_regressions_winsorized']).to_csv(os.path.join(OUT, 'daily_regressions_winsorized.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(res['half_sample_stability']).to_csv(os.path.join(OUT, 'half_sample.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(res['rolling']).to_csv(os.path.join(OUT, 'rolling_corr.csv'), index=False, encoding='utf-8-sig')
    ll = pd.DataFrame(res['leadlag']).drop(columns=['granger_factor_to_gold_p_by_lag', 'granger_gold_to_factor_p_by_lag'], errors='ignore')
    ll.to_csv(os.path.join(OUT, 'leadlag.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(res['daily_event_absret']).to_csv(os.path.join(OUT, 'daily_event_absret.csv'), index=False, encoding='utf-8-sig')

#終端機摘要：重點數字，方便跟 README 的「預期結果」對照
def print_summary(res):
    print('\n=== 事件檢定（GC_F 1h，高低幅度 range）===')
    for r in res['event_tests']:
        if r['data'] == 'GC_F 1h' and r['metric'] == 'range' and r['window'].startswith(('08:00-09:00', '14:00-15:00')):
            print(f"{r['release']:<20} n={r['n_event']:>3}/{r['n_control']:<3} 倍數={r['ratio']:<6} p_welch={r['p_welch']:.2g}  "
                  f"BH校正後={r.get('p_adj_bh', float('nan')):.2g}  {r['grade']}")
    print('\n=== 日資料單變量迴歸（Newey-West HAC，5 期）===')
    for r in res['daily_regressions']:
        print(f"{r['factor']:<10} 係數={r['coef']:<9} t={r['t']:<7} p={r['p']:.3g}  R²={r['r2']:<6} {r['grade']}")
    mv = res['multivariate']
    print(f"\n=== 多變量迴歸 R²={mv['r2']} n={mv['n']} ===")
    for c, v in mv['coefs'].items():
        print(f"{c:<8} 係數={v['coef']:<9} p={v['p']:.3g}  {v['grade']}")
    print('\n=== 領先-落後（昨天因素 → 今天黃金）===')
    for r in res['leadlag']:
        print(f"{r['factor']:<10} p={r['p']:.3g}  BH校正後={r.get('p_adj_bh', float('nan')):.2g}  {r['grade']}")
    c = res['cot']
    print(f"\n=== COT 同週 p={c['contemporaneous']['p']:.3g}（{c['contemporaneous']['grade']}）  下週預測 p={c['predictive']['p']:.3g}（{c['predictive']['grade']}）===")
    print(f"\n主要家族檢定數 = {res['meta']['n_tests_primary_family']}；結果存在 {OUT}")

if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')  #Windows 終端機顯示中文
    except Exception:
        pass
    ap = argparse.ArgumentParser(description='黃金價格驅動因素統計檢定（重現用）')
    ap.add_argument('--quick', action='store_true', help='跳過置換檢定（p_perm 會是空值，其他數字不變）')
    args = ap.parse_args()
    res = run_all(do_perm=not args.quick)
    save_results(res)
    print_summary(res)
