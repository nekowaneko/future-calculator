# 黃金價格驅動因素檢定：重現包

這個資料夾讓你在自己的電腦上，從原始資料重新算出「第 2 層：價格的驅動因素」報告裡的每一個統計數字。

- 分析期間：2024-11-13 ~ 2026-11-13（約兩年、490 個交易日）
- 標的：COMEX 黃金期貨 GC=F（主要），以及微型黃金 MGC=F（穩健性檢查）
- 已驗證：本程式輸出和報告使用的結果逐一比對，**1,310 個數字完全相同**，包含置換檢定的 p 值（亂數種子固定）

> 不是投資建議。統計顯著不代表扣掉手續費與滑價後交易會賺錢。

---

## 1. 資料夾內容

```
repro/
├─ README.md              ← 本檔
├─ gold_stats.py          ← 主程式：所有統計函數＋一鍵執行
├─ build_data.py          ← （參考用）從原始下載檔整理出 data/ 的程式
├─ data/                  ← 輸入資料（7 個 CSV）
│  ├─ daily_levels.csv
│  ├─ release_calendar.csv
│  ├─ cot_gold_managed_money.csv
│  ├─ gold_1h_GC.csv   gold_1h_MGC.csv
│  └─ gold_5m_GC.csv   gold_5m_MGC.csv
└─ results/               ← 執行後產生（輸出）
   ├─ results.json        ← 全部數字
   └─ event_tests.csv、daily_regressions.csv …（各檢定表，Excel 可直接開）
```

---

## 2. 怎麼執行

### 2.1 安裝套件（只需一次）

```bat
c:\Python36\python.exe -m pip install pandas numpy scipy statsmodels
```

### 2.2 執行

在 `repro` 資料夾裡開命令提示字元：

```bat
cd 路徑\repro
c:\Python36\python.exe gold_stats.py
```

大約 10 秒。想更快可加 `--quick`，會跳過置換檢定（`p_perm` 欄位變空白，其他數字不變）。

### 2.3 預期結果（對照用）

終端機會印出摘要，重點數字應該是：

| 檢定 | 你應該看到 |
|---|---|
| NFP 公布那 1 小時的高低幅度 | 平常的 **1.99 倍**，p_welch ≈ 0.00011，BH 校正後 ≈ 0.00071 |
| CPI | **1.49 倍**，p_welch ≈ 0.0069 |
| FOMC 14:00–15:00 | **3.506 倍**，p_welch ≈ 7.5e-05 |
| 任何主要 08:30 數據合併 | **1.389 倍**，p_welch ≈ 5.2e-06 |
| 黃金 vs 廣義美元（日報酬） | 係數 **−2.29541**，t = −7.936，p ≈ 2.08e-15，R² = 0.1916 |
| 黃金 vs 美元指數 DXY | 係數 −1.38085，p ≈ 5.17e-10 |
| 黃金 vs 實質利率 DFII10 | 係數 −0.04291（每 1bp），p ≈ 0.027 |
| 多變量迴歸 | R² = 0.1648，DXY p ≈ 2.15e-09，實質利率 p ≈ 0.859 |
| 領先-落後 | 只有 VIX p ≈ 0.0139，BH 校正後 0.051（沒通過） |
| COT 同週／下週 | p ≈ 0.0179／0.616 |
| 主要家族檢定數 | 33 |

最後幾位小數若跟上表不同，通常是套件版本差異（驗證環境：Python 3.13、pandas 3.0.5、numpy 2.5.3、scipy 1.18.1、statsmodels 0.15.0）。

### 2.4 單獨呼叫某個函數

```python
#在 repro 資料夾裡開 python
import gold_stats as gs

cal, days = gs.load_calendar()           #公布日曆與各類日期集合
lvl, chg = gs.load_daily_changes()       #日資料水準值與每日變化

#黃金 vs 廣義美元的單變量 HAC 迴歸
r = gs.regress_one(chg['gold'], chg['broad_usd'])
print(r['coef'], r['p'], r['grade'])

#NFP 那 1 小時的高低幅度 vs 平常
h1 = gs.load_intraday('gold_1h_GC.csv')
m = gs.window_metrics(h1, '08:00', '09:00')
nfp = gs.release_days(cal, 'NFP')
print(gs.compare_event('NFP', m, nfp, days['ctl_830'], '08:00-09:00', 'GC_F 1h', '08:30', 'range', do_perm=False))

#自己挑一組 p 值做多重檢定校正
print(gs.adjust_pvalues([0.0001, 0.01, 0.04, 0.2]))
```

---

## 3. 資料字典

所有時間都標明時區。盤中 K 棒是 UTC，程式讀入後轉成美東時間（America/New_York，會自動處理夏令時間）。台灣時間 = 美東 +13（美國冬令）或 +12（美國夏令）。

### 3.1 `daily_levels.csv`：日資料水準值

一列 = 一個黃金期貨交易日（2024-10-01 ~ 2026-11-13，535 列）。其他市場那天休市或還沒公布就留空白。

| 欄位 | 內容 | 單位 | 來源代號 |
|---|---|---|---|
| date | 日期 | | |
| gold | COMEX 黃金期貨連續近月收盤 | 美元／盎司 | Yahoo `GC=F` |
| dxy | 美元指數 | 點 | Yahoo `DX-Y.NYB` |
| broad_usd | Fed 名目廣義美元指數 | 指數（2006-01=100） | FRED `DTWEXBGS` |
| dfii10 | 10 年期抗通膨公債（TIPS）殖利率＝實質利率 | % | FRED `DFII10` |
| dgs10 | 10 年期公債殖利率 | % | FRED `DGS10` |
| dgs2 | 2 年期公債殖利率 | % | FRED `DGS2` |
| t10yie | 10 年期損益平衡通膨率 | % | FRED `T10YIE` |
| vix | CBOE 波動率指數 | 點 | Yahoo `^VIX` |
| spx | S&P 500 指數 | 點 | Yahoo `^GSPC` |
| oil | WTI 原油期貨連續近月 | 美元／桶 | Yahoo `CL=F` |
| silver | 白銀期貨連續近月（對照組） | 美元／盎司 | Yahoo `SI=F` |
| gpr | 每日地緣政治風險指數 GPRD | 指數 | Caldara & Iacoviello |

程式怎麼算「每日變化」：價格類取對數報酬 ×100（%），利率類取差 ×100（基點），VIX 取點數差。只用「當天真的有新數值」的日子做差分，不用前值補齊，以免把變化擠到下一天。

### 3.2 `release_calendar.csv`：經濟數據公布日曆

| 欄位 | 內容 |
|---|---|
| date | 公布日期（美東） |
| release | NFP 非農就業、CPI、PPI、PCE、GDP、GDP_advance（GDP 初估）、Retail_sales 零售銷售、Jobless_claims 初領失業金、FOMC 利率決策 |
| time_et | 美東公布時間（08:30 或 14:00 等） |
| source | 這筆日期的出處 |
| fred_release_name | FRED 上的發布名稱 |

出處：2025-01 之後來自 FRED 發布日曆（含政府停擺後的實際改期）；FOMC 來自聯準會官網會議行事曆（聲明日＝會議第 2 天）；**2024-11 ~ 2024-12 共 18 筆是依標準時程回填，信心較低**（source 欄有標註）。

### 3.3 `cot_gold_managed_money.csv`：CFTC 持倉報告（每週）

| 欄位 | 內容 |
|---|---|
| report_date | 持倉基準日（週二；報告在週五公布） |
| open_interest | 全市場未平倉量（口） |
| mm_long／mm_short | 資產管理人（managed money）多單／空單（口） |

市場：COMEX 黃金（CFTC 代碼 088691），Disaggregated Futures Only 報告。

### 3.4 `gold_1h_*.csv`、`gold_5m_*.csv`：盤中 K 棒

| 欄位 | 內容 |
|---|---|
| time_utc | K 棒開始時間（UTC） |
| Open／High／Low／Close | 開高低收（美元／盎司） |
| Volume | 成交量（口） |

`GC` = Yahoo `GC=F`，`MGC` = Yahoo `MGC=F`（都是連續近月）。1 小時資料涵蓋 2024-10-01 起；**5 分鐘資料只有 2026-09-04 起**（Yahoo 限制最多 60 天），所以 5 分鐘檢定的事件數很少。

---

## 4. 函數一覽與方法

| 函數 | 做什麼 | 對應報告的哪一段 |
|---|---|---|
| `grade(p)` | p 值換成證據力：<0.001 非常強、<0.01 強、<0.05 中等、<0.10 弱，其他為無證據 | 全部 |
| `permutation_p(a, b)` | 置換檢定：兩組隨機重分 3,000 次，看平均差至少這麼大的機率 | 事件檢定 |
| `hac_ols(y, X)`／`regress_one(y, x)` | OLS 迴歸，標準誤用 Newey-West HAC（5 期落後），處理自我相關與異質變異 | 日資料迴歸 |
| `adjust_pvalues(p)` | Benjamini-Hochberg 與 Bonferroni 多重檢定校正 | 校正 |
| `window_metrics(df, 起, 迄)` | 每天某個美東時段的「高低幅度 %」與「\|報酬\| %」；K 棒不齊全的日子剔除 | 事件檢定 |
| `compare_event(...)` | 事件日 vs 對照日：平均倍數、Welch t、Mann-Whitney U、置換檢定 | 事件檢定 |
| `run_event_tests(...)` | 跑全部 66 個事件檢定（1 小時與 5 分鐘、GC 與 MGC、兩種指標） | 事件檢定 |
| `daily_event_absret(...)` | 日線層級：公布日 vs 非公布日的 \|日報酬\| | 補充 |
| `daily_regressions(chg)` | 黃金同日報酬 ~ 各因素同日變化（11 個單變量迴歸） | 日資料迴歸 |
| `winsorized_regressions(chg)` | 黃金報酬 1%／99% 縮尾後重跑，排除極端日的影響 | 穩健性 |
| `multivariate(chg)` | 同時放入美元、實質利率、通膨預期、VIX、S&P、GPR | 多變量 |
| `half_sample(chg)` | 前後半段各跑一次，看關係穩不穩定 | 穩健性 |
| `rolling_corr(chg)` | 60 日滾動相關係數的分布 | 穩定性 |
| `lead_lag(chg)` | 昨天的因素變化 → 今天黃金報酬，另附 Granger 因果檢定 1~5 期 | 領先-落後 |
| `cot_tests(lvl)` | 資產管理人淨多單週變化 vs 同週／下週黃金報酬（HAC 2 期） | COT |
| `apply_family_correction(res)` | 主要家族 33 個檢定一起做 BH 與 Bonferroni 校正 | 校正 |
| `run_all()`／`save_results()`／`print_summary()` | 一鍵執行、存檔、印摘要 | |

**對照組怎麼選**：08:30 事件的對照組是「沒有任何主要數據、也不是 FOMC 的平日」；初領失業金的對照組另外排除所有星期四的初領日；FOMC 的對照組是其他所有平日。

**主要家族（33 個檢定，有做多重校正）**：GC=F 1 小時高低幅度的 10 個事件檢定、10 個日資料單變量迴歸（白銀是對照組，不算）、11 個領先-落後檢定、2 個 COT 檢定。其餘（MGC、絕對報酬、2 小時窗口、5 分鐘、縮尾、前後半段）都是穩健性檢查，沒有納入校正，應該當成輔助證據。

---

## 5. 已知限制

1. 沒有市場共識預期資料，算不出「實際值 − 預期值」的驚奇幅度，所以只能檢定公布時段的**波動大小**，不能檢定方向。
2. 1 小時 K 棒以整點切分，08:30 的公布落在 08:00–09:00 那根，包含公布前 30 分鐘，會稀釋事件效果（偏向低估）。
3. 5 分鐘資料只有約 60 天，事件數很少。
4. GC=F／MGC=F 是連續近月合約，換月可能跳空；有一天日報酬 −11.4%（2026-01-30），已用縮尾做穩健性檢查。
5. 各市場收盤時間不同（黃金、美元約 17:00 ET；S&P、VIX 16:00），同日相關可能有一部分是時間錯位。
6. 相關與迴歸只代表同時變動，不代表因果。
7. 只有約兩年資料，關係可能隨市場階段改變。

---

## 6. 資料來源與參考文獻（APA 第 7 版）

### 6.1 資料來源

Board of Governors of the Federal Reserve System (US). (2026a). *Market yield on U.S. Treasury securities at 2-year constant maturity, quoted on an investment basis* [DGS2] [Data set]. FRED, Federal Reserve Bank of St. Louis. Retrieved November 13, 2026, from https://fred.stlouisfed.org/series/DGS2

Board of Governors of the Federal Reserve System (US). (2026b). *Market yield on U.S. Treasury securities at 10-year constant maturity, quoted on an investment basis* [DGS10] [Data set]. FRED, Federal Reserve Bank of St. Louis. Retrieved November 13, 2026, from https://fred.stlouisfed.org/series/DGS10

Board of Governors of the Federal Reserve System (US). (2026c). *Market yield on U.S. Treasury securities at 10-year constant maturity, quoted on an investment basis, inflation-indexed* [DFII10] [Data set]. FRED, Federal Reserve Bank of St. Louis. Retrieved November 13, 2026, from https://fred.stlouisfed.org/series/DFII10

Board of Governors of the Federal Reserve System (US). (2026d). *Nominal broad U.S. dollar index* [DTWEXBGS] [Data set]. FRED, Federal Reserve Bank of St. Louis. Retrieved November 13, 2026, from https://fred.stlouisfed.org/series/DTWEXBGS

Board of Governors of the Federal Reserve System. (2026e). *Meeting calendars and information*. Retrieved November 13, 2026, from https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm

Caldara, D., & Iacoviello, M. (2026). *Daily geopolitical risk index* [Data set]. Retrieved November 13, 2026, from https://www.matteoiacoviello.com/gpr.htm

Federal Reserve Bank of St. Louis. (2026a). *10-year breakeven inflation rate* [T10YIE] [Data set]. FRED. Retrieved November 13, 2026, from https://fred.stlouisfed.org/series/T10YIE

Federal Reserve Bank of St. Louis. (2026b). *Release calendar*. FRED. Retrieved November 13, 2026, from https://fred.stlouisfed.org/releases/calendar

U.S. Commodity Futures Trading Commission. (2026). *Commitments of traders: Disaggregated futures only, gold – Commodity Exchange Inc. (088691)* [Data set]. CFTC Public Reporting Environment. Retrieved November 13, 2026, from https://publicreporting.cftc.gov/

Yahoo Finance. (2026). *Historical prices for GC=F, MGC=F, SI=F, CL=F, DX-Y.NYB, ^GSPC, and ^VIX* [Data set]. Retrieved November 13, 2026, from https://finance.yahoo.com/

### 6.2 統計方法

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B (Methodological), 57*(1), 289–300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x

Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica, 37*(3), 424–438. https://doi.org/10.2307/1912791

Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *The Annals of Mathematical Statistics, 18*(1), 50–60. https://doi.org/10.1214/aoms/1177730491

Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica, 55*(3), 703–708. https://doi.org/10.2307/1913610

Welch, B. L. (1947). The generalization of "Student's" problem when several different population variances are involved. *Biometrika, 34*(1–2), 28–35. https://doi.org/10.1093/biomet/34.1-2.28

### 6.3 相關研究（報告中拿來對照的文獻）

Barsky, R., Epstein, C., Lafont-Mueller, A., & Yoo, Y. (2021). What drives gold prices? *Chicago Fed Letter*, (464). Federal Reserve Bank of Chicago. https://www.chicagofed.org/publications/chicago-fed-letter/2021/464

Baur, D. G., & Lucey, B. M. (2010). Is gold a hedge or a safe haven? An analysis of stocks, bonds and gold. *Financial Review, 45*(2), 217–229. https://doi.org/10.1111/j.1540-6288.2010.00244.x

Baur, D. G., & McDermott, T. K. (2010). Is gold a safe haven? International evidence. *Journal of Banking & Finance, 34*(8), 1886–1898. https://doi.org/10.1016/j.jbankfin.2009.12.008

Baur, D. G., & Smales, L. A. (2020). Hedging geopolitical risk with precious metals. *Journal of Banking & Finance, 117*, Article 105823. https://doi.org/10.1016/j.jbankfin.2020.105823

Cai, J., Cheung, Y.-L., & Wong, M. C. S. (2001). What moves the gold market? *Journal of Futures Markets, 21*(3), 257–278. https://ideas.repec.org/a/wly/jfutmk/v21y2001i3p257-278.html

Caldara, D., & Iacoviello, M. (2022). Measuring geopolitical risk. *American Economic Review, 112*(4), 1194–1225. https://doi.org/10.1257/aer.20191823

Christie-David, R., Chaudhry, M., & Koch, T. W. (2000). Do macroeconomics news releases affect gold and silver prices? *Journal of Economics and Business, 52*(5), 405–421. https://doi.org/10.1016/S0148-6195(00)00029-1

Elder, J., Miao, H., & Ramchander, S. (2012). Impact of macroeconomic news on metal futures. *Journal of Banking & Finance, 36*(1), 51–65. https://doi.org/10.1016/j.jbankfin.2011.06.007

Erb, C. B., & Harvey, C. R. (2013). The golden dilemma. *Financial Analysts Journal, 69*(4), 10–42. https://doi.org/10.2469/faj.v69.n4.1

Roache, S. K., & Rossi, M. (2010). The effects of economic news on commodity prices. *The Quarterly Review of Economics and Finance, 50*(3), 377–385. https://www.sciencedirect.com/science/article/pii/S1062976910000207

Smales, L. A., & Yang, Y. (2015). The importance of belief dispersion in the response of gold futures to macroeconomic announcements. *International Review of Financial Analysis, 41*, 292–302. https://doi.org/10.1016/j.irfa.2015.01.017

> 附註：Cai 等人（2001）與 Roache 和 Rossi（2010）沒有查到可確認的 DOI，所以附上文獻頁面網址。資料集的年份標為擷取年（2026），因為這些資料會持續更新。
