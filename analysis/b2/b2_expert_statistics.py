"""B2 - Small-sample statistics and diverging Likert figure for the JurisMap expert evaluation.

Input : the anonymised questionnaire export (5 respondents, 15 Likert items).
Output: b2_statistics.xlsx, fig_likert_diverging.png (300 dpi) and .pdf
Respondents are reported only as E1-E5; roles are never linked to responses.
"""
import sys, itertools
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = sys.argv[1] if len(sys.argv) > 1 else "Avaliação do JurisMap_5resp_20250920.xlsx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
SEED = 20250611
MAP = {"Discordo totalmente": 1, "Discordo": 2, "Não concordo nem discordo": 3,
       "Concordo": 4, "Concordo totalmente": 5}
ITEMS = [  # (column index in export, short English label, construct)
    (9,  "Task 1 scenario (filter & map) is relevant", "Scenario relevance"),
    (10, "Task 2 scenario (chatbot) is relevant", "Scenario relevance"),
    (11, "Task 3 scenario (defense draft) is relevant", "Scenario relevance"),
    (12, "Task 4 scenario (draft with template) is relevant", "Scenario relevance"),
    (13, "Interface is intuitive", "Interface quality"),
    (14, "Interface is attractive", "Interface quality"),
    (15, "Response time is adequate", "Interface quality"),
    (16, "Easy to identify potential precedents", "Core functionality"),
    (17, "Easy to use decision excerpts as citations", "Core functionality"),
    (18, "Chatbot answered correctly", "Core functionality"),
    (19, "Chatbot answered completely", "Core functionality"),
    (20, "JurisMap is useful", "Value proposition"),
    (21, "JurisMap is easy to learn", "Value proposition"),
    (22, "JurisMap would save my (team's) time", "Value proposition"),
    (23, "I would pay to use JurisMap", "Value proposition"),
]
CONSTRUCTS = ["Scenario relevance", "Interface quality", "Core functionality", "Value proposition"]
NO_TASKS_3_4 = "E4"   # respondent who reported being unable to run Tasks 3-4 (see manuscript)

raw = pd.read_excel(SRC)
X = pd.DataFrame({lab: raw.iloc[:, c].map(MAP) for c, lab, _ in ITEMS}).astype(float)
X.index = [f"E{i+1}" for i in range(len(X))]
assert X.notna().all().all() and len(X) == 5
cons_of = {lab: con for _, lab, con in ITEMS}
rng = np.random.default_rng(SEED)

def boot_ci(v, B=10000):
    v = np.asarray(v, float)
    means = rng.choice(v, (B, len(v)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])

def sign_p(v):  # exact one-sided sign test vs neutral (3), ties dropped
    d = np.asarray(v) - 3; d = d[d != 0]
    return stats.binomtest(int((d > 0).sum()), len(d), 0.5, alternative="greater").pvalue if len(d) else np.nan

def t_ci(v):
    v = np.asarray(v, float)
    if v.std(ddof=1) == 0: return (v.mean(), v.mean())
    return stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=stats.sem(v))

def rwg_j(M):  # James, Demaree & Wolf (1984); uniform null variance for a 5-point scale = 2
    J = M.shape[1]; s2 = M.var(axis=0, ddof=1).mean(); r = 1 - s2 / 2
    return J * r / (J * r + s2 / 2)

# ---- item table
rows = []
for _, lab, con in ITEMS:
    v = X[lab].values; lo, hi = t_ci(v)
    rows.append(dict(Construct=con, Item=lab, N=len(v), Mean=v.mean(), SD=v.std(ddof=1),
                     Median=np.median(v), Q1=np.percentile(v, 25), Q3=np.percentile(v, 75),
                     Pct_agree=100 * (v >= 4).mean(), Above_neutral=int((v > 3).sum()),
                     Sign_test_p_one_sided=sign_p(v), t_CI95_low=lo, t_CI95_high=hi,
                     **{f"n_{k}": int((v == k).sum()) for k in range(1, 6)}))
items = pd.DataFrame(rows)

# ---- construct table (per-respondent construct scores)
crow = []
for con in CONSTRUCTS:
    cols = [lab for _, lab, c in ITEMS if c == con]
    s = X[cols].mean(axis=1); lo, hi = boot_ci(s)
    w = stats.wilcoxon(s - 3, alternative="greater", method="exact") if (s != 3).any() else None
    crow.append(dict(Construct=con, Items=len(cols), **{e: s[e] for e in X.index},
                     Mean=s.mean(), Median=s.median(), Boot_CI95_low=lo, Boot_CI95_high=hi,
                     Wilcoxon_exact_p_one_sided=w.pvalue if w else np.nan, rwg_j=rwg_j(X[cols].values)))
s_all = X.mean(axis=1); lo, hi = boot_ci(s_all)
crow.append(dict(Construct="Overall (15 items)", Items=15, **{e: s_all[e] for e in X.index},
                 Mean=s_all.mean(), Median=s_all.median(), Boot_CI95_low=lo, Boot_CI95_high=hi,
                 Wilcoxon_exact_p_one_sided=np.nan, rwg_j=np.nan))
cons = pd.DataFrame(crow)

# ---- sensitivity: exclude the respondent who could not run Tasks 3-4, for those items only
sens = pd.DataFrame([dict(Item=lab, Mean_all=X[lab].mean(), Mean_excl=X.drop(NO_TASKS_3_4)[lab].mean(),
                          N_excl=len(X) - 1) for _, lab, _ in ITEMS[2:4]])
sc = [lab for _, lab, c in ITEMS if c == "Scenario relevance"]
sens.loc[len(sens)] = dict(Item="Scenario relevance construct (Tasks 3-4 excl. " + NO_TASKS_3_4 + ")",
                           Mean_all=X[sc].mean(axis=1).mean(),
                           Mean_excl=pd.concat([X[sc[:2]], X[sc[2:]].drop(NO_TASKS_3_4)], axis=1).mean().mean(),
                           N_excl=np.nan)

notes = pd.DataFrame({"Note": [
    "N = 5 experts; items on a 1-5 agreement scale (1 = strongly disagree, 5 = strongly agree).",
    "Sign test: exact one-sided binomial test of responses above vs below neutral (3), ties removed. With N = 5 the smallest attainable p is 1/32 = 0.031, reached only when all five respondents are above neutral.",
    "Construct scores: mean of a respondent's items in that construct. Bootstrap CI: 10,000 resamples of respondents, seed " + str(SEED) + ".",
    "Wilcoxon: exact one-sided signed-rank test of construct scores against 3.",
    "rwg(j): within-group agreement index (James, Demaree & Wolf, 1984), uniform null variance 2.",
    "Cronbach's alpha is not reported: it is unstable at N = 5 (value proposition alpha < 0).",
    f"{NO_TASKS_3_4} reported being unable to run Tasks 3-4 but rated them; the sensitivity sheet shows those items without {NO_TASKS_3_4}.",
]})

with pd.ExcelWriter(f"{OUT}/b2_statistics.xlsx") as xw:
    items.round(3).to_excel(xw, sheet_name="Items", index=False)
    cons.round(3).to_excel(xw, sheet_name="Constructs", index=False)
    sens.round(3).to_excel(xw, sheet_name="Sensitivity", index=False)
    X.T.to_excel(xw, sheet_name="Responses_anonymised")
    notes.to_excel(xw, sheet_name="Notes", index=False)

# ---- figure: diverging stacked bars (neutral centred on zero)
COL = {1: "#b8322f", 2: "#f0a09c", 3: "#c9c8c1", 4: "#86b6ef", 5: "#1c5cab"}
LAB = {1: "Strongly disagree", 2: "Disagree", 3: "Neutral", 4: "Agree", 5: "Strongly agree"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=300)
ypos, y, yticks, ylabels, groups = {}, 0, [], [], []
for con in CONSTRUCTS:
    y += 0.6; groups.append((con, y)); y += 0.55
    for _, lab, c in ITEMS:
        if c == con:
            ypos[lab] = y; yticks.append(y); ylabels.append(lab); y += 1
H = 0.62
for _, lab, _ in ITEMS:
    cnt = {k: int((X[lab] == k).sum()) for k in range(1, 6)}
    left = -(cnt[1] + cnt[2] + cnt[3] / 2)
    for k in range(1, 6):
        if cnt[k]:
            ax.barh(ypos[lab], cnt[k], left=left, height=H, color=COL[k], edgecolor="white", linewidth=1.2)
        left += cnt[k]
    it = items.set_index("Item").loc[lab]
    mark = "  *" if it.Above_neutral == 5 else ""
    ax.text(5.25, ypos[lab], f"{it.Median:g}  ({it.Mean:.1f}){mark}", va="center", ha="left", color=INK, fontsize=7.8)
for con, gy in groups:
    ax.text(-2.95, gy, con, ha="left", va="center", fontsize=8.5, fontweight="bold", color=INK)
ax.text(5.25, groups[0][1], "Median (mean)", ha="left", va="center", fontsize=7.8, color=MUTED)
ax.axvline(0, color="#898781", linewidth=0.8, zorder=0)
ax.set_yticks(yticks); ax.set_yticklabels(ylabels, color=INK)
ax.set_ylim(y + 0.1, 0); ax.set_xlim(-3, 5.2)
ax.set_xticks(range(-3, 6)); ax.set_xticklabels([str(abs(t)) for t in range(-3, 6)], color=MUTED)
ax.set_xlabel("Number of experts (left: disagree side, right: agree side; neutral split at zero)", color=MUTED)
ax.xaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#c3c2b7"); ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", colors=MUTED)
handles = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in range(1, 6)]
ax.legend(handles, [LAB[k] for k in range(1, 6)], ncol=5, loc="lower center", bbox_to_anchor=(0.45, 1.0),
          frameon=False, fontsize=7.8, handlelength=1.2, columnspacing=1.0)
fig.text(0.01, 0.005, "* all five experts above neutral (exact one-sided sign test p = 0.031, the minimum attainable with N = 5).",
         fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 0.93, 1))
fig.savefig(f"{OUT}/fig_likert_diverging.png", dpi=300)
fig.savefig(f"{OUT}/fig_likert_diverging.pdf")
print(items[["Item", "Median", "Mean", "Pct_agree", "Above_neutral", "Sign_test_p_one_sided"]].round(3).to_string())
print(cons.round(3).to_string()); print(sens.round(3).to_string())
