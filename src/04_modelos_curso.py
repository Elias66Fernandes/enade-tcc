"""Estimativa do efeito da modalidade com estrutura hierárquica, só com numpy/pandas/scipy.
Como a modalidade é atributo do CURSO, o efeito é estimado sobre as médias dos cursos:
  media_curso_j = b0 + b1*EaD_j + g*G_j + c*Z_j + (u_j + e_j)
com pesos = 1/(tau2 + s2/n_j) (equivale a interceptos aleatórios de curso, tau2 e s2 por momentos)
e erros-padrão robustos agrupados por IES (CR1, t com G-1 gl).  Também: pesos n_j (= OLS de alunos) e pesos iguais.
Uso: python -I 04_modelos_curso.py <pasta_com_estudantes.csv> [min_alunos]   (min_alunos: exclui cursos menores)"""
import sys, numpy as np, pandas as pd
from scipy import stats
out = sys.argv[1]
MIN_N = int(sys.argv[2]) if len(sys.argv) > 2 else 0   # exclui cursos com menos de MIN_N alunos presentes
e = pd.read_csv(f'{out}/estudantes.csv'); e = e[e.AREA_COMPARAVEL == 1].copy()
COMP = ['IDADE_MEDIA', 'P_FEM', 'P_RENDA_ATE15SM', 'P_PAIS_SUP', 'P_TRAB_21H']

def tabela_cursos(d):
    g = d.groupby('CO_CURSO')
    c = g.agg(y=('NT_GER', 'mean'), n=('NT_GER', 'size'), v=('NT_GER', 'var'), EAD=('EAD', 'first'), IES=('CO_IES', 'first'),
              GRUPO=('CO_GRUPO', 'first'), PRIV=('PRIVADA', 'first'), ORG=('CO_ORGACAD', 'first'), REG=('CO_REGIAO_CURSO', 'first'),
              **{k: (k, 'first') for k in COMP}).reset_index()
    return c[c.n >= MIN_N].reset_index(drop=True)

def desenho(c, cols_cat, cols_num, fe_ies=False):
    X = pd.DataFrame({'const': 1.0, 'EAD': c.EAD.astype(float)}, index=c.index)
    for k in cols_num: X[k] = c[k].astype(float)
    for k in cols_cat: X = X.join(pd.get_dummies(c[k], prefix=k, drop_first=True, dtype=float))
    if fe_ies: X = X.join(pd.get_dummies(c.IES, prefix='ies', drop_first=True, dtype=float))
    X = X.loc[:, X.nunique() > 1].copy() if len(X) else X
    if 'const' not in X: X.insert(0, 'const', 1.0)
    return X

def wls_cluster(y, X, w, cl):
    Xv, yv, wv = X.values, y.values, w.values
    A = Xv.T @ (Xv * wv[:, None]); Ai = np.linalg.pinv(A)
    b = Ai @ (Xv.T @ (wv * yv)); r = yv - Xv @ b
    S = (Xv * (wv * r)[:, None]); ids = pd.factorize(cl)[0]; G = ids.max() + 1
    Sg = np.zeros((G, Xv.shape[1])); np.add.at(Sg, ids, S)
    N, K = Xv.shape; adj = G / (G - 1) * (N - 1) / (N - K)
    V = adj * Ai @ (Sg.T @ Sg) @ Ai
    i = list(X.columns).index('EAD'); se = np.sqrt(V[i, i]); t = b[i] / se; p = 2 * stats.t.sf(abs(t), G - 1)
    h = stats.t.ppf(.975, G - 1)
    return b[i], se, b[i] - h * se, b[i] + h * se, p, G, N

def momentos(c, cols_cat, cols_num):
    """tau2 (entre cursos, residual) e s2 (dentro do curso) por momentos."""
    s2 = np.average(c.v.fillna(c.v.mean()), weights=(c.n - 1).clip(lower=1))
    X = desenho(c, cols_cat, cols_num); w = c.n.astype(float)
    Xv = X.values; b = np.linalg.pinv(Xv.T @ (Xv * w.values[:, None])) @ (Xv.T @ (w.values * c.y.values))
    r = c.y.values - Xv @ b
    var_r = np.sum(r**2) / (len(c) - Xv.shape[1])          # variância residual das médias dos cursos
    tau2 = max(0.0, var_r - np.mean(s2 / c.n))             # menos a parte devida ao erro amostral dentro do curso
    return tau2, s2

def roda(rotulo, c, cols_cat, cols_num, fe_ies=False):
    c = c.dropna(subset=cols_num + ['y']).copy()
    X = desenho(c, cols_cat, cols_num, fe_ies)
    tau2, s2 = momentos(c, cols_cat, cols_num)
    res = {}
    for nome, w in [('FGLS (int. aleatorio)', 1 / (tau2 + s2 / c.n)), ('peso n (=OLS aluno)', c.n.astype(float)), ('peso igual', pd.Series(1.0, index=c.index))]:
        b, se, lo, hi, p, G, N = wls_cluster(c.y, X, w, c.IES)
        res[nome] = (b, se, lo, hi, p)
    b, se, lo, hi, p = res['FGLS (int. aleatorio)']
    sd = e.NT_GER.std()
    print(f'[{rotulo}] cursos={N} IES(clusters)={G} tau2={tau2:.1f} s2={s2:.1f}')
    for k, (b_, se_, lo_, hi_, p_) in res.items():
        print(f'    {k:24s} beta_EaD={b_:6.2f}  EP={se_:5.2f}  IC95%[{lo_:6.2f},{hi_:6.2f}]  p={p_:.3g}  d={b_ / sd:5.2f}')

for area in ['Engenharia', 'Saude']:
    d = e[e.AREA == area]; c = tabela_cursos(d)
    print(f'\n##### {area}: alunos={len(d)} cursos={len(c)} IES={c.IES.nunique()}')
    roda('M(ii)  EaD + area', c, ['GRUPO'], [])
    roda('M(iii) + privada/org/regiao + composicao', c, ['GRUPO', 'ORG', 'REG'], ['PRIV'] + COMP)
    roda('so privadas (M iii)', c[c.PRIV == 1], ['GRUPO', 'ORG', 'REG'], COMP)
    ies2 = c.groupby('IES').EAD.nunique(); m = c[c.IES.isin(ies2[ies2 == 2].index)]
    roda(f'dentro da IES (FE, {m.IES.nunique()} IES)', m, ['GRUPO'], [], fe_ies=True)
    roda(f'dentro da IES + composicao', m, ['GRUPO'], COMP, fe_ies=True)
    a, b = c[c.EAD == 1].y, c[c.EAD == 0].y
    print(f'[curso, bruto] dif={a.mean() - b.mean():.2f} Welch p={stats.ttest_ind(a, b, equal_var=False).pvalue:.3g} MWU p={stats.mannwhitneyu(a, b).pvalue:.3g}')
