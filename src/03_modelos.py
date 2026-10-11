"""Modelos multinível + robustez. NÃO TESTADO (statsmodels indisponível no sandbox da sessão).
Requer: pip install statsmodels scipy pandas
Uso: python3 -I 03_modelos.py <dir_saida>   (a pasta com estudantes.csv e cursos.csv)
"""
import sys, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
warnings.filterwarnings('ignore')
out = sys.argv[1]
est = pd.read_csv(f'{out}/estudantes.csv')
est = est[est.AREA_COMPARAVEL == 1].copy()
COMP = 'IDADE_MEDIA + P_FEM + P_RENDA_ATE15SM + P_PAIS_SUP + P_TRAB_21H'
sd_tot = est.NT_GER.std()

def mixed(df, formula, rotulo):
    # Dois níveis (aluno em curso). O modelo de 3 níveis (IES) não convergiu de forma confiável no statsmodels;
    # a dependência entre cursos da mesma IES é tratada nas análises 'OLS cluster CO_IES' e em 04_modelos_curso.py.
    m = smf.mixedlm(formula, df, groups='CO_CURSO').fit(reml=True)
    b, se = m.params['EAD'], m.bse['EAD']
    tau2, s2 = float(m.cov_re.iloc[0, 0]), float(m.scale)
    ok = (tau2 < sd_tot ** 2) and (s2 < sd_tot ** 2)  # sanidade: variâncias não podem exceder a variância total
    print(f'[{rotulo}] beta_EaD={b:.2f} (EP {se:.2f}) IC95% [{b - 1.96 * se:.2f}, {b + 1.96 * se:.2f}] '
          f'p={m.pvalues["EAD"]:.3g} d={b / sd_tot:.2f} | tau2={tau2:.1f} s2={s2:.1f} '
          f'conv={m.converged} {"" if ok else "<<< VARIANCIAS IMPLAUSIVEIS: NAO USAR"}')
    return m

def ols_cluster(df, formula, rotulo, cluster='CO_CURSO'):
    d = df.dropna(subset=[c for c in ['P_RENDA_ATE15SM', 'P_PAIS_SUP', 'P_TRAB_21H'] if c in formula])
    m = smf.ols(formula, d).fit(cov_type='cluster', cov_kwds={'groups': d[cluster]})
    b, se = m.params['EAD'], m.bse['EAD']
    print(f'[OLS cluster {cluster} | {rotulo}] beta_EaD={b:.2f} (EP {se:.2f}) p={m.pvalues["EAD"]:.3g}')

for area, df in [('Engenharia', est[est.AREA == 'Engenharia']), ('Saude', est[est.AREA == 'Saude'])]:
    dc = df.dropna(subset=['P_RENDA_ATE15SM', 'P_PAIS_SUP', 'P_TRAB_21H'])
    print(f'\n##### {area}: n={len(df)} cursos={df.CO_CURSO.nunique()} IES={df.CO_IES.nunique()}')
    mixed(df, 'NT_GER ~ EAD + C(CO_GRUPO)', f'{area} M(ii)')
    mixed(dc, f'NT_GER ~ EAD + C(CO_GRUPO) + PRIVADA + C(CO_ORGACAD) + C(CO_REGIAO_CURSO) + {COMP}', f'{area} M(iii)')
    ols_cluster(df, 'NT_GER ~ EAD + C(CO_GRUPO)', area)
    ols_cluster(df, 'NT_GER ~ EAD + C(CO_GRUPO)', area, 'CO_IES')
    priv = dc[dc.PRIVADA == 1]
    mixed(priv, f'NT_GER ~ EAD + C(CO_GRUPO) + C(CO_ORGACAD) + C(CO_REGIAO_CURSO) + {COMP}', f'{area} so privadas')
    # dentro da mesma IES (efeito fixo de IES) nas IES com as duas modalidades
    ies2 = df.groupby('CO_IES').EAD.nunique(); mix = df[df.CO_IES.isin(ies2[ies2 == 2].index)]
    m = smf.ols('NT_GER ~ EAD + C(CO_GRUPO) + C(CO_IES)', mix).fit(cov_type='cluster', cov_kwds={'groups': mix.CO_CURSO})
    print(f'[FE de IES, {mix.CO_IES.nunique()} IES] beta_EaD={m.params["EAD"]:.2f} (EP {m.bse["EAD"]:.2f}) p={m.pvalues["EAD"]:.3g}')
    # nível do curso
    c = df.groupby('CO_CURSO').agg(m=('NT_GER', 'mean'), ead=('EAD', 'first'), n=('NT_GER', 'size'))
    a, b = c[c.ead == 1].m, c[c.ead == 0].m
    print(f'[curso] Welch p={stats.ttest_ind(a, b, equal_var=False).pvalue:.3g} | MWU p={stats.mannwhitneyu(a, b).pvalue:.3g} | dif={a.mean() - b.mean():.2f}')
    # baseline ingênuo
    x, y = df[df.EAD == 1].NT_GER, df[df.EAD == 0].NT_GER
    print(f'[ingenuo aluno] dif={x.mean() - y.mean():.2f} Welch p={stats.ttest_ind(x, y, equal_var=False).pvalue:.3g}')
