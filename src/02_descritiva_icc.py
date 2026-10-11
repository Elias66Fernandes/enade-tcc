"""Descritiva e ICC (modelo nulo de 3 níveis). Uso: python3 -I 02_descritiva_icc.py <dir_saida>"""
import sys, time, warnings
import pandas as pd, numpy as np
warnings.filterwarnings('ignore')
out = sys.argv[1]
est = pd.read_csv(f'{out}/estudantes.csv')
cur = pd.read_csv(f'{out}/cursos.csv')
est['MOD'] = np.where(est.EAD == 1, 'EaD', 'Presencial')
cur['MOD'] = np.where(cur.EAD == 1, 'EaD', 'Presencial')

pd.set_option('display.width', 200)
print('=== Descritiva (estudante) NT_GER ===')
print(est.groupby(['AREA', 'MOD']).NT_GER.agg(['count', 'mean', 'median', 'std', 'min', 'max']).round(2).to_string())
print('\n=== Descritiva (curso: média do curso) ===')
print(cur.groupby(['AREA', 'MOD']).agg(cursos=('CO_CURSO', 'size'), media_cursos=('MEDIA', 'mean'),
      dp_cursos=('MEDIA', 'std'), n_mediano=('N', 'median'), n_max=('N', 'max'), pequenos_lt10=('N', lambda s: (s < 10).sum())).round(2).to_string())
print('\n=== Por área (CO_GRUPO) x modalidade: alunos / cursos / média ===')
g = est.groupby(['CO_GRUPO', 'MOD']).agg(alunos=('NT_GER', 'size'), cursos=('CO_CURSO', 'nunique'), media=('NT_GER', 'mean')).round(1).unstack()
print(g.to_string())
print('\n=== Categoria administrativa (cursos) ===')
print(cur.groupby(['MOD', 'PRIVADA']).size().unstack(fill_value=0).to_string())
ies2 = cur.groupby('CO_IES').EAD.agg(['min', 'max'])
print('\nIES com as duas modalidades (nas áreas do estudo):', int(((ies2['min'] == 0) & (ies2['max'] == 1)).sum()))
print('Assimetria NT_GER:', round(est.NT_GER.skew(), 3), '| curtose:', round(est.NT_GER.kurt(), 3))

def icc_modelo(df, rotulo):
    """Modelo nulo de 2 níveis: NT_GER ~ 1 + (1|curso). (O de 3 níveis com IES não convergiu.)"""
    import statsmodels.formula.api as smf
    t = time.time()
    m = smf.mixedlm('NT_GER ~ 1', df, groups='CO_CURSO').fit(reml=True)
    tau2, s2 = float(m.cov_re.iloc[0, 0]), float(m.scale)
    mbar = df.groupby('CO_CURSO').size().mean(); icc = tau2 / (tau2 + s2)
    print(f'[{rotulo}] n={len(df)} cursos={df.CO_CURSO.nunique()} | tau2_curso={tau2:.1f} s2_e={s2:.1f} | '
          f'ICC_curso={icc:.3f} | m_medio={mbar:.1f} DEFF={1 + (mbar - 1) * icc:.1f} | conv={m.converged} ({time.time() - t:.0f}s)')

def icc_anova(df, col='CO_CURSO'):
    g = df.groupby(col).NT_GER; n = g.size(); k = len(n); N = n.sum()
    gm = df.NT_GER.mean(); ssb = (n * (g.mean() - gm) ** 2).sum(); ssw = ((df.NT_GER - g.transform('mean')) ** 2).sum()
    msb, msw = ssb / (k - 1), ssw / (N - k); n0 = (N - (n ** 2).sum() / N) / (k - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw), n.mean()
print('\n=== ICC(1) por ANOVA, agrupando por curso (aproximacao; ignora IES) ===')
for a in ['Engenharia', 'Saude']:
    d_ = est[(est.AREA == a) & (est.AREA_COMPARAVEL == 1)]; r, mb = icc_anova(d_)
    print(f'{a}: ICC_curso={r:.3f} | alunos/curso medio={mb:.1f} | DEFF={1 + (mb - 1) * r:.1f}')
d_ = est[est.AREA_COMPARAVEL == 1]; r, mb = icc_anova(d_); print(f'Conjunto: ICC_curso={r:.3f} | DEFF={1 + (mb - 1) * r:.1f}')
d_ = est[est.AREA_COMPARAVEL == 1]; r, _ = icc_anova(d_, 'CO_IES'); print(f'Conjunto, agrupando por IES: ICC_IES={r:.3f}')
try:
    import statsmodels  # noqa
except ImportError:
    print('\n[statsmodels ausente: modelo nulo REML nao executado]'); sys.exit(0)
print('\n=== ICC (modelo nulo, REML) ===')
for a in ['Engenharia', 'Saude']:
    icc_modelo(est[(est.AREA == a) & (est.AREA_COMPARAVEL == 1)], a + ' (areas comparaveis)')
