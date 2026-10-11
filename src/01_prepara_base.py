"""Prepara a base analítica do TCC (ENADE 2023).
Uso: python3 -I 01_prepara_base.py <dir_DADOS> <dir_saida>
Saídas: estudantes.parquet/csv (nível aluno: nota + atributos do curso) e cursos.csv (nível curso + composição).
Obs.: os arquivos do INEP só se ligam por CO_CURSO (ver manual, seção LGPD).
"""
import sys, os
import pandas as pd
import numpy as np

d, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

ENG = {5710, 5806, 5814, 5902, 6002, 6008, 6208, 6307, 6405, 6411}
SAUDE = {6, 12, 19, 23, 27, 28, 36, 55}

# --- curso (arq1): uma linha por curso
a1 = pd.read_csv(f'{d}/microdados2023_arq1.txt', sep=';',
                 usecols=['CO_CURSO', 'CO_IES', 'CO_CATEGAD', 'CO_ORGACAD', 'CO_GRUPO',
                          'CO_MODALIDADE', 'CO_REGIAO_CURSO'])
curso = a1.drop_duplicates('CO_CURSO').copy()
assert curso.CO_CURSO.is_unique
curso['AREA'] = np.where(curso.CO_GRUPO.isin(ENG), 'Engenharia',
                 np.where(curso.CO_GRUPO.isin(SAUDE), 'Saude', 'Outra'))
curso['PRIVADA'] = curso.CO_CATEGAD.isin([4, 5, 8]).astype(int)  # priv. c/ e s/ fins lucr. e comunitária
curso['EAD'] = (curso.CO_MODALIDADE == 0).astype(int)

# --- notas (arq3)
a3 = pd.read_csv(f'{d}/microdados2023_arq3.txt', sep=';',
                 usecols=['CO_CURSO', 'TP_PRES', 'TP_PR_GER', 'NT_GER', 'NT_FG', 'NT_CE'], dtype=str)
for c in ['NT_GER', 'NT_FG', 'NT_CE']:
    a3[c] = pd.to_numeric(a3[c].str.replace(',', '.'), errors='coerce')
a3['TP_PRES'] = a3.TP_PRES.astype(int)
a3['CO_CURSO'] = a3.CO_CURSO.astype(int)
n0 = len(a3)
fun = {'inscritos': n0}
a3 = a3[a3.TP_PRES == 555]; fun['TP_PRES=555'] = len(a3)
a3 = a3[a3.TP_PR_GER == '555']; fun['TP_PR_GER=555'] = len(a3)
a3 = a3[a3.NT_GER.notna()]; fun['NT_GER nao nulo'] = len(a3)
est = a3.merge(curso, on='CO_CURSO', how='left', validate='m:1')
assert est.CO_GRUPO.notna().all()
est = est[est.AREA != 'Outra']; fun['areas Eng+Saude'] = len(est)
# áreas sem nenhum curso EaD (com alunos válidos) ficam só na descritiva
ead_areas = set(est.loc[est.EAD == 1, 'CO_GRUPO'].unique())
est['AREA_COMPARAVEL'] = est.CO_GRUPO.isin(ead_areas).astype(int)
print('Funil de filtros:', fun)

# --- composição por curso (perfil dos inscritos presentes não é identificável: usa todos os inscritos do curso)
def agg(arq, col, fn, name):
    x = pd.read_csv(f'{d}/microdados2023_arq{arq}.txt', sep=';', usecols=['CO_CURSO', col], dtype=str)
    x['CO_CURSO'] = x.CO_CURSO.astype(int)
    x = x[x[col].notna() & (x[col] != '')]
    x['v'] = fn(x[col])
    return x.groupby('CO_CURSO').v.mean().rename(name)

idade = pd.read_csv(f'{d}/microdados2023_arq6.txt', sep=';', usecols=['CO_CURSO', 'NU_IDADE']).groupby('CO_CURSO').NU_IDADE.mean().rename('IDADE_MEDIA')
comp = pd.concat([
    idade,
    agg(5, 'TP_SEXO', lambda s: (s == 'F').astype(float), 'P_FEM'),
    agg(14, 'QE_I08', lambda s: (s == 'A').astype(float), 'P_RENDA_ATE15SM'),
    agg(10, 'QE_I04', lambda s: s.isin(['E', 'F']).astype(float), 'P_PAI_SUP'),
    agg(11, 'QE_I05', lambda s: s.isin(['E', 'F']).astype(float), 'P_MAE_SUP'),
    agg(16, 'QE_I10', lambda s: s.isin(['D', 'E']).astype(float), 'P_TRAB_21H'),
], axis=1).reset_index()
comp['P_PAIS_SUP'] = comp[['P_PAI_SUP', 'P_MAE_SUP']].mean(axis=1)  # média das proporções pai e mãe (aproximação)

est = est.merge(comp, on='CO_CURSO', how='left')
est.to_csv(f'{out}/estudantes.csv', index=False)

cur = est.groupby('CO_CURSO').agg(N=('NT_GER', 'size'), MEDIA=('NT_GER', 'mean'), DP=('NT_GER', 'std')).reset_index()
cur = cur.merge(curso, on='CO_CURSO').merge(comp, on='CO_CURSO', how='left')
cur.to_csv(f'{out}/cursos.csv', index=False)

print('\nEstudantes:', len(est), '| cursos:', est.CO_CURSO.nunique(), '| IES:', est.CO_IES.nunique())
print(est.groupby(['AREA', 'EAD']).agg(alunos=('NT_GER', 'size'), cursos=('CO_CURSO', 'nunique'), ies=('CO_IES', 'nunique')).to_string())
print('\nGrupos sem curso EaD:', sorted(set(est.CO_GRUPO) - ead_areas))
print('NaN em composição (cursos):', cur[['IDADE_MEDIA', 'P_FEM', 'P_RENDA_ATE15SM', 'P_PAIS_SUP', 'P_TRAB_21H']].isna().sum().to_dict())
