"""Figuras e tabela de composição usadas no texto do TCC (Cap. 4).
Uso: python -I 05_figuras_tabelas.py <pasta_resultados>
Lê resultados/cursos.csv e resultados/04_modelos_curso.txt; grava PNGs em resultados/figuras/ e
resultados/tabela_composicao.csv. O gráfico de floresta usa as estimativas FGLS (pesos de intercepto aleatório)."""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

res = Path(sys.argv[1]); fig_dir = res / 'figuras'; fig_dir.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})
AZUL, LARANJA = '#1f5fa8', '#d9731a'
GRUPOS = {'Engenharia': [5710, 5806, 5814, 5902, 6002, 6008, 6208, 6307, 6411],
          'Saude': [19, 23, 27, 28, 36, 55]}          # grupos com oferta nas duas modalidades

d = pd.read_csv(res / 'cursos.csv')
d = d[d.apply(lambda r: r.CO_GRUPO in GRUPOS.get(r.AREA, []), axis=1)]

# Tabela de composição (médias simples entre cursos)
comp = ['IDADE_MEDIA', 'P_FEM', 'P_RENDA_ATE15SM', 'P_PAIS_SUP', 'P_TRAB_21H']
t = d.groupby(['AREA', 'EAD'])[comp].mean()
t.insert(0, 'cursos', d.groupby(['AREA', 'EAD']).size())
t.round(4).to_csv(res / 'tabela_composicao.csv'); print(t.round(3))

# Boxplot das médias dos cursos
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
for ax, (a, titulo) in zip(axs, [('Engenharia', 'Engenharia'), ('Saude', 'Saúde')]):
    x = d[d.AREA == a]; dados = [x[x.EAD == 0].MEDIA, x[x.EAD == 1].MEDIA]
    bp = ax.boxplot(dados, widths=.5, patch_artist=True, showfliers=False, medianprops=dict(color='black', lw=1.5))
    for p, c in zip(bp['boxes'], [AZUL, LARANJA]): p.set_facecolor(c); p.set_alpha(.55); p.set_edgecolor('#333')
    for i, v in enumerate(dados, 1):
        ax.plot(i, v.mean(), marker='D', color='white', mec='black', ms=6, zorder=5)
        ax.text(i, -3, f'n = {len(v):,}'.replace(',', '.'), ha='center', fontsize=8.5)
    ax.set_xticks([1, 2]); ax.set_xticklabels(['Presencial', 'EaD']); ax.set_title(titulo, fontsize=10.5)
    ax.set_ylim(-6, 100); ax.grid(axis='y', color='#ddd', lw=.6); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
axs[0].set_ylabel('Nota geral média do curso (0–100)')
fig.text(.5, .005, 'Caixa: quartis; linha: mediana; losango: média dos cursos; n = número de cursos.', ha='center', fontsize=8, color='#444')
fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(fig_dir / 'fig_boxplot_cursos.png', dpi=300); plt.close()

# Gráfico de floresta: estimativas FGLS do 04_modelos_curso.txt (5 especificações por área)
txt = (res / '04_modelos_curso.txt').read_text(encoding='utf-8')
est = {}
for area, bloco in zip(['Engenharia', 'Saúde'], re.split(r'#####', txt)[1:]):
    linhas = re.findall(r'FGLS \(int\. aleatorio\)\s+beta_EaD=\s*(-?[\d.]+)\s+EP=\s*[\d.]+\s+IC95%\[\s*(-?[\d.]+),\s*(-?[\d.]+)\]', bloco)
    assert len(linhas) == 5, (area, len(linhas))
    est[area] = [tuple(map(float, l)) for l in linhas]
specs = ['(i) Modalidade + grupo de curso', '(ii) Todos os controles', '(iii) Somente IES privadas',
         '(iv) Dentro da mesma IES', '(v) Dentro da mesma IES + composição']
fig, ax = plt.subplots(figsize=(7.2, 4.0)); y = np.arange(5)[::-1]
for off, area, c, m in [(0.15, 'Engenharia', AZUL, 'o'), (-0.15, 'Saúde', LARANJA, 's')]:
    for yi, (b, lo, hi) in zip(y, est[area]):
        ax.plot([lo, hi], [yi + off] * 2, color=c, lw=2); ax.plot(b, yi + off, m, color=c, ms=7, mec='white', mew=1.2)
    ax.plot([], [], m + '-', color=c, label=area)
ax.axvline(0, color='#333', lw=1); ax.set_yticks(y); ax.set_yticklabels(specs)
ax.set_xlabel('Diferença EaD − presencial na nota geral (pontos; IC 95%)')
ax.grid(axis='x', color='#ddd', lw=.6); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc='lower left'); fig.tight_layout(); fig.savefig(fig_dir / 'fig_forest_modelo.png', dpi=300)
print('Figuras em', fig_dir)
