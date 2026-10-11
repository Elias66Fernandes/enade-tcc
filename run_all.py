"""Roda o pipeline completo. Uso: python run_all.py   (dados já extraídos em data/raw)
Etapas: 01 base -> 02 descritiva/ICC -> 04 modelo principal -> 04 (cursos >= 10 alunos) -> 03 modelo misto -> 05 figuras.
Saídas em resultados/."""
import subprocess, sys
from pathlib import Path

raiz = Path(__file__).resolve().parent
raw, res = raiz / 'data' / 'raw', raiz / 'resultados'
res.mkdir(exist_ok=True)

def etapa(script, *args, log=None):
    print(f'\n>>> {script}', flush=True)
    r = subprocess.run([sys.executable, '-I', str(raiz / 'src' / script), *map(str, args)],
                       capture_output=True, text=True)
    txt = r.stdout + r.stderr
    print(txt)
    if log:
        (res / log).write_text(txt, encoding='utf-8')
    if r.returncode:
        sys.exit(f'Falhou em {script} (código {r.returncode})')

etapa('01_prepara_base.py', raw, res, log='01_base.txt')
etapa('02_descritiva_icc.py', res, log='02_descritiva_icc.txt')
etapa('04_modelos_curso.py', res, log='04_modelos_curso.txt')
etapa('04_modelos_curso.py', res, 10, log='04_modelos_curso_min10.txt')
etapa('03_modelos.py', res, log='03_modelos.txt')
etapa('05_figuras_tabelas.py', res)
print('\nConcluído. Veja a pasta resultados/.')
