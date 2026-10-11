"""Extrai do zip do INEP apenas os arquivos usados no TCC.
Uso: python src/00_extrai_dados.py <caminho_do_zip> [<pasta_destino>]   (padrão: data/raw)
"""
import sys, zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
dest = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / 'data' / 'raw'
dest.mkdir(parents=True, exist_ok=True)
precisa = {f'microdados2023_arq{i}.txt' for i in (1, 3, 5, 6, 10, 11, 14, 16)}
with zipfile.ZipFile(zip_path) as z:
    achou = set()
    for info in z.infolist():
        nome = Path(info.filename).name
        if nome in precisa and '/DADOS/' in info.filename.replace('\\', '/'):
            (dest / nome).write_bytes(z.read(info))  # grava só o nome do arquivo (sem caminhos do zip)
            achou.add(nome)
faltam = precisa - achou
print('Extraídos em', dest, ':', sorted(achou))
if faltam:
    sys.exit(f'ERRO: arquivos ausentes no zip: {sorted(faltam)}')
