# Desempenho no ENADE 2023: presencial × EaD (Engenharia e Saúde)

Código e resultados do TCC *Análise Comparativa do Desempenho Acadêmico entre Modalidades de Ensino em Cursos de
Graduação: um estudo com dados do ENADE nas áreas de Engenharia e Saúde* (UFMA, Engenharia da Computação).

Os microdados do INEP são agrupados e só podem ser ligados pelo código do curso (`CO_CURSO`). Por isso a modalidade
é estimada **no nível do curso** (estudante → curso → IES), e as características dos estudantes entram como
indicadores de composição do curso. Detalhes e limitações estão no Capítulo 3 do trabalho.

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `src/00_extrai_dados.py` | extrai do zip oficial os arquivos usados (`microdados2023_arq*.txt`) para `data/raw/` |
| `src/01_prepara_base.py` | filtros de elegibilidade, classificação das áreas, indicadores de composição → `estudantes.csv`, `cursos.csv` |
| `src/02_descritiva_icc.py` | estatística descritiva, ICC e efeito de delineamento |
| `src/04_modelos_curso.py` | **modelo principal** (nível do curso, pesos de intercepto aleatório, EP robustos por IES) e robustez; opcional `min_alunos` |
| `src/03_modelos.py` | modelo misto de dois níveis (REML) e regressões de comparação (requer `statsmodels`) |
| `src/05_figuras_tabelas.py` | figuras e tabela de composição do texto |
| `resultados/` | saídas geradas (`*.txt`, `cursos.csv`, `tabela_composicao.csv`, `figuras/`) |

## Como reproduzir

1. Baixe o zip dos microdados do ENADE 2023 no portal de dados abertos do INEP
   (<https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enade>). Os dados **não** estão neste repositório.
2. Python 3.10 ou superior:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
   pip install -r requirements.txt
   python src/00_extrai_dados.py caminho/para/microdados_enade_2023.zip
   python run_all.py
   ```
3. As saídas aparecem em `resultados/`. O arquivo `resultados/estudantes.csv` (≈ 50 MB) é gerado localmente e não é versionado.

A execução completa do passo 01 lê o arquivo de notas (~90 MB) e pode levar alguns minutos.
Se já tiver `resultados/cursos.csv` e `estudantes.csv`, é possível rodar só uma etapa, por exemplo
`python -I src/04_modelos_curso.py resultados` (modelo principal) ou `python -I src/04_modelos_curso.py resultados 10` (cursos com ≥ 10 estudantes).

## Decisões de classificação (conferir em `src/01_prepara_base.py`)

- Presença válida: `TP_PRES = 555` e `TP_PR_GER = 555`, com `NT_GER` não nulo.
- Modalidade: `CO_MODALIDADE` 0 = EaD, 1 = presencial.
- Privada: `CO_CATEGAD` ∈ {4, 5, 8}.
- Engenharia: grupos 5710, 5806, 5814, 5902, 6002, 6008, 6208, 6307, 6405, 6411. Saúde: 6, 12, 19, 23, 27, 28, 36, 55.
  A comparação entre modalidades usa apenas os grupos com cursos nas duas modalidades.
- Indicadores de composição calculados por curso, com todos os inscritos que responderam o questionário.

## Resultados de referência

`resultados/04_modelos_curso.txt` contém as estimativas do modelo principal (por exemplo, diferença EaD − presencial com
todos os controles: 0,02 ponto em Engenharia e −0,92 ponto em Saúde). A execução de `04_modelos_curso.py` sobre
`resultados/` reproduz exatamente esses números.

## Licença e dados

Código sob licença MIT (`LICENSE`). Os microdados pertencem ao INEP e devem ser obtidos na fonte oficial.
