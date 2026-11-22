"""Caminhos do projeto, calculados a partir desta pasta.

Assim os scripts funcionam qualquer que seja a pasta de onde são rodados.
"""
from pathlib import Path

PASTA_BACKEND = Path(__file__).resolve().parents[1]

PASTA_DADOS = PASTA_BACKEND / "dados"
ARQUIVO_REDACOES = PASTA_DADOS / "essay-br.csv"
ARQUIVO_PROPOSTAS = PASTA_DADOS / "prompts.csv"
ARQUIVO_TEMAS_CURTOS = PASTA_DADOS / "temas_curtos.csv"

# Modelo treinado (pesos, tokenizador, config.json com max_len e cortes calibrados)
PASTA_MODELO = PASTA_BACKEND / "saved_model"

# Resultados de experimentos versionados no repositório
PASTA_RESULTADOS = PASTA_BACKEND / "resultados"
