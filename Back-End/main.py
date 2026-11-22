import json
import os
import random
import re
import time
from collections import Counter
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel
from transformers import AutoTokenizer

from avaliador.caminhos import PASTA_BACKEND, PASTA_MODELO
from avaliador.modelo import EnemEssayEvaluator

# Carrega variáveis do arquivo .env (ex.: GEMINI_API_KEY), se existir
load_dotenv(PASTA_BACKEND / ".env")


# A arquitetura do modelo (EnemEssayEvaluator) fica em avaliador/modelo.py,
# a mesma usada no treino.


# Schemas de Entrada e Saída
class EssayRequest(BaseModel):
    tema: str
    redacao: str


class EssayResponse(BaseModel):
    c1_norma_culta: int
    c2_compreensao_tema: int
    c3_projeto_de_texto: int
    c4_coesao: int
    c5_proposta_intervencao: int
    nota_final: int
    feedback: List[str]

class NotasCompetencias(BaseModel):
    c1_norma_culta: int
    c2_compreensao_tema: int
    c3_projeto_de_texto: int
    c4_coesao: int
    c5_proposta_intervencao: int
    nota_final: int


class RewriteResponse(BaseModel):
    redacao_reescrita: str
    # Notas do BERTimbau para a redação original e para a versão reescrita
    notas_original: NotasCompetencias
    notas_reescrita: NotasCompetencias


# Tamanho máximo da sequência (tema + redação) em tokens.
# Ao carregar o modelo, é substituído pelo valor de saved_model/config.json,
# gravado pelo train.py, para que a inferência use o mesmo valor do treino.
# 384 é o valor usado por modelos treinados antes de existir o config.json.
MAX_LEN = 384

# Configuração do Gemini (usado na reescrita da redação)
# GEMINI_MODEL pode ser trocado no .env por qualquer modelo disponível para a sua chave
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or "gemini-3.5-flash-lite"

# Backoff exponencial para falhas temporárias do Gemini (429 e 5xx)
GEMINI_TENTATIVAS = int(os.getenv("GEMINI_TENTATIVAS") or 5)
GEMINI_ESPERA_INICIAL = float(os.getenv("GEMINI_ESPERA_INICIAL") or 1.0)  # segundos
GEMINI_ESPERA_MAXIMA = float(os.getenv("GEMINI_ESPERA_MAXIMA") or 16.0)  # segundos

# Cortes que convertem a previsão contínua (0–200) em notas 0, 40, ..., 200.
# Padrão: múltiplo de 40 mais próximo. Se saved_model/config.json tiver "cuts"
# (gerados pelo calibrate.py), cada competência usa os seus cortes calibrados.
CORTES_PADRAO = [20.0, 60.0, 100.0, 140.0, 180.0]
CUTS = [CORTES_PADRAO] * 5
NOTAS_POSSIVEIS = np.array([0, 40, 80, 120, 160, 200])


def converter_notas(raw_outputs) -> np.ndarray:
    """Converte a saída do modelo (escala 0–1, 5 competências) em notas 0–200 múltiplas de 40."""
    escala = np.asarray(raw_outputs, dtype=float) * 200.0
    return np.array([
        int(NOTAS_POSSIVEIS[np.searchsorted(CUTS[i], escala[i], side="right")])
        for i in range(5)
    ])


# Estado Global
MODEL = None
TOKENIZER = None
GEMINI_CLIENT = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Gerenciador de Ciclo de Vida
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, TOKENIZER, MAX_LEN, CUTS
    MODEL_PATH = PASTA_MODELO / "model_weights.pt"
    TOKENIZER_PATH = PASTA_MODELO / "tokenizer"
    CONFIG_PATH = PASTA_MODELO / "config.json"

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Arquivo não encontrado em '{MODEL_PATH}'. "
            "Verifique se a pasta 'saved_model' foi extraída na raiz do projeto."
        )

    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            config = json.load(f)
        MAX_LEN = int(config.get("max_len", MAX_LEN))
        if config.get("cuts"):
            CUTS = [list(map(float, c)) for c in config["cuts"]]
            print(f"Usando cortes calibrados: {CUTS}")

    print(f"Carregando modelo e tokenizador em {DEVICE} (max_len={MAX_LEN})...")
    TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    MODEL = EnemEssayEvaluator().to(DEVICE)
    MODEL.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    MODEL.eval()
    print("Modelo carregado e pronto para inferência!")

    yield

    # Limpeza ao desligar o servidor
    MODEL = None
    TOKENIZER = None


app = FastAPI(title="API Avaliadora de Redações ENEM", lifespan=lifespan)

# Libera requisições do Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def gerar_diagnostico(redacao: str, notas: dict, model, tokenizer, device, tema: str):
    feedbacks = []

    # Regras diagnósticas baseadas nas notas previstas por competência
    criterios = {
        "c1_norma_culta": (
            notas["c1_norma_culta"],
            "Competência 1 (Norma Culta): Foram detectados desvios gramaticais, de pontuação, acentuação ou escolhas vocabulares informais. Revise a concordância verbal/nominal e evite marcas de oralidade.",
        ),
        "c2_compreensao_tema": (
            notas["c2_compreensao_tema"],
            "Competência 2 (Tema e Estrutura): A abordagem ao tema está superficial ou falta repertório sociocultural legitimado (citações, dados históricos, filósofos ou leis) bem articulado à discussão.",
        ),
        "c3_projeto_de_texto": (
            notas["c3_projeto_de_texto"],
            "Competência 3 (Argumentação): Há falhas no projeto de texto. Desenvolva melhor a relação de causa e consequência dos seus argumentos, evitando afirmações sem comprovação.",
        ),
        "c4_coesao": (
            notas["c4_coesao"],
            "Competência 4 (Coesão): Faltam conectivos interparágrafos (ex: 'Nesse sentido', 'Ademais', 'Portanto') ou há repetição excessiva de palavras ao longo dos períodos.",
        ),
        "c5_proposta_intervencao": (
            notas["c5_proposta_intervencao"],
            "Competência 5 (Proposta de Intervenção): A conclusão está incompleta. Garanta a presença explícita dos 5 elementos: Agente, Ação, Meio/Modo, Efeito e Detalhamento.",
        ),
    }

    for nota, mensagem in criterios.values():
        if nota <= 120:
            feedbacks.append(f"[Atenção - {nota}/200] {mensagem}")
        elif nota == 160:
            feedbacks.append(f"[Bom, mas pode melhorar - 160/200] {mensagem}")

    # Análise de impacto por parágrafo: a rede avalia o texto sem cada parágrafo.
    # Usa a nota contínua (antes da conversão em múltiplos de 40); com a nota
    # convertida, retirar um parágrafo quase nunca mudava nada e tudo empatava em 0.
    paragrafos = [p.strip() for p in redacao.split("\n") if p.strip()]
    if len(paragrafos) >= 2:
        variantes = [redacao] + [
            "\n".join(paragrafos[:i] + paragrafos[i + 1 :]) for i in range(len(paragrafos))
        ]
        brutos = prever_brutos(tema, variantes)  # (1 + n, 5), escala 0–200
        # Contribuição de cada parágrafo = quanto a nota cai quando ele é retirado
        contribuicoes = brutos[0] - brutos[1:]  # (n, 5)
        totais = contribuicoes.sum(axis=1)
        i = int(np.argmin(totais))
        total = float(totais[i])
        comp = int(np.argmax(np.abs(contribuicoes[i]))) + 1
        trecho = paragrafos[i][:80]

        if total < 0:
            efeito = (
                f"sem ele, a nota estimada subiria {abs(total):.0f} pts, "
                f"sobretudo na Competência {comp}. Este trecho está prejudicando o texto"
            )
        else:
            efeito = (
                f"sem ele, a nota estimada cairia só {total:.0f} pts "
                f"(maior efeito na Competência {comp}), o menor peso entre os parágrafos"
            )
        feedbacks.append(
            f'[Análise de Parágrafo] O {i + 1}º parágrafo ("{trecho}...") é o que menos '
            f"contribui para a nota: {efeito}. Recomenda-se enriquecer a argumentação "
            f"ou o detalhamento deste trecho."
        )

    if not len([f for f in feedbacks if "Atenção" in f or "Bom" in f]):
        feedbacks.append(
            "Excelente desempenho! Todas as competências atingiram a nota máxima."
        )

    return feedbacks


@app.get("/")
def health_check():
    return {"status": "online", "device": str(DEVICE)}


def prever_brutos(tema: str, redacoes: List[str]) -> np.ndarray:
    """Notas contínuas (escala 0–200, sem conversão) de várias redações, num lote só."""
    inputs = TOKENIZER(
        [f"Tema: {tema} [SEP] Redação: {r}" for r in redacoes],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        saida = MODEL(inputs["input_ids"], inputs["attention_mask"]).float().cpu().numpy()
    return saida * 200.0


def prever_notas(tema: str, redacao: str) -> dict:
    """Roda o BERTimbau e devolve as notas por competência (múltiplos de 40) e a nota final."""
    raw_outputs = prever_brutos(tema, [redacao])[0] / 200.0
    rounded = converter_notas(raw_outputs)

    return {
        "c1_norma_culta": int(rounded[0]),
        "c2_compreensao_tema": int(rounded[1]),
        "c3_projeto_de_texto": int(rounded[2]),
        "c4_coesao": int(rounded[3]),
        "c5_proposta_intervencao": int(rounded[4]),
        "nota_final": int(np.sum(rounded)),
    }


def validar_entrada(payload: EssayRequest, acao: str):
    if len(payload.redacao.split()) < 30:
        raise HTTPException(
            status_code=400,
            detail=f"A redação é muito curta para ser {acao} (mínimo de 30 palavras).",
        )

    if MODEL is None or TOKENIZER is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo ainda não foi inicializado. Tente novamente em alguns instantes.",
        )

    problema = verificar_texto(payload.redacao)
    if problema:
        raise HTTPException(
            status_code=400,
            detail=f"O texto não parece uma redação e não será {acao}: {problema}",
        )


# Filtro de textos sem sentido.
# A rede neural dá nota a qualquer entrada (ex.: "palavra" repetida 40 vezes
# recebia 320 pontos). Os limites abaixo foram escolhidos com folga em relação
# ao pior caso das 4.570 redações do Essay-BR, para nenhuma redação real ser barrada:
#   palavras diferentes / total      pior redação 0,338  -> barra abaixo de 0,25
#   peso da palavra mais repetida    pior redação 12,3%  -> barra acima de 20%
#   palavras reconhecidas (3+ letras) pior redação 55,9% -> barra abaixo de 45%
#   fração de letras (sem espaços)   pior redação 92,8%  -> barra abaixo de 80%
LIMITE_DIVERSIDADE = 0.25
LIMITE_PALAVRA_REPETIDA = 0.20
LIMITE_PALAVRAS_CONHECIDAS = 0.45
LIMITE_LETRAS = 0.80


def verificar_texto(redacao: str):
    """Devolve uma explicação se o texto parecer sem sentido, ou None se estiver ok."""
    palavras = re.findall(r"[^\W\d_]+", redacao.lower())
    visiveis = sum(not ch.isspace() for ch in redacao)
    if not palavras or sum(ch.isalpha() for ch in redacao) / max(1, visiveis) < LIMITE_LETRAS:
        return "há poucas letras no texto (muitos números ou símbolos)."

    contagem = Counter(palavras)
    if len(contagem) / len(palavras) < LIMITE_DIVERSIDADE:
        return "há pouquíssimas palavras diferentes; o texto se repete demais."
    palavra, vezes = contagem.most_common(1)[0]
    if vezes / len(palavras) > LIMITE_PALAVRA_REPETIDA:
        return f'a palavra "{palavra}" ocupa {100 * vezes / len(palavras):.0f}% do texto.'

    longas = [p for p in palavras if len(p) >= 3]
    if longas:
        conhecidas = sum(1 for p in longas if len(TOKENIZER.tokenize(p)) == 1) / len(longas)
    else:
        conhecidas = 0.0
    if conhecidas < LIMITE_PALAVRAS_CONHECIDAS:
        return "a maioria das palavras não foi reconhecida como português."
    return None


@app.post("/api/v1/avaliar", response_model=EssayResponse)
def avaliar_redacao(payload: EssayRequest):
    validar_entrada(payload, "avaliada")

    notas = prever_notas(payload.tema, payload.redacao)
    feedback = gerar_diagnostico(
        payload.redacao, notas, MODEL, TOKENIZER, DEVICE, payload.tema
    )
    return {**notas, "feedback": feedback}


# Reescrita com o Gemini
INSTRUCAO_REESCRITA = """Você é um corretor experiente de redações do ENEM.
Sua tarefa é reescrever a redação do estudante para que ela alcance nota alta nas
cinco competências do ENEM, corrigindo os problemas apontados no diagnóstico.

Regras:
- Mantenha a tese, o ponto de vista e as ideias principais do autor. Melhore o texto,
  não troque o assunto nem a opinião defendida.
- Siga o tipo dissertativo-argumentativo: introdução com tese, dois parágrafos de
  desenvolvimento e conclusão, em 4 ou 5 parágrafos e no máximo 400 palavras.
- Use a norma culta, sem marcas de oralidade e sem primeira pessoa do singular.
- Use conectivos variados entre e dentro dos parágrafos e evite repetir palavras.
- Use apenas repertório sociocultural verdadeiro e amplamente conhecido (leis,
  obras, pensadores, fatos históricos). Não invente dados, números, pesquisas ou citações.
  Na dúvida sobre um fato (por exemplo, impostos, estatísticas ou datas), não o use.
- Termine com uma proposta de intervenção completa: agente, ação, meio/modo,
  finalidade e detalhamento, respeitando os direitos humanos.
- Não acrescente palavras ou expressões só porque costumam "dar nota": cada
  mudança deve melhorar de fato o argumento ou a clareza.

Formato da resposta: devolva somente o texto da redação reescrita, sem título,
sem comentários, sem markdown e com os parágrafos separados por uma linha em branco."""

NOMES_COMPETENCIAS = {
    "c1_norma_culta": "C1 - Norma culta",
    "c2_compreensao_tema": "C2 - Compreensão do tema e repertório",
    "c3_projeto_de_texto": "C3 - Projeto de texto e argumentação",
    "c4_coesao": "C4 - Coesão",
    "c5_proposta_intervencao": "C5 - Proposta de intervenção",
}


def obter_cliente_gemini():
    """Cria o cliente do Gemini na primeira reescrita, a partir de GEMINI_API_KEY."""
    global GEMINI_CLIENT
    if GEMINI_CLIENT is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Reescrita indisponível: defina GEMINI_API_KEY no arquivo "
                    "Back-End/.env e reinicie o servidor."
                ),
            )
        # As novas tentativas são feitas por com_backoff(); o SDK tenta só uma vez,
        # para os dois mecanismos não se somarem
        GEMINI_CLIENT = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(
                retry_options=genai_types.HttpRetryOptions(attempts=1)
            ),
        )
    return GEMINI_CLIENT


def montar_prompt_reescrita(tema: str, redacao: str, notas: dict, feedback: List[str]) -> str:
    linhas_notas = "\n".join(
        f"- {nome}: {notas[chave]}/200" for chave, nome in NOMES_COMPETENCIAS.items()
    )
    linhas_feedback = "\n".join(f"- {f}" for f in feedback)
    return (
        f"Tema da redação: {tema or '(não informado)'}\n\n"
        f"Notas estimadas pela rede neural (nota final {notas['nota_final']}/1000):\n"
        f"{linhas_notas}\n\n"
        f"Diagnóstico:\n{linhas_feedback}\n\n"
        f"Redação original do estudante:\n<<<\n{redacao}\n>>>"
    )


def erro_temporario(exc: Exception) -> bool:
    """429 (limite de uso) e 5xx (sobrecarga ou falha do servidor) costumam passar sozinhos."""
    codigo = getattr(exc, "code", None)
    return isinstance(exc, genai_errors.APIError) and (codigo == 429 or (codigo or 0) >= 500)


def com_backoff(funcao, tentativas=None, espera_inicial=None, espera_maxima=None,
                dormir=time.sleep, sortear=random.uniform):
    """Executa funcao() e, em erro temporário, tenta de novo com backoff exponencial.

    A espera dobra a cada tentativa (1 s, 2 s, 4 s, 8 s, ..., até espera_maxima) e
    recebe uma variação aleatória ("jitter": entre metade e o valor cheio), para
    várias requisições não repetirem todas no mesmo instante. Erros permanentes
    (chave inválida, modelo inexistente, pedido malformado) não são repetidos.
    """
    tentativas = tentativas or GEMINI_TENTATIVAS
    espera_inicial = espera_inicial or GEMINI_ESPERA_INICIAL
    espera_maxima = espera_maxima or GEMINI_ESPERA_MAXIMA
    for n in range(tentativas):
        try:
            return funcao()
        except genai_errors.APIError as exc:
            if not erro_temporario(exc) or n == tentativas - 1:
                raise
            base = min(espera_maxima, espera_inicial * 2 ** n)
            espera = sortear(base / 2, base)
            print(f"Gemini respondeu {exc.code}; tentativa {n + 2} de {tentativas} em {espera:.1f}s")
            dormir(espera)


def chamar_gemini(prompt: str, instrucao: str, **config):
    """Chama o Gemini (com backoff exponencial) e converte os erros em respostas HTTP claras."""
    cliente = obter_cliente_gemini()
    try:
        return com_backoff(lambda: cliente.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(system_instruction=instrucao, **config),
        ))
    except genai_errors.ClientError as exc:
        if exc.code == 429:
            raise HTTPException(
                status_code=429,
                detail=f"Limite de uso do Gemini atingido (tentamos {GEMINI_TENTATIVAS} vezes). Aguarde um pouco e tente de novo.",
            )
        if exc.code == 404:
            raise HTTPException(
                status_code=502,
                detail=f"O modelo '{GEMINI_MODEL}' não existe para esta chave. Ajuste GEMINI_MODEL no .env.",
            )
        if exc.code in (400, 401, 403):
            raise HTTPException(
                status_code=502,
                detail=f"O Gemini recusou a requisição ({exc.code}). Confira a GEMINI_API_KEY. Detalhe: {exc.message}",
            )
        raise HTTPException(status_code=502, detail=f"Erro do Gemini ({exc.code}): {exc.message}")
    except genai_errors.APIError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"O Gemini está indisponível no momento ({exc.code}), mesmo após {GEMINI_TENTATIVAS} tentativas. Tente novamente em alguns minutos.",
        )


@app.post("/api/v1/reescrever", response_model=RewriteResponse)
def reescrever_redacao(payload: EssayRequest):
    validar_entrada(payload, "reescrita")
    obter_cliente_gemini()  # falha cedo, antes de rodar o modelo, se não houver chave

    # As notas e o diagnóstico do BERTimbau guiam o que o Gemini deve corrigir
    notas = prever_notas(payload.tema, payload.redacao)
    feedback = gerar_diagnostico(
        payload.redacao, notas, MODEL, TOKENIZER, DEVICE, payload.tema
    )
    prompt = montar_prompt_reescrita(payload.tema, payload.redacao, notas, feedback)
    resposta = chamar_gemini(prompt, INSTRUCAO_REESCRITA, temperature=0.7)

    texto = (resposta.text or "").strip()
    if not texto:
        raise HTTPException(
            status_code=502,
            detail="O Gemini não devolveu texto (a resposta pode ter sido bloqueada). Tente novamente.",
        )

    # Reavalia a versão reescrita com o mesmo modelo, para mostrar "antes e depois"
    return {
        "redacao_reescrita": texto,
        "notas_original": notas,
        "notas_reescrita": prever_notas(payload.tema, texto),
    }


# Apontamento de trechos com problema (Gemini)
class Apontamento(BaseModel):
    competencia: int
    trecho: str
    problema: str
    sugestao: str


class ApontamentosResponse(BaseModel):
    apontamentos: List[Apontamento]
    descartados: int


INSTRUCAO_APONTAMENTOS = """Você é um corretor experiente de redações do ENEM.
Leia a redação do estudante e aponte os trechos concretos que fazem a redação perder
pontos, indicando a competência afetada:
1 = norma culta (gramática, ortografia, pontuação, concordância, regência, informalidade)
2 = compreensão do tema e repertório sociocultural
3 = seleção e organização dos argumentos (projeto de texto)
4 = coesão (conectivos, repetições, retomadas)
5 = proposta de intervenção (agente, ação, meio/modo, finalidade, detalhamento)

Regras:
- "trecho" deve ser uma cópia EXATA de um pedaço curto do texto do estudante (de
  3 a 25 palavras), sem alterar nenhuma letra, acento ou pontuação.
- "problema" explica em uma frase o que está errado; "sugestao" mostra como corrigir
  (de preferência reescrevendo o trecho).
- Aponte de 3 a 8 problemas, os mais importantes primeiro. Não aponte problemas
  inexistentes só para completar a lista.
- Julgue o conteúdo e a forma, não a presença de palavras ou expressões "da moda".
- Ao sugerir repertório, cite só fatos de que tenha certeza. Não sugira dados,
  números, leis ou impostos que você não saiba que são verdadeiros.
- Escreva em português."""


def normalizar(texto: str) -> str:
    return " ".join(texto.split()).lower()


@app.post("/api/v1/apontar", response_model=ApontamentosResponse)
def apontar_problemas(payload: EssayRequest):
    validar_entrada(payload, "analisada")
    obter_cliente_gemini()

    notas = prever_notas(payload.tema, payload.redacao)
    linhas_notas = "\n".join(
        f"- {nome}: {notas[chave]}/200" for chave, nome in NOMES_COMPETENCIAS.items()
    )
    prompt = (
        f"Tema da redação: {payload.tema or '(não informado)'}\n\n"
        f"Notas estimadas pela rede neural (use só como referência):\n{linhas_notas}\n\n"
        f"Redação do estudante:\n<<<\n{payload.redacao}\n>>>"
    )
    resposta = chamar_gemini(
        prompt,
        INSTRUCAO_APONTAMENTOS,
        temperature=0.2,
        response_mime_type="application/json",
        response_schema=list[Apontamento],
    )

    itens = resposta.parsed
    if itens is None:
        try:
            itens = [Apontamento(**d) for d in json.loads(resposta.text or "[]")]
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=502,
                detail="O Gemini devolveu uma resposta em formato inesperado. Tente novamente.",
            )

    # Só mantém trechos que existem de fato no texto (o modelo pode inventar citações)
    texto_norm = normalizar(payload.redacao)
    validos = [
        a for a in itens
        if 1 <= a.competencia <= 5 and a.trecho.strip() and normalizar(a.trecho) in texto_norm
    ]
    return {"apontamentos": validos, "descartados": len(itens) - len(validos)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
