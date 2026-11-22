import json
import os
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
from torch import nn
from transformers import AutoModel, AutoTokenizer

# Carrega variáveis do arquivo .env (ex.: GEMINI_API_KEY), se existir
load_dotenv()


# Arquitetura do Modelo
class EnemEssayEvaluator(nn.Module):
    def __init__(self, model_name="neuralmind/bert-base-portuguese-cased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.2)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 5)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.regressor(self.dropout(cls_output))


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

class RewriteResponse(BaseModel):
    redacao_reescrita: str


# Tamanho máximo da sequência (tema + redação) em tokens.
# Ao carregar o modelo, é substituído pelo valor de saved_model/config.json,
# gravado pelo train.py, para que a inferência use o mesmo valor do treino.
# 384 é o valor usado por modelos treinados antes de existir o config.json.
MAX_LEN = 384

# Configuração do Gemini (usado na reescrita da redação)
# GEMINI_MODEL pode ser trocado no .env por qualquer modelo disponível para a sua chave
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or "gemini-3.5-flash-lite"

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
    MODEL_PATH = "./saved_model/model_weights.pt"
    TOKENIZER_PATH = "./saved_model/tokenizer"
    CONFIG_PATH = "./saved_model/config.json"

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

    # Análise de Impacto por Parágrafo
    paragrafos = [p.strip() for p in redacao.split("\n") if p.strip()]
    if len(paragrafos) >= 2:
        nota_base = notas["nota_final"]
        impactos = []

        for i, p in enumerate(paragrafos):
            texto_sem_p = "\n".join(paragrafos[:i] + paragrafos[i + 1 :])
            inputs = tokenizer(
                f"Tema: {tema} [SEP] Redação: {texto_sem_p}",
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out = (
                    model(inputs["input_ids"], inputs["attention_mask"])
                    .cpu()
                    .numpy()[0]
                )
            nota_sem_p = int(np.sum(converter_notas(out)))

            # Verifica qual parágrafo causou a maior queda na nota final ao ser removido
            queda = nota_base - nota_sem_p
            impactos.append((i + 1, p[:80], queda))

        # Ordena os impactos do maior para o menor (maior queda = maior impacto)
        impactos.sort(key=lambda x: x[2], reverse=True)

        # Identifica o parágrafo com menor impacto (ou seja, que menos contribuiu para a nota final)
        mais_fraco = impactos[-1]

        feedbacks.append(
            f'[Análise de Parágrafo] O {mais_fraco[0]}º parágrafo ("{mais_fraco[1]}...") '
            f"apresentou menor impacto na pontuação da rede neural (variação de {mais_fraco[2]} pts). "
            f"Recomenda-se enriquecer a argumentação ou o detalhamento deste trecho."
        )

    if not len([f for f in feedbacks if "Atenção" in f or "Bom" in f]):
        feedbacks.append(
            "Excelente desempenho! Todas as competências atingiram a nota máxima."
        )

    return feedbacks


@app.get("/")
def health_check():
    return {"status": "online", "device": str(DEVICE)}


def prever_notas(tema: str, redacao: str) -> dict:
    """Roda o BERTimbau e devolve as notas por competência (múltiplos de 40) e a nota final."""
    combined_text = f"Tema: {tema} [SEP] Redação: {redacao}"

    inputs = TOKENIZER(
        combined_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        raw_outputs = (
            MODEL(inputs["input_ids"], inputs["attention_mask"]).cpu().numpy()[0]
        )

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
        GEMINI_CLIENT = genai.Client(api_key=api_key)
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


@app.post("/api/v1/reescrever", response_model=RewriteResponse)
def reescrever_redacao(payload: EssayRequest):
    validar_entrada(payload, "reescrita")
    cliente = obter_cliente_gemini()

    # As notas e o diagnóstico do BERTimbau guiam o que o Gemini deve corrigir
    notas = prever_notas(payload.tema, payload.redacao)
    feedback = gerar_diagnostico(
        payload.redacao, notas, MODEL, TOKENIZER, DEVICE, payload.tema
    )
    prompt = montar_prompt_reescrita(payload.tema, payload.redacao, notas, feedback)

    try:
        resposta = cliente.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=INSTRUCAO_REESCRITA,
                temperature=0.7,
            ),
        )
    except genai_errors.ClientError as exc:
        if exc.code == 429:
            raise HTTPException(
                status_code=429,
                detail="Limite de uso do Gemini atingido. Aguarde um pouco e tente de novo.",
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
            detail=f"O Gemini está indisponível no momento ({exc.code}). Tente novamente.",
        )

    texto = (resposta.text or "").strip()
    if not texto:
        raise HTTPException(
            status_code=502,
            detail="O Gemini não devolveu texto (a resposta pode ter sido bloqueada). Tente novamente.",
        )
    return {"redacao_reescrita": texto}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
