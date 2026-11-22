import os
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torch import nn
from transformers import AutoModel, AutoTokenizer


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


# Estado Global
MODEL = None
TOKENIZER = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Gerenciador de Ciclo de Vida
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, TOKENIZER
    MODEL_PATH = "./saved_model/model_weights.pt"
    TOKENIZER_PATH = "./saved_model/tokenizer"

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Arquivo não encontrado em '{MODEL_PATH}'. "
            "Verifique se a pasta 'saved_model' foi extraída na raiz do projeto."
        )

    print(f"Carregando modelo e tokenizador em {DEVICE}...")
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
                max_length=512,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out = (
                    model(inputs["input_ids"], inputs["attention_mask"])
                    .cpu()
                    .numpy()[0]
                )
            nota_sem_p = int(np.sum(np.round(np.clip(out * 200.0, 0, 200) / 40.0) * 40))

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


@app.post("/api/v1/avaliar", response_model=EssayResponse)
def avaliar_redacao(payload: EssayRequest):
    if len(payload.redacao.split()) < 30:
        raise HTTPException(
            status_code=400,
            detail="A redação é muito curta para ser avaliada (mínimo de 30 palavras).",
        )

    if MODEL is None or TOKENIZER is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo ainda não foi inicializado. Tente novamente em alguns instantes.",
        )

    combined_text = f"Tema: {payload.tema} [SEP] Redação: {payload.redacao}"

    inputs = TOKENIZER(
        combined_text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        raw_outputs = (
            MODEL(inputs["input_ids"], inputs["attention_mask"]).cpu().numpy()[0]
        )

    # Converte [0, 1] para múltiplos de 40 na escala [0, 200]
    raw_scores = raw_outputs * 200.0
    clipped = np.clip(raw_scores, 0, 200)
    rounded = (np.round(clipped / 40.0) * 40).astype(int)

    notas = {
        "c1_norma_culta": int(rounded[0]),
        "c2_compreensao_tema": int(rounded[1]),
        "c3_projeto_de_texto": int(rounded[2]),
        "c4_coesao": int(rounded[3]),
        "c5_proposta_intervencao": int(rounded[4]),
        "nota_final": int(np.sum(rounded)),
    }
    feedback = gerar_diagnostico(
        payload.redacao, notas, MODEL, TOKENIZER, DEVICE, payload.tema
    )
    return {**notas, "feedback": feedback}

@app.post("/api/v1/reescrever", response_model=RewriteResponse)
def reescrever_redacao(payload: EssayRequest):
    if len(payload.redacao.split()) < 30:
        raise HTTPException(
            status_code=400,
            detail="A redação é muito curta para ser reescrita (mínimo de 30 palavras).",
        )
    # TODO: integrar modelo gerador de reescrita (ex.: T5/LLM/API)
    raise HTTPException(
        status_code=501,
        detail="Endpoint de reescrita ainda não implementado no back-end.",
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
