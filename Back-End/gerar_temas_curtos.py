"""Gera um título curto para cada um dos 86 temas do Essay-BR (temas_curtos.csv).

O Essay-BR só traz o texto motivador de cada proposta, mas no site o usuário
digita um título curto ("Desafios para a formação de leitores no Brasil"). Os
títulos curtos permitem treinar e avaliar o modelo com o mesmo tipo de tema que
ele recebe no uso real. Os títulos são gerados pelo Gemini a partir do texto de
cada proposta e ficam salvos no repositório, então o script só precisa rodar uma vez.

Uso:  python gerar_temas_curtos.py   (precisa de GEMINI_API_KEY no .env)
"""
import ast
import csv
import os
import time

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

load_dotenv()
MODELO = os.getenv("GEMINI_MODEL") or "gemini-3.5-flash-lite"

INSTRUCAO = """Você recebe o texto motivador de propostas de redação no estilo ENEM.
Para cada proposta, escreva o tema da redação como um título curto, do jeito que
uma banca ou um estudante escreveria (por exemplo: "Desafios para a formação de
leitores no Brasil"). Use de 4 a 14 palavras, em português, sem aspas e sem ponto
final. Se o texto já trouxer o tema explícito, use-o."""


class Tema(BaseModel):
    id: int
    tema: str


def texto_proposta(desc) -> str:
    s = str(desc).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            s = " ".join(map(str, ast.literal_eval(s)))
        except (SyntaxError, ValueError):
            pass
    return " ".join(s.split())[:2500]


def main():
    prompts = pd.read_csv("prompts.csv")
    cliente = genai.Client()
    resultado = {}
    for inicio in range(0, len(prompts), 15):
        lote = prompts.iloc[inicio:inicio + 15]
        corpo = "\n\n".join(f"### Proposta id={r.id}\n{texto_proposta(r.description)}" for r in lote.itertuples())
        for tentativa in range(6):
            try:
                resp = cliente.models.generate_content(
                    model=MODELO,
                    contents=corpo,
                    config=types.GenerateContentConfig(
                        system_instruction=INSTRUCAO, temperature=0.2,
                        response_mime_type="application/json", response_schema=list[Tema],
                    ),
                )
                break
            except errors.ServerError as exc:  # 503: Gemini sobrecarregado, tenta de novo
                espera = 10 * (tentativa + 1)
                print(f"  Gemini indisponível ({exc.code}); nova tentativa em {espera}s")
                time.sleep(espera)
        else:
            raise SystemExit("O Gemini continuou indisponível; tente mais tarde.")
        for t in resp.parsed:
            resultado[t.id] = " ".join(t.tema.split()).strip(' ".')
        print(f"{min(inicio + 15, len(prompts))}/{len(prompts)} temas")

    faltando = sorted(set(prompts.id) - set(resultado))
    if faltando:
        raise SystemExit(f"Faltaram os temas {faltando}; rode de novo.")
    with open("temas_curtos.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "tema_curto"])
        for pid in sorted(resultado):
            w.writerow([pid, resultado[pid]])
    print("Gravado em temas_curtos.csv")


if __name__ == "__main__":
    main()
