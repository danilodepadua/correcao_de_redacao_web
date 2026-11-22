"""Baixa o modelo já treinado (saved_model/) da Release do GitHub, sem precisar treinar.

Uso (dentro da pasta Back-End):
    python baixar_modelo.py            # baixa a versão padrão
    python baixar_modelo.py --force    # substitui um saved_model/ que já exista

O arquivo é conferido pelo SHA-256 antes de ser extraído.
"""
import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

REPO = "danilodepadua/correcao_de_redacao_web"
VERSAO = "modelo-v3"
ARQUIVO = "saved_model.zip"
SHA256 = "b25e2e89e2affe56e733ea8ba1b7ea194b7ab0c6c3086d40afc9c2b3a5c22472"
URL = f"https://github.com/{REPO}/releases/download/{VERSAO}/{ARQUIVO}"

PASTA_BACKEND = os.path.dirname(os.path.abspath(__file__))
DESTINO = os.path.join(PASTA_BACKEND, "saved_model")


def baixar(url: str, caminho: str):
    with urllib.request.urlopen(url) as resp, open(caminho, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        baixado = 0
        while bloco := resp.read(1 << 20):
            out.write(bloco)
            baixado += len(bloco)
            if total:
                print(f"\r  {baixado / 2**20:6.1f} / {total / 2**20:.1f} MB "
                      f"({100 * baixado / total:5.1f}%)", end="", flush=True)
    print()


def sha256(caminho: str) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        while bloco := f.read(1 << 20):
            h.update(bloco)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="substitui um saved_model/ existente")
    args = ap.parse_args()

    if os.path.exists(DESTINO) and not args.force:
        sys.exit(f"Já existe {DESTINO}. Use --force para substituí-lo.")

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, ARQUIVO)
        print(f"Baixando {URL}")
        baixar(URL, zip_path)

        print("Conferindo o arquivo...")
        if sha256(zip_path) != SHA256:
            sys.exit("O arquivo baixado está corrompido (SHA-256 diferente). Tente de novo.")

        extraido = os.path.join(tmp, "extraido")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extraido)
        origem = os.path.join(extraido, "saved_model")
        if not os.path.exists(os.path.join(origem, "model_weights.pt")):
            sys.exit("O arquivo baixado não contém saved_model/model_weights.pt.")

        if os.path.exists(DESTINO):
            shutil.rmtree(DESTINO)
        shutil.move(origem, DESTINO)

    print(f"Modelo pronto em {DESTINO}. Rode: python main.py")


if __name__ == "__main__":
    main()
