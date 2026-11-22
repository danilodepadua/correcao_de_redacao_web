# Correção de Redação
Site de correção de redação, que consiste em pegar uma redação não avaliada previamente e dar nota para ela em diferentes categorias.

## Integrantes
- Danilo de Pádua
- Pedro Henrique
- Sonael

## Dataset
`https://github.com/rafaelanchieta/essay`

## Modelo
`neuralmind/bert-base-portuguese-cased`

## Modelo treinado

Os arquivos gerados em `Back-End/saved_model/` não são versionados por serem grandes. Para gerá-los localmente, execute `python train.py` dentro da pasta `Back-End`.

## Reescrita da redação (Gemini)

O endpoint `POST /api/v1/reescrever` usa o Gemini para gerar uma nova versão da redação. Antes de chamar o Gemini, a redação passa pelo BERTimbau, e as notas por competência e o diagnóstico entram no prompt, para que a reescrita corrija os pontos apontados mantendo as ideias do autor.

Para ativar:

1. Gere uma chave em https://aistudio.google.com/apikey.
2. Copie `Back-End/.env.example` para `Back-End/.env` e preencha `GEMINI_API_KEY`. O `.env` não é versionado.
3. (Opcional) Troque `GEMINI_MODEL` no `.env`. O padrão é `gemini-3.5-flash-lite`. Para listar os modelos liberados para a sua chave: `python -c "from dotenv import load_dotenv; load_dotenv(); from google import genai; c = genai.Client(); [print(m.name) for m in c.models.list()]"`
4. Reinicie o servidor (`python main.py`).

Sem a chave, o endpoint responde 503 com uma mensagem explicando o que configurar.

## Como rodar o back-end

```
cd Back-End
python -m venv venv
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python train.py   # gera saved_model/ (uma vez)
python main.py    # API em http://localhost:8000
```
