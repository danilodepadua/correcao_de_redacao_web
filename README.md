# Nota Mil: correção automática de redações do ENEM

Site que recebe uma redação que ainda não foi corrigida e estima a nota dela nas 5 competências do ENEM, usando uma rede neural treinada com redações reais corrigidas por humanos. Além da nota, o site aponta o que está errado no texto, mostra exemplos de redações boas e ruins e gera uma versão reescrita da redação.

## Integrantes

- Danilo de Pádua
- Pedro Henrique
- Sonael

## Problema

Corrigir uma redação no modelo do ENEM exige um corretor treinado e leva tempo, então o estudante que treina sozinho quase nunca recebe uma correção detalhada. O objetivo do projeto é, a partir de uma redação qualquer:

1. **Avaliar** o texto, dando uma nota de 0 a 200 para cada uma das 5 competências do ENEM (nota final de 0 a 1000).
2. **Mostrar exemplos** de redações boas, medianas e ruins, com as notas e os comentários.
3. **Mostrar o que está errado**, apontando os problemas de cada competência e os trechos do texto onde eles aparecem.
4. **Refazer a redação**, gerando uma versão melhorada que mantém as ideias do autor.
5. **Reconhecer o que eleva o nível** de uma redação, evitando que o modelo premie características superficiais (viés do modelo).

Do ponto de vista de aprendizado de máquina, a avaliação é uma **regressão com 5 saídas**: a entrada é o tema mais o texto da redação, e a saída são as 5 notas.

As 5 competências do ENEM:

| Competência | O que avalia |
|---|---|
| C1 | Domínio da norma culta da língua escrita |
| C2 | Compreensão da proposta e uso de repertório sociocultural |
| C3 | Seleção e organização de argumentos (projeto de texto) |
| C4 | Coesão: conectivos e retomadas |
| C5 | Proposta de intervenção para o problema discutido |

## Base de dados e rótulos

Usamos o **Essay-BR** (https://github.com/rafaelanchieta/essay), um conjunto público de redações do tipo ENEM em português.

**Os rótulos já vêm no dataset:** cada redação tem as 5 notas por competência dadas por corretores humanos. Não é preciso transcrever nem rotular nada à mão, e a base é usada diretamente no treino.

| Item | Valor |
|---|---|
| Redações | 4.570 |
| Temas | 86 (enunciados em `Back-End/prompts.csv`) |
| Rótulos por redação | 5 notas (C1 a C5), cada uma em {0, 40, 80, 120, 160, 200} |
| Nota final | média 614, mediana 600, de 0 a 1000 |
| Tamanho das redações | média de 289 palavras (377 tokens do BERTimbau) |

Arquivos usados:

- `Back-End/essay-br.csv`: colunas `prompt` (id do tema), `title`, `essay` (lista de parágrafos), `competence` (as 5 notas) e `score` (nota final).
- `Back-End/prompts.csv`: colunas `id` e `description` (texto da proposta de cada tema).

As notas se concentram em 120 e 160 em todas as competências. Notas 0 e 200 são raras: em C1, por exemplo, só 97 redações têm 0 e 122 têm 200. Esse desbalanceamento faz o modelo tender a prever notas intermediárias.

Divisão: 80% treino (3.656), 10% validação (457) e 10% teste (457), sorteados com `random_state=42`.

## Modelos

### 1. Avaliador: BERTimbau ajustado (rede neural treinada pelo grupo)

- **Base:** `neuralmind/bert-base-portuguese-cased` (BERTimbau Base, 12 camadas), um BERT pré-treinado em português.
- **Entrada:** `"Tema: <proposta> [SEP] Redação: <texto>"`, cortada em 512 tokens (`MAX_LEN`, o limite do BERT). A proposta é resumida a 300 caracteres.
- **Cabeça:** dropout de 0,2 e uma camada linear de 768 para 5 saídas sobre o token `[CLS]`.
- **Alvo:** as 5 notas divididas por 200 (escala de 0 a 1). Na saída, as previsões voltam para a escala de 0 a 200 e são convertidas em 0, 40, …, 200 por cortes calibrados (veja abaixo).
- **Treino:**
  - todas as 12 camadas do BERT são ajustadas;
  - perda MSE e otimizador AdamW (lr 2e-5, weight decay 0,01), com batch 8 e 4 épocas;
  - a taxa de aprendizado sobe nos primeiros 10% dos passos e depois cai de forma linear;
  - usa precisão mista bf16 na GPU;
  - é salvo o checkpoint com o maior QWK de validação.
- **Calibração dos cortes** (`calibrate.py`):
  - o arredondamento simples (múltiplo de 40 mais próximo) concentra as notas no meio;
  - depois do treino, os 5 cortes de cada competência são ajustados no conjunto de **validação** para maximizar o QWK;
  - os cortes ficam em `saved_model/config.json`, junto com o `max_len`, e a API lê os dois desse arquivo.
- **Métrica:** QWK (*Quadratic Weighted Kappa*), a métrica padrão de correção automática de redações. Ela mede a concordância com as notas humanas e pune mais os erros grandes. Vale 1 para concordância perfeita e 0 para concordância ao acaso.
- **Código:** `Back-End/train.py` (treino; todos os hiperparâmetros podem ser passados por linha de comando, veja `python train.py --help`) e `Back-End/calibrate.py` (calibração).

### 2. Reescrita: Gemini (modelo pronto, via API)

A reescrita usa o Gemini (`gemini-3.5-flash-lite` por padrão). Antes, a redação passa pelo BERTimbau, e as notas e o diagnóstico entram no prompt, para que a reescrita corrija os pontos que o avaliador apontou. As instruções pedem ao Gemini que:

- mantenha as ideias do autor;
- não invente dados nem citações;
- faça uma proposta de intervenção completa;
- não acrescente expressões só porque costumam "dar nota".

### 3. Verificações por regras (sem aprendizado)

O site também confere o texto com regras fixas:

- número de parágrafos e tamanho do texto;
- conectivos no início dos parágrafos;
- palavras repetidas;
- marcas de oralidade e primeira pessoa;
- os 5 elementos da proposta de intervenção.

Essas regras apontam trechos exatos do texto, o que a rede neural sozinha não faz.

## Resultados

### Experimentos

Todos os experimentos usaram a mesma divisão de dados e foram comparados **só pela validação**. O conjunto de teste ficou de fora até a escolha final.

| Experimento | Tokens | Camadas ajustadas | Épocas | Agendador de lr | Checkpoint por | QWK (validação) |
|---|---|---|---|---|---|---|
| E0: modelo original | 384 | 3 últimas | 3 | não | perda | 0,497 |
| E1 | 512 | 3 últimas | 3 | não | perda | 0,514 |
| E2 | 384 | todas | 4 | sim | QWK | 0,551 |
| **E3: modelo final** | **512** | **todas** | **4** | **sim** | **QWK** | **0,558** |

O que mais ajudou foi ajustar o BERT inteiro (+0,05). Usar 512 tokens também ajuda (+0,01 a +0,02), porque 46% das redações passavam de 384 tokens e perdiam o final. Cada configuração foi rodada uma vez só, então diferenças de cerca de 0,01 podem ser ruído.

Também testamos a calibração dos cortes no modelo do E2: o QWK de teste subiu de 0,507 para 0,574.

### Modelo final (E3 + calibração)

Treino em uma RTX 4050 Laptop (6 GB), em 10 min 51 s. O melhor checkpoint foi o da época 3.

| Época | Perda de treino | Perda de validação | QWK médio (validação) |
|---|---|---|---|
| 1 | 0,0697 | 0,0291 | 0,514 |
| 2 | 0,0323 | 0,0281 | 0,520 |
| 3 | 0,0266 | **0,0262** | **0,558** |
| 4 | 0,0235 | 0,0285 | 0,544 |

**QWK no conjunto de teste** (457 redações que o modelo nunca viu):

| Modelo | C1 | C2 | C3 | C4 | C5 | Média |
|---|---|---|---|---|---|---|
| Original (384 tokens, 3 camadas, arredondamento simples) | 0,342 | 0,431 | 0,395 | 0,496 | 0,531 | 0,439 |
| Final, arredondamento simples | 0,469 | 0,480 | 0,499 | 0,547 | 0,603 | 0,520 |
| **Final, cortes calibrados** | **0,522** | **0,557** | **0,538** | **0,595** | **0,636** | **0,570** |

MSE de teste do modelo final: 0,0276 (na escala de 0 a 1).

Como ler: o QWK subiu de 0,44 para 0,57, uma concordância **moderada** com os corretores humanos. A maior melhora foi na C1 (norma culta, de 0,34 para 0,52), que dependia mais do texto inteiro. Na validação, os cortes calibrados chegam a 0,643, mas esse número é otimista porque os cortes foram ajustados nela. O valor honesto é o do teste.

As notas também ficaram mais espalhadas. Um texto sem sentido ("palavra" repetida 40 vezes) caiu de 320 para 80 pontos. Nas 3 redações de exemplo do site (referências 960, 560 e 160), o modelo deu 720, 280 e 40: a ordem está certa, mas as redações mediana e fraca ficaram abaixo da referência.

## Funcionalidades do site

| Funcionalidade | Situação |
|---|---|
| Avaliar uma redação: 5 notas, nota final e diagnóstico por competência | Pronto |
| Análise de impacto por parágrafo (nota do texto sem cada parágrafo) | Pronto, mas pouco informativa (ver limitações) |
| Redação anotada e checklist do corretor (regras) | Pronto |
| Exemplos de redações boa, mediana e fraca | Pronto (textos escritos para o site) |
| Página de dicas e página sobre o modelo | Pronto |
| Reescrever a redação com o Gemini | Pronto |
| Reconhecer características que elevam o nível, evitando viés | A fazer |

Páginas: **Corrigir** (`/`), **Exemplos** (`/exemplos`), **Dicas** (`/dicas`) e **O modelo** (`/sobre`). O endereço da API pode ser trocado no próprio site, em "Servidor".

## Arquitetura

```
Navegador (Front-End: React + TanStack Start, porta 8080)
   │  POST /api/v1/avaliar     { tema, redacao }
   │  POST /api/v1/reescrever  { tema, redacao }
   ▼
API (Back-End: FastAPI, porta 8000)
   ├── BERTimbau ajustado (saved_model/) → 5 notas + diagnóstico
   └── Gemini (API do Google)            → redação reescrita
```

### Endpoints da API

| Método | Rota | Entrada | Saída |
|---|---|---|---|
| GET | `/` | | `{status, device}` |
| POST | `/api/v1/avaliar` | `{tema, redacao}` | `c1_norma_culta` … `c5_proposta_intervencao`, `nota_final`, `feedback` |
| POST | `/api/v1/reescrever` | `{tema, redacao}` | `{redacao_reescrita}` |

A redação precisa ter pelo menos 30 palavras. Com a API rodando, a documentação interativa fica em http://localhost:8000/docs.

## Como rodar

Pré-requisitos: Python 3.12, Node.js 20 ou mais novo e, de preferência, uma GPU NVIDIA.

### Back-End

```
cd Back-End
python -m venv venv
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python baixar_modelo.py   # baixa o modelo já treinado (~192 MB), sem precisar treinar
python main.py            # API em http://localhost:8000
```

Os pesos (`Back-End/saved_model/`) não ficam no git por serem grandes. O modelo final, já calibrado, está publicado na Release [`modelo-v1`](https://github.com/danilodepadua/correcao_de_redacao_web/releases/tag/modelo-v1), e o `baixar_modelo.py` baixa, confere (SHA-256) e extrai esse arquivo. Os pesos da Release estão em float16 para ocupar metade do espaço: em 914 redações de validação e teste, só 2 tiveram uma nota diferente da versão em float32.

Para **treinar do zero** em vez de baixar (cerca de 12 min numa GPU):

```
python train.py       # treina e gera saved_model/
python calibrate.py   # ajusta os cortes das notas na validação
```

Para treinar com outra configuração, passe os parâmetros por linha de comando (por exemplo, `python train.py --max-len 384 --frozen-layers 9`) e, para experimentos, use `--out runs/<nome> --no-test`.

### Reescrita com o Gemini

1. Gere uma chave em https://aistudio.google.com/apikey.
2. Copie `Back-End/.env.example` para `Back-End/.env` e preencha `GEMINI_API_KEY`. O `.env` não é versionado.
3. (Opcional) Troque `GEMINI_MODEL` no `.env`. O padrão é `gemini-3.5-flash-lite`. Para listar os modelos liberados para a sua chave: `python -c "from dotenv import load_dotenv; load_dotenv(); from google import genai; c = genai.Client(); [print(m.name) for m in c.models.list()]"`
4. Reinicie o servidor (`python main.py`).

Sem a chave, a avaliação funciona normalmente e só a reescrita responde 503, com uma mensagem explicando o que configurar.

### Front-End

```
cd Front-End
npm install
npm run dev       # site em http://localhost:8080
npm test          # testes das regras do site
```

## Limitações conhecidas

- **Texto cortado:** mesmo com 512 tokens (o limite do BERT), 11,6% das redações perdem o final, onde fica a proposta de intervenção (C5).
- **Notas ainda puxadas para o meio:** a calibração dos cortes melhorou isso, mas o modelo ainda é treinado com MSE em dados com poucas notas 0 e 200.
- **Temas repetidos nos três conjuntos:** a divisão é aleatória por redação, então os mesmos temas aparecem no treino e no teste. O desempenho em temas inéditos pode ser menor.
- **Uma rodada por experimento:** cada configuração foi treinada com uma única semente, então diferenças pequenas entre experimentos podem ser ruído.
- **Texto sem sentido ainda recebe nota:** caiu de 320 para 80 pontos, mas ainda faltam filtros de entrada.
- **Análise por parágrafo pouco informativa:** retirar um parágrafo costuma não mudar a nota convertida, e em caso de empate o último parágrafo é apontado.
- **Feedback genérico:** as mensagens são fixas por faixa de nota. Os trechos exatos vêm só das verificações por regras.
- **Repertório da reescrita:** o Gemini pode errar fatos. Em dois testes, a versão reescrita afirmou que livros têm "alta carga tributária", o que é falso, porque a Constituição dá imunidade tributária aos livros.

## Próximos passos

- Comparar com um modelo de referência simples (TF-IDF + regressão linear) para medir o ganho do BERT.
- Avaliar com divisão por tema (temas do teste ausentes do treino) e com várias sementes.
- Lidar com redações maiores que 512 tokens (por exemplo, dividir em dois trechos e combinar).
- Investigar o viés: medir se o modelo premia o tamanho do texto ou o número de conectivos sem melhora real do argumento.
- Filtrar entradas sem sentido antes da rede.
- Reavaliar a redação reescrita com o BERTimbau para mostrar "antes e depois".
- Trocar os exemplos do site por redações reais do Essay-BR.

## Estrutura do repositório

```
Back-End/
  train.py            treino e avaliação do BERTimbau
  calibrate.py        calibração dos cortes das notas
  baixar_modelo.py    baixa o modelo treinado da Release do GitHub
  main.py             API (avaliação e reescrita)
  essay-br.csv        redações e notas (Essay-BR)
  prompts.csv         textos dos temas
  requirements.txt    dependências Python
  .env.example        modelo de configuração do Gemini
Front-End/
  src/routes/         páginas (Corrigir, Exemplos, Dicas, O modelo)
  src/components/nota componentes da correção (notas, redação anotada, reescrita)
  src/lib/            cliente da API, regras de verificação do texto
  src/data/           redações de exemplo
```
