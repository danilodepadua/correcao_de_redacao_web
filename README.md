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
- `Back-End/temas_curtos.csv` (criado pela equipe): um título curto para cada um dos 86 temas, como o usuário digita no site ("Desafios para a superação do analfabetismo funcional no Brasil"). O Essay-BR só traz o texto motivador, então os títulos foram gerados com o Gemini a partir do texto de cada proposta (`gerar_temas_curtos.py`).

As notas se concentram em 120 e 160 em todas as competências. Notas 0 e 200 são raras: em C1, por exemplo, só 97 redações têm 0 e 122 têm 200. Esse desbalanceamento faz o modelo tender a prever notas intermediárias.

Divisão: 80% treino (3.656), 10% validação (457) e 10% teste (457), sorteados com `random_state=42`. Nessa divisão aleatória, os mesmos temas aparecem nos três conjuntos. Para medir o desempenho em temas que o modelo nunca viu, também usamos uma **divisão por tema** (`--divisao por_tema`): 68 temas no treino (3.730 redações), 9 na validação (414) e 9 no teste (426), sem temas em comum (veja "Generalização para temas inéditos").

## Modelos

### 1. Avaliador: BERTimbau ajustado (rede neural treinada pelo grupo)

- **Base:** `neuralmind/bert-base-portuguese-cased` (BERTimbau Base, 12 camadas), um BERT pré-treinado em português.
- **Entrada:** `"Tema: <tema> [SEP] Redação: <texto>"`, cortada em 512 tokens (`MAX_LEN`, o limite do BERT).
- **Tema no treino ("tema misto"):** para cada redação, sorteia-se um de dois formatos de tema, com 50% de chance cada: até 300 caracteres do texto motivador da proposta ou o título curto de `temas_curtos.csv`. Assim o modelo funciona com o título que o usuário digita no site (veja "Correção do tema" nos resultados).
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
  - depois do treino, os 5 cortes de cada competência são ajustados no conjunto de **validação**, com título curto como tema (a condição do site), para maximizar o QWK;
  - os cortes ficam em `saved_model/config.json`, junto com o `max_len`, e a API lê os dois desse arquivo.
- **Métrica:** QWK (*Quadratic Weighted Kappa*), a métrica padrão de correção automática de redações. Ela mede a concordância com as notas humanas e pune mais os erros grandes. Vale 1 para concordância perfeita e 0 para concordância ao acaso.
- **Código:** `Back-End/train.py` (treino; todos os hiperparâmetros podem ser passados por linha de comando, veja `python train.py --help`) e `Back-End/calibrate.py` (calibração).

### 2. Apontamentos e reescrita: Gemini (modelo pronto, via API)

O Gemini (`gemini-3.5-flash-lite` por padrão) é usado em duas funções, sempre junto com o BERTimbau:

- **Apontar trechos com problema** (`/api/v1/apontar`): devolve, em JSON estruturado, trechos copiados do texto com a competência afetada, o problema e uma sugestão. A API descarta qualquer trecho que não exista de fato na redação, porque o modelo pode inventar citações.
- **Reescrever a redação** (`/api/v1/reescrever`): as notas e o diagnóstico do BERTimbau entram no prompt, para que a reescrita corrija os pontos apontados. Depois, a versão reescrita é **reavaliada pelo BERTimbau**, e o site mostra a nota "antes → depois" por competência.

As instruções pedem ao Gemini que:

- mantenha as ideias do autor;
- não invente dados nem citações, e não use fatos de que não tenha certeza;
- faça uma proposta de intervenção completa;
- não acrescente expressões só porque costumam "dar nota".

Mesmo assim, o Gemini às vezes erra fatos (veja as limitações), e o site avisa que o repertório gerado deve ser conferido.

### 3. Verificações por regras (sem aprendizado)

O site também confere o texto com regras fixas:

- número de parágrafos e tamanho do texto;
- conectivos no início dos parágrafos;
- palavras repetidas;
- marcas de oralidade e primeira pessoa;
- os 5 elementos da proposta de intervenção.

Essas regras apontam trechos exatos do texto, o que a rede neural sozinha não faz.

### 4. Filtro de entrada e análise por parágrafo

- **Filtro de textos sem sentido:** antes da rede, a API recusa textos com poucas palavras diferentes, uma palavra dominante, palavras que o vocabulário do BERTimbau não reconhece ou poucas letras. Os limites foram escolhidos com folga em relação ao pior caso das 4.570 redações do Essay-BR, e nenhuma delas é barrada. Antes, "palavra" repetida 40 vezes recebia nota.
- **Análise por parágrafo:** o modelo avalia o texto sem cada parágrafo, num lote só, e mede a variação da nota contínua (antes da conversão em múltiplos de 40). O parágrafo que menos contribui é apontado, junto com a competência em que ele mais pesa e com o aviso de quando retirá-lo faria a nota subir. Antes, a comparação usava a nota convertida, que quase nunca mudava, e todos os parágrafos empatavam em 0.

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

### Modelo v1 (E3 + calibração)

Esta seção descreve o primeiro modelo final (Release `modelo-v1`), treinado só com o trecho do texto motivador como tema. Treino em uma RTX 4050 Laptop (6 GB), em 10 min 51 s. O melhor checkpoint foi o da época 3.

| Época | Perda de treino | Perda de validação | QWK médio (validação) |
|---|---|---|---|
| 1 | 0,0697 | 0,0291 | 0,514 |
| 2 | 0,0323 | 0,0281 | 0,520 |
| 3 | 0,0266 | **0,0262** | **0,558** |
| 4 | 0,0235 | 0,0285 | 0,544 |

**QWK no conjunto de teste** (457 redações que o modelo nunca viu, com o trecho do texto motivador como tema):

| Modelo | C1 | C2 | C3 | C4 | C5 | Média |
|---|---|---|---|---|---|---|
| Original (384 tokens, 3 camadas, arredondamento simples) | 0,342 | 0,431 | 0,395 | 0,496 | 0,531 | 0,439 |
| v1, arredondamento simples | 0,469 | 0,480 | 0,499 | 0,547 | 0,603 | 0,520 |
| **v1, cortes calibrados** | **0,522** | **0,557** | **0,538** | **0,595** | **0,636** | **0,570** |

MSE de teste do v1: 0,0276 (na escala de 0 a 1).

Como ler: o QWK subiu de 0,44 para 0,57, uma concordância **moderada** com os corretores humanos. A maior melhora foi na C1 (norma culta, de 0,34 para 0,52), que dependia mais do texto inteiro. Na validação, os cortes calibrados chegam a 0,643, mas esse número é otimista porque os cortes foram ajustados nela. O valor honesto é o do teste.

As notas também ficaram mais espalhadas: um texto sem sentido ("palavra" repetida 40 vezes) caiu de 320 para 80 pontos (e hoje é recusado pelo filtro de entrada).

### Correção do tema (modelo final, v2)

No treino do v1, o "tema" era sempre um trecho do texto motivador da proposta (muitas vezes nem é o tema, e sim o começo do texto de apoio). No site, o usuário digita um título curto. Medimos o efeito dessa diferença no conjunto de teste (`avaliar_tema.py`) e treinamos de novo com o **tema misto** (metade trecho, metade título curto), com os cortes calibrados usando título curto:

| Tema na entrada (teste) | v1 (só trecho) | **v2 (tema misto)** |
|---|---|---|
| **Título curto (como no site)** | 0,481 | **0,571** |
| Trecho do texto motivador | 0,570 | **0,573** |
| Sem tema | 0,360 | **0,433** |

Na condição real do site, o QWK do v1 era só 0,48; o v2 chega a 0,57 sem perder nada com o trecho. A queda grande quando o tema é retirado (0,57 → 0,43) mostra que o modelo usa bastante o tema. Como os mesmos temas aparecem no treino e no teste, parte disso provavelmente é um atalho, ou seja, uma "nota média por tema" que o modelo aprende (veja as limitações).

QWK por competência do **modelo final (v2)**, no teste e com título curto:

| | C1 | C2 | C3 | C4 | C5 | Média |
|---|---|---|---|---|---|---|
| Arredondamento simples | 0,410 | 0,483 | 0,478 | 0,552 | 0,589 | 0,503 |
| **Cortes calibrados** | **0,533** | **0,570** | **0,543** | **0,591** | **0,618** | **0,571** |

O treino do v2 levou cerca de 11 minutos, e o melhor checkpoint foi o da época 4 (QWK de validação 0,544, com tema misto).

### Comparação com um modelo clássico (TF-IDF + regressão linear)

Para medir o ganho da rede neural, treinamos um modelo de referência clássico (`baseline_tfidf.py`) nas mesmas condições: mesma divisão, tema misto no treino, regularização escolhida pela validação e cortes calibrados com título curto. Ele usa TF-IDF de palavras (1 e 2 palavras seguidas) e de pedaços de 2 a 5 caracteres (118.910 atributos), com regressão Ridge.

| Modelo (teste, título curto, cortes calibrados) | C1 | C2 | C3 | C4 | C5 | Média |
|---|---|---|---|---|---|---|
| TF-IDF + Ridge | 0,464 | 0,485 | 0,500 | 0,539 | 0,580 | 0,514 |
| **BERTimbau ajustado (v2)** | **0,533** | **0,570** | **0,543** | **0,591** | **0,618** | **0,571** |

O BERTimbau ganha em todas as competências (+0,057 na média). A maior diferença está na C2 (tema e repertório, +0,085) e na C1 (norma culta, +0,069). O ganho é real, mas moderado: um modelo de palavras e caracteres já captura boa parte do sinal, o que é coerente com a análise de viés (o BERTimbau também presta mais atenção ao vocabulário do que à organização das ideias). Com o trecho do texto motivador como tema, o TF-IDF chega a 0,535.

### Generalização para temas inéditos

Na divisão aleatória, os temas do teste também estão no treino, e vimos que o modelo usa o tema como atalho. Para medir o desempenho com um tema novo (o caso de um estudante que treina com uma proposta que não está no Essay-BR), treinamos e avaliamos os dois modelos com a **divisão por tema**, em que os 9 temas do teste não aparecem no treino. As condições são as mesmas de antes: tema misto no treino, cortes calibrados na validação (também com temas inéditos) e teste com título curto.

| Modelo (teste, título curto, cortes calibrados) | Temas já vistos (divisão aleatória) | **Temas inéditos** | Queda |
|---|---|---|---|
| TF-IDF + Ridge | 0,514 | **0,338** | −0,176 |
| **BERTimbau ajustado** | 0,571 | **0,520** | −0,051 |

QWK do BERTimbau com temas inéditos, por competência: C1 0,433 · C2 0,598 · C3 0,507 · C4 0,573 · C5 0,488.

- **O BERTimbau generaliza muito melhor.** O TF-IDF perde um terço do desempenho quando o tema é novo, porque depende muito das palavras típicas de cada tema. O BERTimbau perde pouco, e a vantagem dele sobre o modelo clássico sobe de +0,06 para **+0,18**. É o principal argumento a favor da rede neural neste problema.
- **Expectativa realista no uso:** com um tema fora do Essay-BR, o QWK esperado é de cerca de 0,52, e não 0,57.
- **Menos dependência do tema:** sem tema, esse modelo cai só de 0,52 para 0,48, contra 0,57 → 0,43 no modelo treinado com a divisão aleatória. Quando o tema não pode ser decorado, o modelo aprende mais com o texto.
- **Ressalva:** é uma única divisão, com 9 temas no teste, e o resultado varia com os temas sorteados (na validação, com outros 9 temas, o QWK foi 0,49). Os números estão em `Back-End/resultados_tema_inedito.json`.

O modelo publicado no site continua sendo o v2, treinado com a divisão aleatória, porque ele viu mais temas no treino.

Nas 3 redações reais da página de exemplos (notas humanas 960, 520 e 240, escolhidas pela nota, pelo tamanho e por não terem correções embutidas, antes de ver a nota do modelo), o v2 deu 840, 520 e 40. O v1 tinha dado 840, 520 e 240. São só 3 casos; a medida confiável é o QWK de teste.

## Análise de viés: o que engana o modelo

Um dos objetivos é reconhecer o que realmente eleva o nível de uma redação sem cair no viés do modelo. Para isso, o script `Back-End/analise_vies.py` aplica mudanças controladas às 914 redações de validação e teste, com título curto como tema (como no site), e mede como a nota prevista pelo modelo final (v2) reage. Cada mudança só é medida nas redações em que o texto alterado ainda cabe inteiro em 512 tokens, para não confundir com o corte.

| Mudança no texto | Tipo | n | Δ nota (0–1000) | C1 | C2 | C3 | C4 | C5 | Nota final subiu |
|---|---|---|---|---|---|---|---|---|---|
| Conectivos no início dos parágrafos | superficial | 725 | +9,4 | +0,7 | +1,6 | +1,1 | +3,1 | +2,8 | 26% |
| Repetir um parágrafo inteiro | superficial | 461 | **+32,6** | +3,2 | +9,4 | +6,4 | +8,0 | +5,7 | **69%** |
| Citação famosa sem relação com o tema | superficial | 689 | **+30,8** | +4,7 | +7,3 | +5,4 | +6,9 | +6,4 | **66%** |
| Tirar todos os acentos | piora | 634 | −114,7 | −26,1 | −30,0 | −20,6 | −20,3 | −17,7 | 0% |
| Embaralhar a ordem das frases | piora | 757 | **−12,6** | −4,1 | +1,0 | −0,6 | −4,3 | −4,5 | 13% |
| Remover a conclusão | piora | 749 | −65,9 | −5,0 | −11,4 | −12,8 | −15,6 | −21,1 | 2% |

Δ é a variação média da nota contínua. A última coluna é a fração de redações cuja nota final (já convertida) aumentou.

O que isso mostra:

- **Vieses:** o modelo premia **texto mais longo** (repetir um parágrafo, sem nenhuma ideia nova, aumenta a nota de 69% das redações) e **repertório genérico** (uma citação de Aristóteles sem relação com o tema aumenta a de 66%). Conectivos inseridos mecanicamente também ajudam um pouco.
- **Ponto cego:** embaralhar as frases quase não muda a nota (−13). O modelo enxerga o vocabulário, mas pouco a coerência e a ordem das ideias.
- **Reações corretas:** tirar os acentos derruba a nota (−115), e remover a conclusão também (−66), com a maior queda na C5, como esperado.

Como o site lida com isso: as verificações por regras e os apontamentos do Gemini não dependem da nota da rede. As instruções do Gemini proíbem acrescentar expressões só porque "dão nota". A página "O modelo" mostra essa tabela ao usuário. Uma forma de reduzir esses vieses no treino seria usar exemplos alterados desse jeito, com nota menor, como dados extras (veja os próximos passos).

Para refazer a análise: `python analise_vies.py` (cerca de 1 minuto numa GPU; grava `saved_model/analise_vies.json`).

## Funcionalidades do site

| Funcionalidade | Situação |
|---|---|
| Avaliar uma redação: 5 notas, nota final e diagnóstico por competência | Pronto |
| Recusar textos sem sentido antes da rede | Pronto |
| Análise de impacto por parágrafo (variação contínua da nota) | Pronto |
| Redação anotada e checklist do corretor (regras) | Pronto |
| Trechos com problema apontados pelo Gemini, com competência e sugestão | Pronto |
| Exemplos de redações boa, mediana e fraca | Pronto (redações reais do Essay-BR, com as notas humanas) |
| Página de dicas e página sobre o modelo (com a tabela de vieses) | Pronto |
| Reescrever a redação com o Gemini e reavaliar a nova versão | Pronto |
| Reconhecer características que elevam o nível, evitando viés | Feito como análise: experimento de viés (abaixo) e regras que não dependem da nota |

Páginas: **Corrigir** (`/`), **Exemplos** (`/exemplos`), **Dicas** (`/dicas`) e **O modelo** (`/sobre`). O endereço da API pode ser trocado no próprio site, em "Servidor".

## Arquitetura

```
Navegador (Front-End: React + TanStack Start, porta 8080)
   │  POST /api/v1/avaliar     { tema, redacao }
   │  POST /api/v1/apontar     { tema, redacao }
   │  POST /api/v1/reescrever  { tema, redacao }
   ▼
API (Back-End: FastAPI, porta 8000)
   ├── filtro de textos sem sentido
   ├── BERTimbau ajustado (saved_model/) → 5 notas + diagnóstico + análise por parágrafo
   └── Gemini (API do Google)            → trechos com problema, redação reescrita
```

### Endpoints da API

| Método | Rota | Entrada | Saída |
|---|---|---|---|
| GET | `/` | | `{status, device}` |
| POST | `/api/v1/avaliar` | `{tema, redacao}` | `c1_norma_culta` … `c5_proposta_intervencao`, `nota_final`, `feedback` |
| POST | `/api/v1/apontar` | `{tema, redacao}` | `{apontamentos: [{competencia, trecho, problema, sugestao}], descartados}` |
| POST | `/api/v1/reescrever` | `{tema, redacao}` | `{redacao_reescrita, notas_original, notas_reescrita}` |

A redação precisa ter pelo menos 30 palavras e passar pelo filtro de textos sem sentido (senão a API responde 400 com o motivo). Com a API rodando, a documentação interativa fica em http://localhost:8000/docs.

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

Os pesos (`Back-End/saved_model/`) não ficam no git por serem grandes. O modelo final (v2, tema misto), já calibrado, está publicado na Release [`modelo-v2`](https://github.com/danilodepadua/correcao_de_redacao_web/releases/tag/modelo-v2), e o `baixar_modelo.py` baixa, confere (SHA-256) e extrai esse arquivo. Os pesos da Release estão em float16 para ocupar metade do espaço: em 914 redações de validação e teste, só 5 tiveram uma nota de competência diferente da versão em float32. O modelo anterior continua na Release `modelo-v1`.

Para **treinar do zero** em vez de baixar (cerca de 12 min numa GPU):

```
python train.py       # treina e gera saved_model/ (tema misto, padrão)
python calibrate.py   # ajusta os cortes das notas na validação (título curto, padrão)
```

Para treinar com outra configuração, passe os parâmetros por linha de comando (por exemplo, `python train.py --max-len 384 --frozen-layers 9`) e, para experimentos, use `--out runs/<nome> --no-test`. Para medir o desempenho em temas inéditos: `python train.py --divisao por_tema --out runs/tema_inedito`, depois `python calibrate.py --model-dir runs/tema_inedito` e `python avaliar_tema.py --model-dir runs/tema_inedito` (os dois leem a divisão do `config.json`), e `python baseline_tfidf.py --divisao por_tema` para o modelo clássico.

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
- **Temas inéditos rendem menos:** os números principais usam a divisão aleatória, em que os temas do teste também estão no treino. Com temas inéditos, o QWK do BERTimbau cai de 0,571 para 0,520 (medido numa única divisão, com 9 temas no teste).
- **Dependência do tema:** sem tema, o QWK do modelo do site cai de 0,57 para 0,43. Ele usa o tema em parte como atalho (provavelmente uma "nota média por tema").
- **Títulos curtos gerados por IA:** os 86 títulos de `temas_curtos.csv` foram gerados pelo Gemini e não foram revisados um a um por pessoas.
- **Uma rodada por experimento:** cada configuração foi treinada com uma única semente, então diferenças pequenas entre experimentos podem ser ruído.
- **Vieses do modelo:** premia texto longo e repertório genérico e quase não percebe a coerência (veja a análise de viés).
- **Correções embutidas no dataset:** em parte das redações, as correções do corretor humano aparecem misturadas ao texto, logo depois do erro (ex.: "o brasil Brasil", "substancias substâncias"). Uma busca simples (palavra seguida dela mesma com outro acento ou maiúscula) encontrou 62 redações (1,4%) com 3 ou mais casos, concentradas nos temas 0 a 12, 20 e 42. O modelo pode estar aprendendo com o texto já corrigido.
- **Filtro de entrada simples:** barra repetição, letras soltas e palavras desconhecidas, mas um texto em português correto e fora do tema passa.
- **Análise por parágrafo:** retirar um parágrafo mede o quanto ele pesa para o modelo, não se ele está bom; com os vieses acima, parágrafos longos tendem a "contribuir" mais.
- **Mensagens por competência genéricas:** as mensagens do diagnóstico da rede continuam fixas por faixa de nota. Os trechos específicos vêm das regras e dos apontamentos do Gemini.
- **Repertório da reescrita:** o Gemini pode errar fatos. Em dois testes, a versão reescrita afirmou que livros têm "alta carga tributária", o que é falso, porque a Constituição dá imunidade tributária aos livros e ao papel usado para imprimi-los. O erro se repetiu mesmo depois de a instrução pedir para não usar fatos incertos. Por isso o site mostra um aviso junto da reescrita e dos apontamentos.

## Próximos passos

- Repetir a avaliação com temas inéditos em várias divisões (validação cruzada por tema) e com várias sementes, para ter uma margem de erro.
- Lidar com redações maiores que 512 tokens (por exemplo, dividir em dois trechos e combinar).
- Reduzir os vieses no treino: acrescentar versões alteradas das redações (parágrafo repetido, frases embaralhadas, citação sem relação) com nota menor.
- Limpar as correções embutidas no dataset e medir o efeito no modelo.
- Verificar se a redação fala do tema (por exemplo, comparando com o texto da proposta) e incluir isso no filtro.
- Testar um modelo do Gemini maior (`gemini-3.5-flash`) para ver se os erros de repertório diminuem.

## Estrutura do repositório

```
Back-End/
  train.py            treino e avaliação do BERTimbau
  calibrate.py        calibração dos cortes das notas
  baseline_tfidf.py   modelo de referência clássico (TF-IDF + Ridge)
  avaliar_tema.py     QWK do modelo com cada tipo de tema (trecho, curto, vazio)
  gerar_temas_curtos.py  gera temas_curtos.csv com o Gemini
  temas_curtos.csv    título curto de cada um dos 86 temas
  resultados_tema_inedito.json  resultados com a divisão por tema
  baixar_modelo.py    baixa o modelo treinado da Release do GitHub
  analise_vies.py     experimento de viés (mudanças controladas no texto)
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
