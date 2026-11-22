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
| Temas | 86 (enunciados em `Back-End/dados/prompts.csv`) |
| Rótulos por redação | 5 notas (C1 a C5), cada uma em {0, 40, 80, 120, 160, 200} |
| Nota final | média 614, mediana 600, de 0 a 1000 |
| Tamanho das redações | média de 289 palavras (377 tokens do BERTimbau) |

Arquivos usados:

- `Back-End/dados/essay-br.csv`: colunas `prompt` (id do tema), `title`, `essay` (lista de parágrafos), `competence` (as 5 notas) e `score` (nota final).
- `Back-End/dados/prompts.csv`: colunas `id` e `description` (texto da proposta de cada tema).
- `Back-End/dados/temas_curtos.csv` (criado pela equipe): um título curto para cada um dos 86 temas, como o usuário digita no site ("Desafios para a superação do analfabetismo funcional no Brasil"). O Essay-BR só traz o texto motivador, então os títulos foram gerados com o Gemini a partir do texto de cada proposta (`experimentos/gerar_temas_curtos.py`).

As notas se concentram em 120 e 160 em todas as competências. Notas 0 e 200 são raras: em C1, por exemplo, só 97 redações têm 0 e 122 têm 200. Esse desbalanceamento faz o modelo tender a prever notas intermediárias.

Divisão: 80% treino (3.656), 10% validação (457) e 10% teste (457), sorteados com `random_state=42`. Nessa divisão aleatória, os mesmos temas aparecem nos três conjuntos. Para medir o desempenho em temas que o modelo nunca viu, também usamos uma **divisão por tema** (`--divisao por_tema`): 68 temas no treino (3.730 redações), 9 na validação (414) e 9 no teste (426), sem temas em comum (veja "Generalização para temas inéditos").

## Modelos

### 1. Avaliador: BERTimbau ajustado (rede neural treinada pelo grupo)

- **Base:** `neuralmind/bert-base-portuguese-cased` (BERTimbau Base, 12 camadas), um BERT pré-treinado em português.
- **Entrada:** `"Tema: <tema> [SEP] Redação: <texto>"`, cortada em 512 tokens (`MAX_LEN`, o limite do BERT).
- **Tema no treino ("tema misto"):** para cada redação, sorteia-se um de dois formatos de tema, com 50% de chance cada: até 300 caracteres do texto motivador da proposta ou o título curto de `dados/temas_curtos.csv`. Assim o modelo funciona com o título que o usuário digita no site (veja "Correção do tema" nos resultados).
- **Cabeça:** dropout de 0,2 e uma camada linear de 768 para 5 saídas sobre o token `[CLS]`.
- **Alvo:** as 5 notas divididas por 200 (escala de 0 a 1). Na saída, as previsões voltam para a escala de 0 a 200 e são convertidas em 0, 40, …, 200 por cortes calibrados (veja abaixo).
- **Treino:**
  - todas as 12 camadas do BERT são ajustadas;
  - perda MSE e otimizador AdamW (lr 2e-5, weight decay 0,01), com batch 8 e 4 épocas;
  - a taxa de aprendizado sobe nos primeiros 10% dos passos e depois cai de forma linear;
  - usa precisão mista bf16 na GPU;
  - é salvo o checkpoint com o maior QWK de validação;
  - **aumento de dados contra vieses** (`avaliador/perturbacoes.py`, modelo v3): metade das redações de treino ganha uma cópia alterada com nota que não sobe, para o modelo parar de premiar características superficiais (veja "Treino contra os vieses").
- **Calibração dos cortes** (`treino/calibrate.py`):
  - o arredondamento simples (múltiplo de 40 mais próximo) concentra as notas no meio;
  - depois do treino, os 5 cortes de cada competência são ajustados no conjunto de **validação**, com título curto como tema (a condição do site), para maximizar o QWK;
  - os cortes ficam em `saved_model/config.json`, junto com o `max_len`, e a API lê os dois desse arquivo.
- **Métrica:** QWK (*Quadratic Weighted Kappa*), a métrica padrão de correção automática de redações. Ela mede a concordância com as notas humanas e pune mais os erros grandes. Vale 1 para concordância perfeita e 0 para concordância ao acaso.
- **Código:** `Back-End/treino/train.py` (treino; todos os hiperparâmetros podem ser passados por linha de comando, veja `python -m treino.train --help`) e `Back-End/treino/calibrate.py` (calibração).

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
| **E3: escolhido (base dos modelos v1, v2 e v3)** | **512** | **todas** | **4** | **sim** | **QWK** | **0,558** |

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

### Correção do tema (modelo v2)

No treino do v1, o "tema" era sempre um trecho do texto motivador da proposta (muitas vezes nem é o tema, e sim o começo do texto de apoio). No site, o usuário digita um título curto. Medimos o efeito dessa diferença no conjunto de teste (`experimentos/avaliar_tema.py`) e treinamos de novo com o **tema misto** (metade trecho, metade título curto), com os cortes calibrados usando título curto:

| Tema na entrada (teste) | v1 (só trecho) | **v2 (tema misto)** |
|---|---|---|
| **Título curto (como no site)** | 0,481 | **0,571** |
| Trecho do texto motivador | 0,570 | **0,573** |
| Sem tema | 0,360 | **0,433** |

Na condição real do site, o QWK do v1 era só 0,48; o v2 chega a 0,57 sem perder nada com o trecho. A queda grande quando o tema é retirado (0,57 → 0,43) mostra que o modelo usa bastante o tema. Como os mesmos temas aparecem no treino e no teste, parte disso provavelmente é um atalho, ou seja, uma "nota média por tema" que o modelo aprende (veja as limitações).

QWK por competência do **modelo v2**, no teste e com título curto:

| | C1 | C2 | C3 | C4 | C5 | Média |
|---|---|---|---|---|---|---|
| Arredondamento simples | 0,410 | 0,483 | 0,478 | 0,552 | 0,589 | 0,503 |
| **Cortes calibrados** | **0,533** | **0,570** | **0,543** | **0,591** | **0,618** | **0,571** |

O treino do v2 levou cerca de 11 minutos, e o melhor checkpoint foi o da época 4 (QWK de validação 0,544, com tema misto).

### Comparação com um modelo clássico (TF-IDF + regressão linear)

Para medir o ganho da rede neural, treinamos um modelo de referência clássico (`experimentos/baseline_tfidf.py`) nas mesmas condições: mesma divisão, tema misto no treino, regularização escolhida pela validação e cortes calibrados com título curto. Ele usa TF-IDF de palavras (1 e 2 palavras seguidas) e de pedaços de 2 a 5 caracteres (118.910 atributos), com regressão Ridge.

| Modelo (teste, título curto, cortes calibrados) | C1 | C2 | C3 | C4 | C5 | Média |
|---|---|---|---|---|---|---|
| TF-IDF + Ridge | 0,464 | 0,485 | 0,500 | 0,539 | 0,580 | 0,514 |
| **BERTimbau ajustado (v2)** | **0,533** | **0,570** | **0,543** | **0,591** | **0,618** | **0,571** |
| BERTimbau ajustado, contra vieses (v3, modelo do site) | 0,533 | 0,581 | 0,492 | 0,537 | 0,643 | 0,557 |

O BERTimbau v2 ganha em todas as competências (+0,057 na média). A maior diferença está na C2 (tema e repertório, +0,085) e na C1 (norma culta, +0,069). O v3, treinado contra os vieses, continua à frente na média (+0,043) e ganha mais na C2 e na C5, mas fica levemente abaixo do TF-IDF na C3 (0,492 contra 0,500), a competência mais afetada pelas penalidades do treino contra vieses. O ganho é real, mas moderado: um modelo de palavras e caracteres já captura boa parte do sinal, o que é coerente com a análise de viés (o BERTimbau também presta mais atenção ao vocabulário do que à organização das ideias). Com o trecho do texto motivador como tema, o TF-IDF chega a 0,535.

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
- **Ressalva:** é uma única divisão, com 9 temas no teste, e o resultado varia com os temas sorteados (na validação, com outros 9 temas, o QWK foi 0,49). Os números estão em `Back-End/resultados/resultados_tema_inedito.json`.

Esta avaliação usou a configuração do v2 (sem o aumento contra vieses). O modelo publicado no site é treinado com a divisão aleatória, porque assim ele vê mais temas no treino.

Nas 3 redações reais da página de exemplos (notas humanas 960, 520 e 240, escolhidas pela nota, pelo tamanho e por não terem correções embutidas, antes de ver a nota do modelo), o modelo final (v3) deu 920, 560 e 240. O v2 tinha dado 840, 520 e 40, e o v1, 840, 520 e 240. São só 3 casos; a medida confiável é o QWK de teste.

### Treino contra os vieses (modelo final, v3)

A análise de viés (próxima seção) mostrou que o v2 premia texto repetido e citações genéricas e quase não percebe frases embaralhadas. Para corrigir isso no treino, o v3 usa **aumento de dados** (`avaliador/perturbacoes.py`, `python -m treino.train --aumento-vies 0.5`). Metade das redações de treino ganha uma cópia alterada, com uma nota que não sobe, seguindo a grade do ENEM:

| Mudança na cópia de treino | Nota da cópia | Justificativa |
|---|---|---|
| Repetir um parágrafo de desenvolvimento | C3 −40 e C4 −40 | A repetição prejudica a progressão (C3) e a coesão (C4) |
| Citação famosa sem relação com o tema | igual à original | Repertório não pertinente não deve pontuar na C2 |
| Embaralhar as frases | C3 −80 e C4 −80 | O texto perde organização e encadeamento |

Foram 1.810 cópias (586, 604 e 620 de cada tipo), e o treino passou de 3.656 para 5.466 redações. A validação e o teste não foram alterados. As penalidades são escolhas da equipe com base na grade, não notas de corretores.

Para a avaliação ser honesta, as cópias de treino são **diferentes** das mudanças medidas na análise de viés: o parágrafo repetido é sorteado, as citações vêm de outra lista (a de Aristóteles, usada na análise, não aparece no treino) e o embaralhamento usa outra semente. Conectivos inseridos, acentos removidos e conclusão removida **ficam fora do treino**, para testar se o efeito se generaliza.

| Teste, título curto, cortes calibrados | C1 | C2 | C3 | C4 | C5 | Média |
|---|---|---|---|---|---|---|
| v2 | 0,533 | 0,570 | 0,543 | 0,591 | 0,618 | 0,571 |
| **v3 (contra vieses)** | **0,533** | **0,581** | **0,492** | **0,537** | **0,643** | **0,557** |

O v3 custa −0,014 de QWK médio, o que está na faixa de ruído de uma única rodada. A perda se concentra na C3 e na C4, justamente as competências que recebem as penalidades, e a C2 e a C5 melhoram. Em troca, os vieses caem muito (tabela da próxima seção). Adotamos o v3 no site porque evitar o viés é um dos objetivos do projeto; o v2 continua disponível na Release `modelo-v2`.

## Análise de viés: o que engana o modelo

Um dos objetivos é reconhecer o que realmente eleva o nível de uma redação sem cair no viés do modelo. Para isso, o script `Back-End/experimentos/analise_vies.py` aplica mudanças controladas às 914 redações de validação e teste, com título curto como tema (como no site), e mede como a nota prevista reage, antes (v2) e depois (v3) do treino contra os vieses. Cada mudança só é medida nas redações em que o texto alterado ainda cabe inteiro em 512 tokens, para não confundir com o corte.

| Mudança no texto | Tipo | No treino do v3? | v2: Δ nota | v2: nota subiu | **v3: Δ nota** | **v3: nota subiu** |
|---|---|---|---|---|---|---|
| Repetir um parágrafo inteiro | superficial | sim (outra versão) | **+32,6** | **69%** | **−55,0** | **9%** |
| Citação famosa sem relação com o tema | superficial | sim (outras citações) | **+30,8** | **66%** | **+8,6** | **31%** |
| Conectivos no início dos parágrafos | superficial | **não** | +9,4 | 26% | +6,2 | 24% |
| Tirar todos os acentos | piora | **não** | −114,7 | 0% | −122,3 | 0% |
| Embaralhar a ordem das frases | piora | sim (outra semente) | **−12,6** | 13% | **−137,9** | 2% |
| Remover a conclusão | piora | **não** | −65,9 | 2% | −113,6 | 2% |

Δ é a variação média da nota contínua (escala 0–1000), em 461 a 757 redações, dependendo da mudança. "Nota subiu" é a fração de redações cuja nota final (já convertida) aumentou.

Por competência, no v3: repetir um parágrafo dá C3 −30,0 e C4 −29,5; embaralhar as frases, C3 −59,4 e C4 −58,9; remover a conclusão, C3 −38,2, C4 −38,0 e C5 −21,0. Os números completos estão em `saved_model/analise_vies.json`.

O que isso mostra:

- **No v2:** o modelo premiava **texto mais longo** (repetir um parágrafo, sem nenhuma ideia nova, aumentava a nota de 69% das redações) e **repertório genérico** (uma citação de Aristóteles sem relação com o tema aumentava a de 66%), e quase não percebia frases embaralhadas (−13).
- **No v3:** repetir um parágrafo passou a **derrubar** a nota (−55), a citação genérica rende um quarto do que rendia (+9), e embaralhar as frases derruba a nota como deveria (−138), sobretudo na C3 e na C4.
- **Limite da correção:** os conectivos inseridos, que ficaram fora do treino, quase não mudaram (+9 → +6). A correção funciona contra os tipos de manipulação ensinados, mas não se estende sozinha a qualquer truque. Repetir e embaralhar são do mesmo tipo das cópias de treino, ainda que em versões diferentes, então parte do ganho nesses dois casos é esperada.
- **Reações corretas mantidas:** tirar os acentos e remover a conclusão, que também ficaram fora do treino, continuam derrubando a nota.

Como o site lida com o que sobra: as verificações por regras e os apontamentos do Gemini não dependem da nota da rede. As instruções do Gemini proíbem acrescentar expressões só porque "dão nota". A página "O modelo" mostra a tabela antes e depois ao usuário.

Para refazer a análise: `python -m experimentos.analise_vies` (cerca de 1 minuto numa GPU; grava `saved_model/analise_vies.json`). Para o v2: `python -m experimentos.analise_vies --model-dir <pasta do v2>`.

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
| Reconhecer características que elevam o nível, evitando viés | Pronto: vieses medidos, modelo v3 treinado contra eles e regras que não dependem da nota |

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

Os pesos (`Back-End/saved_model/`) não ficam no git por serem grandes. O modelo final (v3: tema misto e treino contra vieses), já calibrado, está publicado na Release [`modelo-v3`](https://github.com/danilodepadua/correcao_de_redacao_web/releases/tag/modelo-v3), e o `baixar_modelo.py` baixa, confere (SHA-256) e extrai esse arquivo. Os pesos da Release estão em float16 para ocupar metade do espaço: em 914 redações de validação e teste, só 4 tiveram uma nota de competência diferente da versão em float32. Os modelos anteriores continuam nas Releases `modelo-v1` e `modelo-v2`.

Para **treinar do zero** em vez de baixar (cerca de 12 min numa GPU):

```
python -m treino.train       # treina e gera saved_model/ (tema misto e aumento contra vieses, padrão; ~16 min)
python -m treino.calibrate   # ajusta os cortes das notas na validação (título curto, padrão)
```

Para treinar com outra configuração, passe os parâmetros por linha de comando (por exemplo, `python -m treino.train --max-len 384 --frozen-layers 9`) e, para experimentos, use `--out runs/<nome> --no-test`. Para medir o desempenho em temas inéditos: `python -m treino.train --divisao por_tema --out runs/tema_inedito`, depois `python -m treino.calibrate --model-dir runs/tema_inedito` e `python -m experimentos.avaliar_tema --model-dir runs/tema_inedito` (os dois leem a divisão do `config.json`), e `python -m experimentos.baseline_tfidf --divisao por_tema` para o modelo clássico.

### Reescrita com o Gemini

1. Gere uma chave em https://aistudio.google.com/apikey.
2. Copie `Back-End/.env.example` para `Back-End/.env` e preencha `GEMINI_API_KEY`. O `.env` não é versionado.
3. (Opcional) Troque `GEMINI_MODEL` no `.env`. O padrão é `gemini-3.5-flash-lite`. Para listar os modelos liberados para a sua chave: `python -c "from dotenv import load_dotenv; load_dotenv(); from google import genai; c = genai.Client(); [print(m.name) for m in c.models.list()]"`
4. Reinicie o servidor (`python main.py`).

Sem a chave, a avaliação funciona normalmente e só a reescrita responde 503, com uma mensagem explicando o que configurar.

**Falhas temporárias do Gemini:** quando o Gemini responde 429 (limite de uso) ou 5xx (sobrecarga, comum no `flash-lite` em horários de pico), a API tenta de novo com **backoff exponencial**. A espera dobra a cada tentativa (1 s, 2 s, 4 s, 8 s…, até 16 s), com uma variação aleatória para várias requisições não repetirem ao mesmo tempo, em até 5 tentativas. Erros permanentes (chave inválida, modelo inexistente) não são repetidos. Os valores podem ser ajustados no `.env` (`GEMINI_TENTATIVAS`, `GEMINI_ESPERA_INICIAL`, `GEMINI_ESPERA_MAXIMA`).

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
- **Dependência do tema:** sem tema, o QWK do modelo do site cai de 0,56 para 0,43. Ele usa o tema em parte como atalho (provavelmente uma "nota média por tema").
- **Títulos curtos gerados por IA:** os 86 títulos de `dados/temas_curtos.csv` foram gerados pelo Gemini e não foram revisados um a um por pessoas.
- **Uma rodada por experimento:** cada configuração foi treinada com uma única semente, então diferenças pequenas entre experimentos podem ser ruído.
- **Vieses restantes:** o treino contra vieses (v3) corrige os tipos de manipulação ensinados, mas não se estende sozinho a outros (conectivos inseridos ainda ajudam um pouco). As notas das cópias alteradas são heurísticas da equipe, baseadas na grade, e custaram um pouco de concordância na C3 e na C4.
- **Correções embutidas no dataset:** em parte das redações, as correções do corretor humano aparecem misturadas ao texto, logo depois do erro (ex.: "o brasil Brasil", "substancias substâncias"). Uma busca simples (palavra seguida dela mesma com outro acento ou maiúscula) encontrou 62 redações (1,4%) com 3 ou mais casos, concentradas nos temas 0 a 12, 20 e 42. O modelo pode estar aprendendo com o texto já corrigido.
- **Filtro de entrada simples:** barra repetição, letras soltas e palavras desconhecidas, mas um texto em português correto e fora do tema passa.
- **Análise por parágrafo:** retirar um parágrafo mede o quanto ele pesa para o modelo, não se ele está bom; com os vieses acima, parágrafos longos tendem a "contribuir" mais.
- **Mensagens por competência genéricas:** as mensagens do diagnóstico da rede continuam fixas por faixa de nota. Os trechos específicos vêm das regras e dos apontamentos do Gemini.
- **Repertório da reescrita:** o Gemini pode errar fatos. Em dois testes, a versão reescrita afirmou que livros têm "alta carga tributária", o que é falso, porque a Constituição dá imunidade tributária aos livros e ao papel usado para imprimi-los. O erro se repetiu mesmo depois de a instrução pedir para não usar fatos incertos. Por isso o site mostra um aviso junto da reescrita e dos apontamentos.

## Próximos passos

- Repetir a avaliação com temas inéditos em várias divisões (validação cruzada por tema) e com várias sementes, para ter uma margem de erro.
- Lidar com redações maiores que 512 tokens (por exemplo, dividir em dois trechos e combinar).
- Ampliar o treino contra vieses a outras manipulações (conectivos mecânicos, vocabulário rebuscado sem conteúdo) e calibrar as penalidades das cópias, para recuperar a concordância na C3 e na C4.
- Limpar as correções embutidas no dataset e medir o efeito no modelo.
- Verificar se a redação fala do tema (por exemplo, comparando com o texto da proposta) e incluir isso no filtro.
- Testar um modelo do Gemini maior (`gemini-3.5-flash`) para ver se os erros de repertório diminuem.

## Estrutura do repositório

```
Back-End/
  main.py               API (avaliação, apontamentos e reescrita)   → python main.py
  baixar_modelo.py      baixa o modelo treinado da Release           → python baixar_modelo.py
  requirements.txt      dependências Python
  .env.example          modelo de configuração do Gemini
  avaliador/            código compartilhado (pacote Python)
    caminhos.py         onde ficam dados, modelo e resultados
    modelo.py           arquitetura (BERTimbau + regressão), dataset e predição
    dados.py            leitura do Essay-BR e divisões (aleatória ou por tema)
    notas.py            conversão em notas 0–200, calibração dos cortes e QWK
    perturbacoes.py     cópias alteradas para o treino contra vieses
  treino/
    train.py            treino do avaliador                          → python -m treino.train
    calibrate.py        calibração dos cortes na validação           → python -m treino.calibrate
  experimentos/
    baseline_tfidf.py   modelo clássico de referência (TF-IDF + Ridge) → python -m experimentos.baseline_tfidf
    avaliar_tema.py     QWK com cada tipo de tema (trecho, curto, vazio) → python -m experimentos.avaliar_tema
    analise_vies.py     experimento de viés (mudanças controladas)   → python -m experimentos.analise_vies
    gerar_temas_curtos.py  gera dados/temas_curtos.csv com o Gemini  → python -m experimentos.gerar_temas_curtos
  dados/
    essay-br.csv        redações e notas (Essay-BR)
    prompts.csv         textos das propostas
    temas_curtos.csv    título curto de cada um dos 86 temas
  resultados/
    baseline_tfidf.json            resultados do modelo clássico
    resultados_tema_inedito.json   resultados com a divisão por tema
  saved_model/          modelo treinado (fora do git; baixado ou gerado pelo treino)
Front-End/
  src/routes/           páginas (Corrigir, Exemplos, Dicas, O modelo)
  src/components/nota/  componentes da correção (notas, redação anotada, trechos, reescrita)
  src/lib/              cliente da API, regras de verificação do texto
  src/data/             redações de exemplo
```

Os comandos são rodados de dentro de `Back-End`. Os scripts de `treino/` e `experimentos/` também funcionam se chamados direto pelo caminho (por exemplo, `python treino/train.py`), e os caminhos de dados e do modelo não dependem da pasta de onde são rodados.
