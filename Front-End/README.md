# Nota Mil: Front-End

Site do projeto de correção de redações. Ele envia o tema e a redação para a API Python (pasta `Back-End`) e mostra as notas das 5 competências do ENEM, o diagnóstico, a redação anotada, um checklist de verificações automáticas, os trechos com problema apontados por IA e a versão reescrita, com a nota de antes e depois.

A visão geral do projeto (problema, base de dados, modelo e resultados) está no [README principal](../README.md).

## Páginas

| Rota | Página | O que faz |
|---|---|---|
| `/` | Corrigir | Formulário de tema e redação; mostra notas, diagnóstico, parágrafo que menos contribui, redação anotada, checklist, trechos com problema (Gemini) e reescrita com a nota de antes e depois |
| `/exemplos` | Exemplos | Três redações reais do Essay-BR (boa, mediana e fraca), com as notas dos corretores humanos e comentários; cada uma pode ser aberta no corretor |
| `/dicas` | Dicas | Recomendações por competência |
| `/sobre` | O modelo | Como o BERTimbau foi treinado, como a nota é calculada e a tabela de vieses conhecidos do modelo |

## Como rodar

Precisa de Node.js 20 ou mais novo, e da API rodando em `http://localhost:8000` (veja o README principal).

```
npm install
npm run dev      # http://localhost:8080
npm test         # testes das regras e das rotas
npm run build    # build de produção
```

O endereço da API pode ser trocado de três formas:
- no próprio site, em "Servidor" no topo da página (fica salvo no navegador);
- na variável `VITE_API_URL`;
- se nenhuma das duas for definida, o site usa `http://localhost:8000`.

## Organização do código

| Pasta / arquivo | Conteúdo |
|---|---|
| `src/routes/` | Páginas (roteamento por arquivo do TanStack Router) |
| `src/components/nota/` | Componentes da correção: notas, diagnóstico, redação anotada, checklist, trechos com problema (`pointers-panel.tsx`), painel de reescrita |
| `src/components/ui/` | Componentes de interface (shadcn/ui) |
| `src/lib/api.ts` | Cliente da API (`/api/v1/avaliar`, `/api/v1/apontar` e `/api/v1/reescrever`) |
| `src/lib/text-checks.ts` | Verificações por regras: conectivos, repetição, oralidade, primeira pessoa, elementos da proposta de intervenção |
| `src/lib/scoring.ts` | Competências, faixas de nota e leitura do feedback |
| `src/data/exemplos.ts` | Redações de exemplo: redações reais do conjunto de teste do Essay-BR, com as notas humanas |

## Tecnologias

React 19, TypeScript, TanStack Start/Router, TanStack Query, Tailwind CSS 4, shadcn/ui e Vitest. O projeto foi criado no [Lovable](https://lovable.dev) e continua conectado a ele: commits enviados ao branch conectado aparecem no editor do Lovable.
