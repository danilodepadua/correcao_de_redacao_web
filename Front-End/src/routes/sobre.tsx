import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/sobre")({
  head: () => ({
    meta: [
      { title: "Como funciona o modelo — Nota Mil" },
      { name: "description", content: "Rede neural BERTimbau treinada no Essay-BR para dar nota às 5 competências do ENEM." },
      { property: "og:title", content: "Como funciona o modelo — Nota Mil" },
      { property: "og:description", content: "BERTimbau ajustado com redações reais corrigidas." },
      { property: "og:type", content: "article" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Sobre,
});

const FATOS = [
  { valor: "4.570", rotulo: "redações corrigidas no treino (Essay-BR)" },
  { valor: "86", rotulo: "temas diferentes" },
  { valor: "5", rotulo: "notas, uma por competência" },
  { valor: "0,57", rotulo: "QWK: concordância com os corretores em redações inéditas" },
];

const PASSOS = [
  ["Filtro", "Textos sem sentido (palavras repetidas, letras soltas, números) são recusados antes da rede."],
  ["Leitura", "O tema e a redação (até 512 tokens) viram entrada para o BERTimbau, um modelo de linguagem em português."],
  ["Nota", "Uma camada final prevê as 5 competências; cortes ajustados nos dados convertem a previsão em 0, 40, …, 200."],
  ["Diagnóstico", "Notas até 160 geram comentários sobre o que corrigir em cada competência."],
  ["Parágrafos", "O modelo dá nota ao texto sem cada parágrafo para achar o que menos contribui."],
  ["IA generativa", "O Gemini aponta trechos com problema e reescreve a redação; a versão nova volta para o modelo."],
];

/** Resultado do experimento de viés (Back-End/analise_vies.py), em 914 redações inéditas. */
const VIESES = [
  { mudanca: "Repetir um parágrafo inteiro", efeito: "+36", subiu: "69%", reacao: "Viés: premia texto mais longo", ok: false },
  { mudanca: "Colar uma citação famosa sem relação com o tema", efeito: "+31", subiu: "66%", reacao: "Viés: premia repertório genérico", ok: false },
  { mudanca: "Pôr conectivos no início dos parágrafos", efeito: "+11", subiu: "29%", reacao: "Viés leve", ok: false },
  { mudanca: "Embaralhar a ordem das frases", efeito: "−9", subiu: "13%", reacao: "Cai pouco: quase não vê a coerência", ok: false },
  { mudanca: "Remover a conclusão", efeito: "−60", subiu: "1%", reacao: "Esperada (cai mais na C5)", ok: true },
  { mudanca: "Tirar todos os acentos", efeito: "−85", subiu: "1%", reacao: "Esperada", ok: true },
];

function Sobre() {
  return (
    <div className="mx-auto max-w-6xl px-4 sm:px-6">
      <header className="max-w-3xl pb-6 pt-14 md:pt-20">
        <p className="eyebrow">O modelo</p>
        <h1 className="mt-4 text-5xl font-semibold leading-tight">Uma rede neural que leu milhares de redações.</h1>
        <p className="mt-4 font-serif text-lg text-muted-foreground">
          O Nota Mil usa o BERTimbau (neuralmind/bert-base-portuguese-cased), ajustado com redações
          reais corrigidas por especialistas, para estimar a nota em cada competência do ENEM.
        </p>
      </header>

      <dl className="mt-10 grid grid-cols-2 gap-px overflow-hidden rounded-md border border-border bg-border md:grid-cols-4">
        {FATOS.map((f) => (
          <div key={f.rotulo} className="bg-card p-6">
            <dt className="text-sm text-muted-foreground">{f.rotulo}</dt>
            <dd className="mt-2 font-display text-4xl font-semibold text-ink">{f.valor}</dd>
          </div>
        ))}
      </dl>

      <ol className="mt-16 grid gap-8 md:grid-cols-2">
        {PASSOS.map(([titulo, texto], i) => (
          <li key={titulo} className="flex gap-4">
            <span className="font-display text-3xl italic text-pen">{i + 1}.</span>
            <div>
              <h2 className="text-2xl font-semibold">{titulo}</h2>
              <p className="mt-1 font-serif text-lg text-muted-foreground">{texto}</p>
            </div>
          </li>
        ))}
      </ol>

      <section className="mt-20 max-w-4xl">
        <p className="eyebrow">Vieses conhecidos</p>
        <h2 className="mt-2 text-3xl font-semibold">O que engana o modelo</h2>
        <p className="mt-2 font-serif text-lg text-muted-foreground">
          Aplicamos mudanças controladas a 914 redações que o modelo não viu no treino e medimos a
          variação média da nota (de 0 a 1000). O modelo reage bem a erros reais, como tirar os
          acentos ou a conclusão, mas premia texto mais longo e repertório genérico e quase não
          percebe quando a ordem das ideias é embaralhada. Por isso o checklist do corretor e os
          apontamentos não dependem só da nota.
        </p>
        <div className="mt-6 overflow-x-auto">
          <table className="w-full min-w-[28rem] text-left text-sm">
            <thead className="text-muted-foreground">
              <tr className="border-b border-border">
                <th className="py-2 pr-4 font-medium">Mudança no texto</th>
                <th className="py-2 pr-4 font-medium">Efeito na nota</th>
                <th className="py-2 pr-4 font-medium">Redações com nota maior</th>
                <th className="py-2 font-medium">Reação</th>
              </tr>
            </thead>
            <tbody>
              {VIESES.map((v) => (
                <tr key={v.mudanca} className="border-b border-border/60">
                  <td className="py-2 pr-4">{v.mudanca}</td>
                  <td className="py-2 pr-4 font-semibold">{v.efeito}</td>
                  <td className="py-2 pr-4">{v.subiu}</td>
                  <td className={v.ok ? "py-2 text-score-great" : "py-2 text-pen"}>
                    {v.reacao}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <p className="mt-16 max-w-3xl rounded-md bg-pen-soft p-5 font-serif text-lg text-ink">
        A nota é uma estimativa. O modelo aprende padrões dos dados de treino e pode ter vieses; use o
        resultado como orientação, junto com as dicas e o checklist do corretor.
      </p>
    </div>
  );
}
