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
  { valor: "0–200", rotulo: "por competência, em passos de 40" },
];

const PASSOS = [
  ["Leitura", "O tema e a redação viram tokens para o BERTimbau, um modelo de linguagem em português."],
  ["Nota", "Uma camada final prevê as 5 competências, que são arredondadas para múltiplos de 40."],
  ["Diagnóstico", "Notas até 160 geram comentários sobre o que corrigir em cada competência."],
  ["Parágrafos", "O modelo dá nota ao texto sem cada parágrafo para achar o que menos contribui."],
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

      <p className="mt-16 max-w-3xl rounded-md bg-pen-soft p-5 font-serif text-lg text-ink">
        A nota é uma estimativa. O modelo aprende padrões dos dados de treino e pode ter vieses; use o
        resultado como orientação, junto com as dicas e o checklist do corretor.
      </p>
    </div>
  );
}
