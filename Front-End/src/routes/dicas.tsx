import { createFileRoute, Link } from "@tanstack/react-router";
import { COMPETENCIAS } from "@/lib/scoring";

export const Route = createFileRoute("/dicas")({
  head: () => ({
    meta: [
      { title: "Dicas para subir a nota da redação — Nota Mil" },
      { name: "description", content: "Características que elevam a nota em cada competência da redação do ENEM." },
      { property: "og:title", content: "Dicas para subir a nota da redação — Nota Mil" },
      { property: "og:description", content: "O que fazer em cada uma das 5 competências do ENEM." },
      { property: "og:type", content: "article" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Dicas,
});

const DICAS: Record<number, string[]> = {
  1: [
    "Revise concordância verbal e nominal, especialmente com sujeitos longos.",
    "Troque marcas de oralidade (“a gente”, “pra”, “né”) por equivalentes formais.",
    "Use vírgula antes de conjunções adversativas e depois de expressões deslocadas.",
  ],
  2: [
    "Cite o tema com as próprias palavras já na introdução.",
    "Use repertório legitimado: leis, dados, obras, autores ou fatos históricos.",
    "Ligue o repertório ao argumento; citação solta não pontua.",
  ],
  3: [
    "Apresente a tese no fim da introdução e anuncie os dois argumentos.",
    "Desenvolva causa e consequência em cada argumento.",
    "Cada parágrafo de desenvolvimento defende uma única ideia.",
  ],
  4: [
    "Comece todo parágrafo com um conectivo: “Além disso”, “Nesse sentido”, “Portanto”.",
    "Evite repetir a mesma palavra; use sinônimos e pronomes.",
    "Varie os conectivos dentro dos parágrafos também.",
  ],
  5: [
    "Inclua os cinco elementos: agente, ação, meio, finalidade e detalhamento.",
    "Seja específico: “o Ministério da Educação” em vez de “o governo”.",
    "Respeite os direitos humanos e ligue a proposta aos problemas discutidos.",
  ],
};

function Dicas() {
  return (
    <div className="mx-auto max-w-6xl px-4 sm:px-6">
      <header className="max-w-3xl pb-6 pt-14 md:pt-20">
        <p className="eyebrow">Recomendações</p>
        <h1 className="mt-4 text-5xl font-semibold leading-tight">Como subir de nível, competência por competência.</h1>
        <p className="mt-4 font-serif text-lg text-muted-foreground">
          Características que elevam uma redação segundo a grade oficial do ENEM, e não só o que o
          modelo favorece. Teste cada uma no{" "}
          <Link to="/" className="text-ink underline decoration-pen underline-offset-4">
            corretor
          </Link>
          .
        </p>
      </header>

      <ol className="mt-10 divide-y divide-border border-y border-border">
        {COMPETENCIAS.map((c) => (
          <li key={c.key} className="grid gap-6 py-10 md:grid-cols-[14rem_1fr]">
            <div>
              <span className="font-display text-6xl font-semibold text-pen">{c.numero}</span>
              <h2 className="mt-2 text-2xl font-semibold">{c.nome}</h2>
              <p className="mt-1 text-sm text-muted-foreground">{c.resumo}</p>
            </div>
            <ul className="space-y-4 self-center">
              {(DICAS[c.numero] ?? []).map((d) => (
                <li key={d} className="flex gap-3 font-serif text-lg leading-relaxed">
                  <span className="mt-3 h-px w-6 shrink-0 bg-pen" />
                  {d}
                </li>
              ))}
            </ul>
          </li>
        ))}
      </ol>
    </div>
  );
}
