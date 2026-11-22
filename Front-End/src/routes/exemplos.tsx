import { createFileRoute, Link } from "@tanstack/react-router";
import { ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { CompetenceList } from "@/components/nota/competence-list";
import { ScoreSeal } from "@/components/nota/score-seal";
import { EXEMPLOS, type Exemplo } from "@/data/exemplos";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/exemplos")({
  head: () => ({
    meta: [
      { title: "Exemplos de redações boas e ruins — Nota Mil" },
      { name: "description", content: "Redações reais do Essay-BR com nota alta, média e baixa, com as notas dos corretores e o que muda a nota." },
      { property: "og:title", content: "Exemplos de redações boas e ruins — Nota Mil" },
      { property: "og:description", content: "Redações comentadas, competência por competência." },
      { property: "og:type", content: "article" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Exemplos,
});

const ROTULO: Record<Exemplo["qualidade"], { texto: string; cor: string }> = {
  boa: { texto: "Redação boa", cor: "text-score-great" },
  mediana: { texto: "Redação mediana", cor: "text-score-mid" },
  fraca: { texto: "Redação fraca", cor: "text-score-low" },
};

function Exemplos() {
  return (
    <div className="mx-auto max-w-6xl px-4 sm:px-6">
      <header className="max-w-3xl pb-6 pt-14 md:pt-20">
        <p className="eyebrow">Exemplos comentados</p>
        <h1 className="mt-4 text-5xl font-semibold leading-tight">O que separa um 960 de um 240.</h1>
        <p className="mt-4 font-serif text-lg text-muted-foreground">
          Três redações reais do Essay-BR, a base usada para treinar o modelo, com as notas dadas
          por corretores humanos. O texto está como o estudante escreveu, com os erros originais. Os
          comentários são da equipe e explicam as notas. Abra qualquer uma no corretor para comparar
          com a nota do modelo.
        </p>
      </header>

      <div className="space-y-24 pt-10">
        {EXEMPLOS.map((ex) => (
          <article key={ex.id} className="grid gap-10 lg:grid-cols-[1.4fr_1fr]">
            <div>
              <p className={cn("eyebrow", ROTULO[ex.qualidade].cor)}>{ROTULO[ex.qualidade].texto}</p>
              <h2 className="mt-2 text-3xl font-semibold">{ex.titulo}</h2>
              <p className="mt-1 text-sm text-muted-foreground">Tema: {ex.tema}</p>
              <p className="text-xs text-muted-foreground">
                Fonte: {ex.fonte} · nota dos corretores humanos
              </p>
              <div className="ruled-paper paper-shadow mt-6 whitespace-pre-line rounded-md">
                {ex.redacao}
              </div>
            </div>
            <aside className="space-y-6 lg:sticky lg:top-24 lg:self-start">
              <ScoreSeal nota={ex.notas.nota_final} className="mx-auto size-44" compacto />
              <ul className="space-y-2">
                {ex.comentarios.map((c) => (
                  <li key={c} className="border-l-2 border-pen/50 pl-3 font-serif text-[1.05rem] italic text-foreground/85">
                    {c}
                  </li>
                ))}
              </ul>
              <CompetenceList resultado={ex.notas} />
              <Button asChild variant="paper" className="w-full">
                <Link to="/" search={{ exemplo: ex.id }}>
                  Abrir no corretor <ArrowRight />
                </Link>
              </Button>
            </aside>
          </article>
        ))}
      </div>
    </div>
  );
}
