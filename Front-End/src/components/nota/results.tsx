import { useMemo } from "react";
import type { EssayResponse } from "@/lib/api";
import { interpretarFeedback } from "@/lib/scoring";
import { analisarTexto } from "@/lib/text-checks";
import { ScoreSeal } from "./score-seal";
import { CompetenceList } from "./competence-list";
import { FeedbackList } from "./feedback-list";
import { ChecksList } from "./checks-list";
import { AnnotatedEssay } from "./annotated-essay";
import { RewritePanel } from "./rewrite-panel";
import { PointersPanel } from "./pointers-panel";

export function Results({
  resultado,
  tema,
  redacao,
  demonstracao,
  onAvaliar,
}: {
  resultado: EssayResponse;
  tema: string;
  redacao: string;
  demonstracao: boolean;
  onAvaliar: (texto: string) => void;
}) {
  const feedback = useMemo(() => resultado.feedback.map(interpretarFeedback), [resultado]);
  const checks = useMemo(() => analisarTexto(redacao), [redacao]);
  const termos = useMemo(() => checks.flatMap((c) => c.termos ?? []), [checks]);
  const paragrafoFraco = feedback.find((f) => f.tipo === "paragrafo")?.paragrafo;

  return (
    <section id="resultado" className="scroll-mt-24 space-y-16">
      {demonstracao && (
        <p className="rounded-md bg-pen-soft px-4 py-3 text-sm text-pen">
          Demonstração: estas notas foram estimadas por regras simples, sem a rede neural. Ligue o
          servidor para receber a correção real.
        </p>
      )}

      <div className="grid gap-10 md:grid-cols-[auto_1fr] md:items-start">
        <div className="flex flex-col items-center gap-2">
          <p className="eyebrow">Nota final</p>
          <ScoreSeal nota={resultado.nota_final} />
        </div>
        <div>
          <p className="eyebrow">Competências do ENEM</p>
          <CompetenceList resultado={resultado} />
        </div>
      </div>

      <div className="grid gap-10 lg:grid-cols-[1fr_1.1fr]">
        <div>
          <p className="eyebrow">Diagnóstico da rede neural</p>
          <h2 className="mt-2 text-3xl font-semibold">O que corrigir</h2>
          <div className="mt-6">
            <FeedbackList itens={feedback} />
          </div>
        </div>
        <div>
          <p className="eyebrow">Sua redação, anotada</p>
          <h2 className="mt-2 text-3xl font-semibold">Onde está o problema</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            <span className="pen-underline">Sublinhado</span> marca repetições, oralidade e primeira
            pessoa. <span className="marker">Grifo</span> marca o parágrafo mais fraco.
          </p>
          <div className="mt-6">
            <AnnotatedEssay texto={redacao} termos={termos} paragrafoFraco={paragrafoFraco} />
          </div>
        </div>
      </div>

      <div>
        <p className="eyebrow">Verificações do texto</p>
        <h2 className="mt-2 text-3xl font-semibold">Checklist do corretor</h2>
        <p className="mt-2 max-w-2xl text-muted-foreground">
          Conferências automáticas feitas direto no seu texto, independentes da nota do modelo.
        </p>
        <div className="mt-6">
          <ChecksList itens={checks} />
        </div>
      </div>

      <PointersPanel key={`${tema}\n${redacao}`} tema={tema} redacao={redacao} />

      <RewritePanel tema={tema} redacao={redacao} onAvaliar={onAvaliar} />
    </section>
  );
}
