import { useEffect, useRef, useState } from "react";
import { createFileRoute, Link } from "@tanstack/react-router";
import { useMutation } from "@tanstack/react-query";
import { ArrowRight, Loader2, PenLine } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Results } from "@/components/nota/results";
import { avaliarRedacao, type EssayResponse } from "@/lib/api";
import { resultadoDemonstracao } from "@/lib/demo";
import { contarPalavras, MIN_PALAVRAS, separarParagrafos, validarRedacao } from "@/lib/scoring";
import { buscarExemplo } from "@/data/exemplos";
import { cn } from "@/lib/utils";

type Busca = { exemplo?: string };

export const Route = createFileRoute("/")({
  validateSearch: (s: Record<string, unknown>): Busca =>
    typeof s["exemplo"] === "string" ? { exemplo: s["exemplo"] } : {},
  head: () => ({
    meta: [
      { title: "Nota Mil — corrija sua redação do ENEM" },
      {
        name: "description",
        content: "Cole sua redação e receba nota de 0 a 1000 nas 5 competências do ENEM, com o que corrigir.",
      },
      { property: "og:title", content: "Nota Mil — corrija sua redação do ENEM" },
      {
        property: "og:description",
        content: "Nota nas 5 competências, diagnóstico por parágrafo e sugestão de reescrita.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

type Avaliacao = { resultado: EssayResponse; tema: string; redacao: string; demonstracao: boolean };

function Index() {
  const { exemplo } = Route.useSearch();
  const inicial = buscarExemplo(exemplo);
  const [tema, setTema] = useState(inicial?.tema ?? "");
  const [redacao, setRedacao] = useState(inicial?.redacao ?? "");
  const [erroForm, setErroForm] = useState<string | null>(null);
  const [avaliacao, setAvaliacao] = useState<Avaliacao | null>(null);
  const formRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const ex = buscarExemplo(exemplo);
    if (ex) {
      setTema(ex.tema);
      setRedacao(ex.redacao);
    }
  }, [exemplo]);

  const mut = useMutation({
    mutationFn: avaliarRedacao,
    onSuccess: (resultado, vars) =>
      setAvaliacao({ resultado, tema: vars.tema, redacao: vars.redacao, demonstracao: false }),
  });

  useEffect(() => {
    if (avaliacao) document.getElementById("resultado")?.scrollIntoView({ behavior: "smooth" });
  }, [avaliacao]);

  function enviar(textoRedacao = redacao) {
    const erro = validarRedacao(tema, textoRedacao);
    setErroForm(erro);
    if (erro) return;
    mut.mutate({ tema: tema.trim(), redacao: textoRedacao.trim() });
  }

  function demonstrar() {
    setAvaliacao({
      resultado: resultadoDemonstracao(redacao),
      tema,
      redacao,
      demonstracao: true,
    });
  }

  const palavras = contarPalavras(redacao);
  const paragrafos = separarParagrafos(redacao).length;

  return (
    <div className="mx-auto max-w-6xl px-4 sm:px-6">
      <section className="grid gap-8 pb-10 pt-14 md:grid-cols-[1.3fr_1fr] md:items-end md:pt-20">
        <div>
          <p className="eyebrow">Correção modelo ENEM</p>
          <h1 className="mt-4 text-5xl font-semibold leading-[1.02] sm:text-6xl">
            Sua redação, corrigida <em className="text-pen">antes</em> da prova.
          </h1>
        </div>
        <p className="font-serif text-lg leading-relaxed text-muted-foreground">
          Uma rede neural treinada com milhares de redações dá nota às 5 competências, mostra o
          parágrafo mais fraco e aponta o que corrigir.{" "}
          <Link to="/exemplos" className="text-ink underline decoration-pen underline-offset-4">
            Veja exemplos
          </Link>
          .
        </p>
      </section>

      <section ref={formRef} className="space-y-5">
        <div className="space-y-2">
          <Label htmlFor="tema" className="eyebrow">
            Tema
          </Label>
          <Input
            id="tema"
            value={tema}
            onChange={(e) => setTema(e.target.value)}
            placeholder="Ex.: Desafios para a formação de leitores no Brasil"
            className="h-12 border-0 border-b border-ink/30 bg-transparent px-0 font-display text-xl shadow-none focus-visible:border-pen focus-visible:ring-0"
          />
        </div>

        <div className="space-y-2">
          <div className="flex items-end justify-between">
            <Label htmlFor="redacao" className="eyebrow">
              Redação
            </Label>
            <span
              className={cn(
                "text-xs tabular-nums",
                palavras > 0 && palavras < MIN_PALAVRAS ? "text-pen" : "text-muted-foreground",
              )}
            >
              {palavras} palavras · {paragrafos} parágrafo{paragrafos === 1 ? "" : "s"}
            </span>
          </div>
          <textarea
            id="redacao"
            value={redacao}
            onChange={(e) => setRedacao(e.target.value)}
            placeholder="Cole ou escreva sua redação aqui. Separe os parágrafos com uma linha."
            rows={16}
            className="ruled-paper paper-shadow block w-full resize-y rounded-md border border-border text-ink outline-none placeholder:text-muted-foreground/70 focus:border-pen"
          />
        </div>

        {erroForm && <p className="text-sm text-pen">{erroForm}</p>}

        {mut.error && (
          <div className="flex flex-col gap-3 rounded-md bg-pen-soft p-4 text-sm text-pen sm:flex-row sm:items-center sm:justify-between">
            <p>{mut.error.message}</p>
            <Button variant="outline" size="sm" onClick={demonstrar}>
              Ver demonstração sem o servidor
            </Button>
          </div>
        )}

        <div className="flex flex-wrap items-center gap-4">
          <Button variant="pen" size="xl" onClick={() => enviar()} disabled={mut.isPending}>
            {mut.isPending ? <Loader2 className="animate-spin" /> : <PenLine />}
            {mut.isPending ? "Corrigindo..." : "Corrigir redação"}
          </Button>
          {!redacao && (
            <Link
              to="/"
              search={{ exemplo: "leitura-mediana" }}
              className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-ink"
            >
              Testar com uma redação de exemplo <ArrowRight className="size-4" />
            </Link>
          )}
        </div>
      </section>

      {avaliacao && (
        <div className="mt-20 border-t border-border pt-16">
          <Results
            key={`${avaliacao.redacao.length}-${avaliacao.resultado.nota_final}-${avaliacao.demonstracao}`}
            {...avaliacao}
            onAvaliar={(texto) => {
              setRedacao(texto);
              formRef.current?.scrollIntoView({ behavior: "smooth" });
              enviar(texto);
            }}
          />
        </div>
      )}
    </div>
  );
}
