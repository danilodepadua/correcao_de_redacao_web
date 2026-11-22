import { Fragment } from "react";
import { separarParagrafos } from "@/lib/scoring";
import { cn } from "@/lib/utils";

function escapar(s: string) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function destacar(texto: string, termos: string[]) {
  if (termos.length === 0) return texto;
  const re = new RegExp(`(?<![\\p{L}])(${termos.map(escapar).join("|")})(?![\\p{L}])`, "giu");
  const partes = texto.split(re);
  return partes.map((parte, i) =>
    i % 2 === 1 ? (
      <mark key={i} className="pen-underline bg-transparent text-inherit">
        {parte}
      </mark>
    ) : (
      <Fragment key={i}>{parte}</Fragment>
    ),
  );
}

/** Mostra a redação em folha pautada, destacando termos problemáticos e o parágrafo mais fraco. */
export function AnnotatedEssay({
  texto,
  termos,
  paragrafoFraco,
}: {
  texto: string;
  termos: string[];
  paragrafoFraco?: number | undefined;
}) {
  const paragrafos = separarParagrafos(texto);
  return (
    <div className="ruled-paper paper-shadow relative rounded-md">
      {paragrafos.map((p, i) => {
        const fraco = paragrafoFraco === i + 1;
        return (
          <div key={i} className="relative">
            <span className="absolute -left-12 top-0 w-8 text-right font-sans text-xs text-muted-foreground">
              §{i + 1}
            </span>
            <p className={cn("indent-8", fraco && "marker")}>{destacar(p, termos)}</p>
            {fraco && (
              <p className="font-serif text-base italic text-pen">
                ↳ Parágrafo que menos pesou na nota: enriqueça os argumentos ou o detalhamento aqui.
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}
