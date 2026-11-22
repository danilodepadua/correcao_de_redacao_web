import { nivelFinal } from "@/lib/scoring";
import { cn } from "@/lib/utils";
import { TOM_TEXTO } from "./tone";

/** Nota final circulada à mão, como no canto da folha corrigida. */
export function ScoreSeal({
  nota,
  className,
  compacto = false,
}: {
  nota: number;
  className?: string;
  compacto?: boolean;
}) {
  const nivel = nivelFinal(nota);
  const comprimento = 640;
  return (
    <div className={cn("relative grid size-56 place-items-center", className)}>
      <svg viewBox="0 0 220 220" className="absolute inset-0 size-full text-pen" fill="none" aria-hidden>
        <path
          d="M110 18c52 0 92 36 92 90 0 56-42 96-96 94-52-2-88-42-86-92 2-48 40-86 96-88 20 0 34 4 46 10"
          stroke="currentColor"
          strokeWidth="3"
          strokeLinecap="round"
          strokeDasharray={comprimento}
          style={{ ["--dash" as string]: `${comprimento}`, ["--dash-end" as string]: "0" }}
          className="animate-draw"
        />
      </svg>
      <div className="text-center">
        <div
          className={cn(
            "font-display font-semibold leading-none text-pen tabular-nums",
            compacto ? "text-5xl" : "text-7xl",
          )}
        >
          {nota}
        </div>
        <div className="mt-1 text-sm text-muted-foreground">de 1000</div>
        <div className={cn("mt-2 font-serif text-lg italic", TOM_TEXTO[nivel.tom])}>{nivel.rotulo}</div>
      </div>
    </div>
  );
}
