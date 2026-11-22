import type { Tom } from "@/lib/scoring";

export const TOM_TEXTO: Record<Tom, string> = {
  otimo: "text-score-great",
  bom: "text-score-good",
  medio: "text-score-mid",
  ruim: "text-score-low",
};

export const TOM_FUNDO: Record<Tom, string> = {
  otimo: "bg-score-great",
  bom: "bg-score-good",
  medio: "bg-score-mid",
  ruim: "bg-score-low",
};
