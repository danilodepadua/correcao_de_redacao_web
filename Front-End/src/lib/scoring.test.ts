import { describe, expect, it } from "vitest";
import { interpretarFeedback, nivelCompetencia, validarRedacao } from "./scoring";
import { analisarTexto, elementosIntervencao } from "./text-checks";

const palavras = (n: number) => Array.from({ length: n }, () => "palavra").join(" ");

describe("regra das 30 palavras", () => {
  it("recusa redação com 29 palavras", () => {
    expect(validarRedacao("Tema", palavras(29))).toMatch(/30 palavras/);
  });
  it("aceita redação com 30 palavras", () => {
    expect(validarRedacao("Tema", palavras(30))).toBeNull();
  });
  it("exige tema", () => {
    expect(validarRedacao("  ", palavras(40))).not.toBeNull();
  });
});

describe("leitura do feedback do back-end", () => {
  it("lê competência e nota de um alerta", () => {
    const item = interpretarFeedback(
      "[Atenção - 80/200] Competência 1 (Norma Culta): Foram detectados desvios gramaticais.",
    );
    expect(item.tipo).toBe("atencao");
    expect(item.competencia).toBe(1);
    expect(item.nota).toBe(80);
    expect(item.titulo).toBe("Competência 1 (Norma Culta)");
  });
  it("lê o número do parágrafo mais fraco", () => {
    const item = interpretarFeedback(
      '[Análise de Parágrafo] O 3º parágrafo ("Texto...") apresentou menor impacto na pontuação.',
    );
    expect(item.tipo).toBe("paragrafo");
    expect(item.paragrafo).toBe(3);
  });
  it("marca nota 160 como 'pode melhorar'", () => {
    expect(interpretarFeedback("[Bom, mas pode melhorar - 160/200] Competência 4 (Coesão): x").tipo).toBe(
      "melhorar",
    );
  });
});

describe("níveis por competência", () => {
  it("200 é excelente e 120 é mediano", () => {
    expect(nivelCompetencia(200).rotulo).toBe("Excelente");
    expect(nivelCompetencia(120).rotulo).toBe("Mediano");
  });
});

describe("verificações do texto", () => {
  it("aponta parágrafo sem conectivo", () => {
    const texto = "Introdução do texto.\nO segundo parágrafo começa seco.\nAlém disso, o terceiro.\nPortanto, fim.";
    const c = analisarTexto(texto).find((v) => v.id === "conectivos");
    expect(c?.ok).toBe(false);
    expect(c?.detalhe).toContain("2");
  });
  it("detecta marca de oralidade", () => {
    const o = analisarTexto("A gente precisa pensar nisso.").find((v) => v.id === "oralidade");
    expect(o?.ok).toBe(false);
  });
  it("encontra os cinco elementos da proposta", () => {
    const conclusao =
      "Portanto, o Ministério da Educação deve criar oficinas, por meio de parcerias com escolas, a fim de ampliar a leitura, ou seja, formar leitores críticos.";
    expect(elementosIntervencao(conclusao).every((e) => e.presente)).toBe(true);
  });
});
