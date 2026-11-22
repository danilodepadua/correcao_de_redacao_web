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
  { valor: "4.570", rotulo: "redações corrigidas no Essay-BR (80% usadas no treino)" },
  { valor: "86", rotulo: "temas diferentes" },
  { valor: "5", rotulo: "notas, uma por competência" },
  { valor: "0,56", rotulo: "QWK: concordância com os corretores em redações fora do treino" },
];

const PASSOS = [
  ["Filtro", "Textos sem sentido (palavras repetidas, letras soltas, números) são recusados antes da rede."],
  ["Leitura", "O tema e a redação (até 512 tokens) viram entrada para o BERTimbau, um modelo de linguagem em português."],
  ["Nota", "Uma camada final prevê as 5 competências; cortes ajustados nos dados convertem a previsão em 0, 40, …, 200."],
  ["Diagnóstico", "Notas até 160 geram comentários sobre o que corrigir em cada competência."],
  ["Parágrafos", "O modelo dá nota ao texto sem cada parágrafo para achar o que menos contribui."],
  ["IA generativa", "O Gemini aponta trechos com problema e reescreve a redação; a versão nova volta para o modelo."],
];

/**
 * Experimento de viés (Back-End/experimentos/analise_vies.py) em 914 redações fora do treino:
 * variação média da nota (0–1000) antes (v2) e depois (v3) do treino contra vieses.
 */
const VIESES = [
  { mudanca: "Repetir um parágrafo inteiro", antes: "+33", depois: "−55", treino: true, leitura: "Corrigido: repetição passou a tirar ponto" },
  { mudanca: "Colar uma citação famosa sem relação com o tema", antes: "+31", depois: "+9", treino: true, leitura: "Bem menor, mas ainda ajuda um pouco" },
  { mudanca: "Embaralhar a ordem das frases", antes: "−13", depois: "−138", treino: true, leitura: "Corrigido: agora percebe a desorganização" },
  { mudanca: "Pôr conectivos no início dos parágrafos", antes: "+9", depois: "+6", treino: false, leitura: "Quase igual: fora do treino, o viés continua" },
  { mudanca: "Remover a conclusão", antes: "−66", depois: "−114", treino: false, leitura: "Esperada: cai mais" },
  { mudanca: "Tirar todos os acentos", antes: "−115", depois: "−122", treino: false, leitura: "Esperada" },
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
        <h2 className="mt-2 text-3xl font-semibold">O que engana o modelo, e como corrigimos</h2>
        <p className="mt-2 font-serif text-lg text-muted-foreground">
          Aplicamos mudanças controladas a 914 redações fora do treino e medimos a variação média da
          nota (de 0 a 1000). A primeira versão do modelo premiava texto repetido e citações genéricas
          e quase não percebia frases embaralhadas. A versão atual foi treinada também com cópias
          alteradas de redações, com nota menor ou igual, e esses vieses caíram muito. Manipulações que
          ficaram fora desse treino, como conectivos inseridos à força, ainda enganam um pouco. Por
          isso o checklist do corretor e os apontamentos não dependem só da nota.
        </p>
        <div className="mt-6 overflow-x-auto">
          <table className="w-full min-w-[36rem] text-left text-sm">
            <thead className="text-muted-foreground">
              <tr className="border-b border-border">
                <th className="py-2 pr-4 font-medium">Mudança no texto</th>
                <th className="py-2 pr-4 font-medium">Antes</th>
                <th className="py-2 pr-4 font-medium">Agora</th>
                <th className="py-2 pr-4 font-medium">No treino?</th>
                <th className="py-2 font-medium">Leitura</th>
              </tr>
            </thead>
            <tbody>
              {VIESES.map((v) => (
                <tr key={v.mudanca} className="border-b border-border/60">
                  <td className="py-2 pr-4">{v.mudanca}</td>
                  <td className="py-2 pr-4 text-muted-foreground">{v.antes}</td>
                  <td className="py-2 pr-4 font-semibold">{v.depois}</td>
                  <td className="py-2 pr-4">{v.treino ? "sim" : "não"}</td>
                  <td className="py-2">{v.leitura}</td>
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
