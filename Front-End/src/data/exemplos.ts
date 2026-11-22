import type { EssayResponse } from "@/lib/api";

export type Exemplo = {
  id: string;
  qualidade: "boa" | "mediana" | "fraca";
  titulo: string;
  /** Resumo do tema da proposta (o Essay-BR traz só o texto motivador). */
  tema: string;
  /** De onde vem a redação. */
  fonte: string;
  redacao: string;
  /** Notas dadas por corretores humanos, como estão no dataset. */
  notas: Omit<EssayResponse, "feedback">;
  comentarios: string[];
};

/**
 * Redações reais do Essay-BR, escolhidas no conjunto de teste (o modelo não as viu no
 * treino). O texto é o original, sem correções; as notas são as dos corretores humanos.
 * Os comentários foram escritos pela equipe para explicar as notas.
 */
export const EXEMPLOS: Exemplo[] = [
  {
    id: "esporte-boa",
    qualidade: "boa",
    titulo: "Tese clara, argumentos encadeados e proposta com agentes",
    tema: "A importância do esporte na formação das pessoas",
    fonte: "Essay-BR, índice 3544 do essay-br.csv (conjunto de teste do modelo)",
    redacao: `O esporte como componente curricular nas escolas é um poderoso instrumento pedagógico de inserção e interação entre os alunos, além de proporcionar hábitos de vida saudável e bem-estar físico e social. Porém, para que esses efeitos benéficos sejam atingidos, as práticas educativas devem ser tratadas de forma eficiente e pensadas de modo a serem resolutivas em suas propostas, evitando o paradigma de que esta disciplina está limitada apenas à prática de esportes como futebol, vôlei e handebol, por exemplo.

Para isso, o planejamento das aulas deve ser voltado ao prejulgamento de que os indivíduos são distintos e particulares em suas características, evitando assim um pensamento errôneo de que todos têm aptidões para as atividades importas pelo professor. Caso contrário, os esportes perderão seus papéis de inclusão e integração, dando espaço à distinção dos alunos, principalmente àqueles que não se sentem bem participando das dinâmicas propostas, sendo por vezes alvos de brincadeiras e discriminações.

Com isso, além da Educação Física propor a prática de esportes em suas aulas, outros aspectos serão trabalhados como a cultura, a saúde, a socialização e o incentivo ao respeito pelas diferenças, formando cidadãos melhores para a sociedade ao atender as expectativas dos alunos que se sentirão menos pressionados e aprenderão de forma prazerosa a se relacionarem e compartilharem seus talentos, seja fazendo gols ou dedilhando notas em um violão.

Portanto, para que a Educação Física e as práticas esportivas atinjam seus objetivos de forma satisfatória, deve haver um incentivo por parte do Governo, junto aos Ministérios da Educação e do Esporte, bem como a qualificação dos profissionais e professores que atuam na área. Assim, de forma conjunta, pode-se educar e construir sem discriminações e sem padrões preestabelecidos, propondo uma transição da educação esportiva onde meninos jogam futebol, meninas vôlei e alguns apenas observam, para uma realidade bem menos discriminativa, onde cada um pratica aquilo que lhe faz bem, sem ninguém ficar de fora.`,
    notas: {
      c1_norma_culta: 200,
      c2_compreensao_tema: 160,
      c3_projeto_de_texto: 200,
      c4_coesao: 200,
      c5_proposta_intervencao: 200,
      nota_final: 960,
    },
    comentarios: [
      "A tese aparece já na introdução: o esporte na escola só inclui se não se limitar a futebol, vôlei e handebol.",
      "Os parágrafos ligam causa e consequência: aulas iguais para todos levam à exclusão e à discriminação de alguns alunos.",
      "A proposta tem agentes (Governo, Ministérios da Educação e do Esporte), ação e finalidade.",
      "Perdeu 40 pontos só na C2, provavelmente por não trazer repertório externo (dados, leis ou autores).",
    ],
  },
  {
    id: "analfabetismo-mediana",
    qualidade: "mediana",
    titulo: "Tema compreendido, mas afirmações sem prova e proposta vaga",
    tema: "Analfabetismo funcional no Brasil",
    fonte: "Essay-BR, índice 4250 do essay-br.csv (conjunto de teste do modelo)",
    redacao: `Analfabetismo funcional no Brasil: por que esse problema ainda persiste?

São chamados de analfabetos funcionais aqueles que reconhecem algumas palavras, porém não possuem capacidade de compreender textos, por mais que hajam palavras de seu dia-a-dia, a coerência não é introduzida em suas mentes.

O analfabetismo funcional vem diminuindo cada vez mais com o passar dos anos, pois as pessoas têm grande acesso à internet e tecnologias, o que os estimula a se interessar pelas notícias globais ou videogames, e consequentemente, tal ato aprimora sua leitura e vocabulário.

Contudo, há aproximadamente 60 anos, essas inovações tecnológicas não faziam parte do cotidiano da população. Muitas das pessoas que viviam no campo, não conseguiam chegar às escolas, por estas serem longe o suficiente para não possuírem meios de transportes para se locomover.

Todavia, ainda há analfabetos pelo Brasil e alguns casos são responsabilidades do grau de necessidade passada dentro de seus lares familiares, como trocar a escola pelo trabalho e ter de colocar o sustento dentro de casa. Há casos também, em que o estudante vai perdendo o ânimo por acreditar que não é capaz de concluir, e com isso sua motivação vai diminuindo, até ele desistir completamente e passar a vida toda se sujeitando em trabalhos com míseros salários.

Portanto, o governo atual possui sistemas para melhorar essa situação, mas não são todos que se interessam. Estudos comprovam que encontrar uma forma divertida para os professores passarem as matérias mais tediosas, seria introduzindo jogos com o tema da matéria a ser estudada, fazendo com que todos os alunos se interessassem cada vez mais pelas aulas e gerando interesse de estudar por conta própria.`,
    notas: {
      c1_norma_culta: 120,
      c2_compreensao_tema: 120,
      c3_projeto_de_texto: 80,
      c4_coesao: 120,
      c5_proposta_intervencao: 80,
      nota_final: 520,
    },
    comentarios: [
      "Começa definindo o problema, mas não apresenta uma tese sobre por que ele persiste.",
      "Desvios de norma: “por mais que hajam palavras” (o certo é “haja”) e vírgula entre sujeito e verbo em “Muitas das pessoas que viviam no campo, não conseguiam”.",
      "Afirma sem comprovar que o problema “vem diminuindo” por causa da internet e cita “estudos” sem dizer quais.",
      "A proposta (jogos nas aulas) não diz quem fará, como nem com que recursos.",
    ],
  },
  {
    id: "polarizacao-fraca",
    qualidade: "fraca",
    titulo: "Opiniões soltas, sem projeto de texto nem proposta",
    tema: "Polarização e o respeito a opiniões diferentes",
    fonte: "Essay-BR, índice 457 do essay-br.csv (conjunto de teste do modelo)",
    redacao: `Brasileiros são muitas vez um povo que gosta de uma discussão. Seja nas redes sócias ou num barzinho, também na maioria das vezes em casa mesmo. Ocorre que essa discussões não tem fundamento pois os argumentos são baseados em uma ideologia fragmentada. Quando ocorre atos de desentendimentos a opinião do outro pode ser vista como uma ameaça e devido a isto ocorre brigas sem nexo.

A maioria não sabe lidar com opiniões diferentes e já partem para agressão física. Muitas vezes os fatos são baseados em caráter religioso ou político. A crença de ter uma argumentação melhor dá a entender que esse tipo de indivíduo e um ser preferível no padrão social.

A lógica argumentativa e de que o conhecimento é algo a ser praticado e transmitido de forma simples e racional Não e necessidade de refutar fatos para apresentar uma apresentação plausível. E possível que o debate seja uma forma de respeitar a opinião do próximo através de controle emocional para aprender com o que este de para oferecer. Assim e necessário mais leitura e interpretação dos fatos para chegar em uma conclusão de tolerância e ética.`,
    notas: {
      c1_norma_culta: 80,
      c2_compreensao_tema: 80,
      c3_projeto_de_texto: 0,
      c4_coesao: 80,
      c5_proposta_intervencao: 0,
      nota_final: 240,
    },
    comentarios: [
      "Muitos desvios de norma: “muitas vez”, “redes sócias”, “essa discussões não tem”, “e” no lugar de “é”.",
      "Não há tese nem progressão: os parágrafos reúnem opiniões soltas, o que zerou a C3.",
      "Termina só com “é necessário mais leitura”, sem agente, meio ou detalhamento, o que zerou a C5.",
    ],
  },
];

export function buscarExemplo(id: string | undefined) {
  return EXEMPLOS.find((e) => e.id === id);
}
