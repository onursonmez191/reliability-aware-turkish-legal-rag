// Sample legal Q&A data — illustrative content for the prototype.
// Real system retrieves from a Turkish legal corpus; shown in English here.

const SAMPLE_QUESTIONS = [
  {
    id: "q1",
    q: "Can a tenant move out before the lease term ends?",
    kind: "supported",
  },
  {
    id: "q2",
    q: "I believe I was wrongfully dismissed from my job — what can I do?",
    kind: "partial",
  },
  {
    id: "q3",
    q: "My neighbor's dog bit me — who can I claim compensation from?",
    kind: "risk",
  },
  {
    id: "q4",
    q: "My sibling won't transfer the title of our late mother's flat. What now?",
    kind: "insufficient",
  },
];

const ANSWERS = {
  q1: {
    answer:
      "A tenant may vacate the premises before the lease term expires, but as a rule must observe a reasonable notice period and compensate the landlord for the rent loss until a new tenant is found [1]. If the contract contains a specific early-termination clause, that clause applies first [2]. Where justified grounds exist — such as serious illness, mandatory relocation, or marriage — the compensation obligation may be reduced; this is assessed case by case [3].",
    sources: [
      {
        id: "S-04412",
        title: "Residential Lease — Early Termination and Tenant Liability",
        score: 0.91,
        snippet:
          "If the tenant wishes to vacate the property before the term ends, they must give the landlord reasonable advance notice and pay rent for the reasonable period until a new tenant is found…",
        tag: "TCO Art. 325",
      },
      {
        id: "S-02118",
        title: "Contractual Early-Termination Clauses",
        score: 0.84,
        snippet:
          "Parties may agree in the lease contract on indemnity, notice period, and penal clauses for early termination. In such cases the contract provisions apply with priority…",
        tag: "Doctrine",
      },
      {
        id: "S-07731",
        title: "Termination on Justified Grounds — Relocation, Health, Marriage",
        score: 0.71,
        snippet:
          "Where justified grounds beyond the tenant's control exist (mandatory relocation, serious health condition, marriage, etc.), the compensation calculation may vary…",
        tag: "Case law",
      },
      {
        id: "S-09102",
        title: "Landlord's Duty to Mitigate",
        score: 0.62,
        snippet:
          "The landlord must take reasonable steps to mitigate loss arising from the tenant's early departure; otherwise, the awarded compensation may be reduced…",
        tag: "Precedent",
      },
    ],
    verdict: {
      label: "Supported",
      key: "supported",
      score: 0.86,
      claims: [
        { text: "Tenant may exit before the term ends.", status: "supported", src: [1] },
        { text: "Obligation to give reasonable notice.", status: "supported", src: [1] },
        { text: "Compensation until a new tenant is found.", status: "supported", src: [1, 4] },
        { text: "Contractual clause takes precedence.", status: "supported", src: [2] },
        { text: "Justified-grounds assessment.", status: "partial", src: [3] },
      ],
      risk: "low",
    },
    llmOnly:
      "Yes, a tenant can always leave and isn't required to pay the landlord anything. Usually one month's notice is enough.",
  },
  q2: {
    answer:
      "If an employee believes their contract was terminated by the employer without just cause, they must apply to a mediator within one month of receiving the termination notice [1]. If mediation fails, a reinstatement lawsuit can be filed within two weeks [2]. The employee must have at least six months' seniority at a workplace employing thirty or more workers [3]. If the suit is upheld, reinstatement and compensation follow; concrete amounts and timelines vary by case.",
    sources: [
      {
        id: "S-12044",
        title: "Job Security — Mandatory Mediation",
        score: 0.88,
        snippet:
          "Application to a mediator within one month of the date the termination notice is served to the employee is mandatory; otherwise the lawsuit is dismissed for lack of a procedural prerequisite…",
        tag: "Labor Act Art. 20",
      },
      {
        id: "S-12091",
        title: "Reinstatement Suit — Time Limit and Competent Court",
        score: 0.81,
        snippet:
          "The lawsuit may be filed in the labor court within two weeks of the date of the final mediation minutes recording non-agreement…",
        tag: "Labor Act Art. 20",
      },
      {
        id: "S-12110",
        title: "Job Security Scope — Seniority and Workplace Threshold",
        score: 0.74,
        snippet:
          "To benefit from job-security provisions, the workplace must employ thirty or more workers and the employee must have at least six months' seniority…",
        tag: "Labor Act Art. 18",
      },
    ],
    verdict: {
      label: "Partially supported",
      key: "partial",
      score: 0.62,
      claims: [
        { text: "Mandatory mediation within 1 month.", status: "supported", src: [1] },
        { text: "Reinstatement suit within 2 weeks.", status: "supported", src: [2] },
        { text: "30-worker / 6-month seniority threshold.", status: "supported", src: [3] },
        { text: "Specific compensation amounts.", status: "unsupported", src: [] },
      ],
      risk: "medium",
    },
    llmOnly:
      "File a case in labor court. Hiring a lawyer would be a good idea. You'll usually get about 4 months' salary as compensation.",
  },
  q3: {
    answer:
      "The described situation involves personal injury and potential tort liability, which depends heavily on the specific facts. In general, the rules on strict liability of the animal keeper apply [1]. However, gathering concrete evidence (medical report, witnesses, scene records), calculating damages, and meeting filing deadlines all require consulting a qualified lawyer.",
    sources: [
      {
        id: "S-05231",
        title: "Liability of the Animal Keeper",
        score: 0.69,
        snippet:
          "A person who continuously or temporarily takes on the care and management of an animal is liable for the damages caused by that animal…",
        tag: "TCO Art. 67",
      },
    ],
    verdict: {
      label: "Legal-advice risk",
      key: "risk",
      score: 0.41,
      claims: [
        { text: "Animal keeper bears liability.", status: "supported", src: [1] },
        { text: "Case-specific amount / deadline advice.", status: "risk", src: [] },
      ],
      risk: "high",
    },
    llmOnly:
      "Sue your neighbor and ask for 50,000 lira in damages. The court will definitely grant it.",
  },
  q4: {
    answer:
      "Your question spans inheritance law and land-registry law, which is a multi-layered situation. The available sources do not adequately cover this specific case. Generally, a partition action (action for the dissolution of co-ownership) may be filed; however, concrete details require expert legal advice.",
    sources: [
      {
        id: "S-08820",
        title: "Inheritance — Partition Action (General)",
        score: 0.58,
        snippet:
          "If the heirs cannot reach agreement on partition, any heir may request the dissolution of co-ownership from the civil court of peace…",
        tag: "Civil Code Art. 642",
      },
    ],
    verdict: {
      label: "Insufficient context",
      key: "insufficient",
      score: 0.35,
      claims: [
        { text: "Partition action is possible.", status: "partial", src: [1] },
        { text: "Concrete title-transfer steps.", status: "unsupported", src: [] },
      ],
      risk: "medium",
    },
    llmOnly:
      "Go to the land registry and transfer the title without your sibling's consent. A notary is sufficient.",
  },
};

const PIPELINE_STEPS = [
  { id: "embed", label: "Embed query", sub: "multilingual-e5-base" },
  { id: "retrieve", label: "Vector search", sub: "FAISS · top-k=5" },
  { id: "rerank", label: "Rerank", sub: "cross-encoder" },
  { id: "generate", label: "Generate answer", sub: "source-grounded" },
  { id: "verify", label: "Verifier", sub: "claim-by-claim check" },
];

window.SAMPLE_QUESTIONS = SAMPLE_QUESTIONS;
window.ANSWERS = ANSWERS;
window.PIPELINE_STEPS = PIPELINE_STEPS;
