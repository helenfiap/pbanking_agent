# Personal Banker Copilot

> Agente Text-to-SQL para Personal Bankers da Franq — consulte dados comportamentais de clientes em linguagem natural e receba SQL, gráficos e respostas em PT-BR.

![Arquitetura](docs/architecture.png)

---

## O Problema

A Franq opera um marketplace com mais de 150 produtos financeiros. Personal Bankers precisam entender o comportamento dos clientes — padrões de compra, respostas a campanhas, tendências de reclamações — para fazer o match certo com cada produto.

Hoje isso significa escrever SQL manualmente ou esperar pelo time de BI. Este copilot elimina os dois gargalos: o banker digita uma pergunta em português e recebe uma resposta estruturada com gráfico, o SQL que a gerou e o raciocínio por trás.

---

## Como Executar

```bash
# 1. Clone e instale
git clone <repo>
cd personal_banker_copilot
uv sync

# 2. Configure as chaves
cp .env.example .env
# Preencha GOOGLE_API_KEY e/ou AZURE_OPENAI_* — veja .env.example

# 3. Execute
uv run streamlit run app.py
```

Requer Python 3.11+. Usa [uv](https://github.com/astral-sh/uv) para gerenciamento de dependências.

---

## Arquitetura

O agente é implementado como um **StateGraph de 7 nós com LangGraph**. O desafio pede explicitamente navegação em incertezas, correção autônoma de erros e transparência no raciocínio — condições que se mapeiam naturalmente em um grafo direcionado com estado tipado e arestas condicionais.

```
INÍCIO
  │
  ▼
① Inspetor de Schema    ← Descoberta via PRAGMA — sem LLM, sempre atualizado
  │
  ▼
② Planejador            ← LLM pesado — plano de raciocínio antes de qualquer SQL
  │
  ▼
③ Gerador de SQL        ← LLM pesado — query específica para SQLite
  │
  ▼
④ Executor SQL          ← Sem LLM — executa query e captura exceções
  │
  ├── erro + tentativas < MAX ──▶ ⑤ Recuperação de Erro ──┐
  │                                  LLM leve               │ (loop de retry)
  │
  ├── máx. tentativas excedidas ──▶ ⑦ Formatador
  │
  └── sucesso ──▶ ⑥ Visualização ──▶ ⑦ Formatador ──▶ FIM
                    Híbrido: regras + LLM leve
```

### Estado Compartilhado

```python
class AgentState(TypedDict):
    question: str       # pergunta original do usuário
    schema:   str       # saída live do PRAGMA
    plan:     str       # raciocínio do planejador
    sql:      str       # SQL atual (mutado pela recuperação de erro)
    result:   str       # JSON string com as linhas da query
    error:    str       # última exceção do SQLite
    retries:  int       # contador de tentativas
    viz_spec: dict      # {type, x, y, orientation, title}
    response: str       # resposta final em PT-BR
    trace:    list[str] # passos com latência por nó
```

---

## Decisões de Engenharia

### Por que LangGraph?

O desafio pede um agente que "navega incertezas, busca suas próprias respostas, percebe erros e tenta corrigir sozinho." Essas são transições de estado, não uma cadeia linear. O LangGraph as modela como arestas reais de grafo — o loop de retry entre executor e recuperação de erro é uma aresta condicional de verdade, não um `for` loop dentro de um monólito.

### Por que um nó Planejador antes do gerador de SQL?

Perguntas complexas com múltiplos joins falham na geração de SQL sem raciocínio prévio. O planejador força o LLM pesado a identificar quais tabelas são necessárias e por quê antes de escrever qualquer SQL. Nos testes, isso reduziu significativamente nomes de colunas alucinados e JOINs faltando.

### Por que roteamento híbrido de visualização?

Chamar um LLM para decidir "isso deve ser tabela ou gráfico?" custa tokens em cada query. O agente de visualização usa três camadas ordenadas por custo:

1. **Pré-filtros determinísticos** (sem LLM): escalar único → `métrica`, >15 linhas → `tabela`, linha única → `tabela`
2. **Classificador de intenção por palavras-chave** (sem LLM): "top/maior/menor" → ranking → barra horizontal; "tendência/mês a mês" → linha
3. **LLM leve** (só para resultados ambíguos de 2 a 15 linhas): recebe dica de intenção + tipos de coluna + amostra dos dados

~80% das queries nunca chegam à chamada LLM. Isso importa em escala.

### Por que abstração de provedores?

Seleção de LLM é uma decisão de engenharia, não de produto — tem restrições de latência, custo e confiabilidade que mudam ao longo do tempo. Isolar a escolha do provedor atrás de uma interface comum significa que os sete nós nunca mudam quando o modelo subjacente muda:

```python
def get_llm(tier: str = "heavy") -> BaseChatModel:
    # resolve thread-local → variável de ambiente → padrão
    # retorna ChatOpenAI ou ChatGoogleGenerativeAI
    # o chamador nunca sabe qual
```

Isso também habilitou a aba de benchmark — veja abaixo.

> **Alinhamento com produção Franq:** a vaga menciona Vertex AI (Gemini) com FastAPI. Esta demo usa Google AI Studio (free tier) para desenvolvimento — a migração para Vertex AI é apenas uma troca de classe no factory (`VertexAI` em vez de `ChatGoogleGenerativeAI`), sem alterar nenhum nó do agente. O wrapper FastAPI ficaria na camada acima do `agent_graph.invoke()`, expondo um endpoint `POST /query` para integração com outros sistemas internos.

### Por que uma aba de benchmark?

Inicialmente o desafio pedia um deploy com modelo único. Durante o desenvolvimento ficou claro que escolher o LLM certo é parte do problema de engenharia em si. Em vez de fazer essa análise offline em um notebook, expus como ferramenta dentro do app:

- Todos os modelos rodam em paralelo via `ThreadPoolExecutor`
- Resultados aparecem ao vivo conforme cada modelo termina
- Cada run é anexado em `data/llms_comparison.csv` para comparação histórica

O benchmark é uma **ferramenta de desenvolvimento e operações**, não uma feature para o usuário final. Ele responde à pergunta "qual modelo devemos deployar em produção?" com dados reais da carga de trabalho real.

---

## Análise de Custos — LLMs

Resultados de runs reais de benchmark com pergunta de efetividade de campanhas:

| Modelo | Provedor | Latência | Custo/query | Observação |
|---|---|---|---|---|
| GPT-5.4 Nano | Azure Foundry | 8.7s | **$0.00011** | Melhor custo-benefício, respostas concisas |
| Kimi K2.5 | Azure Foundry | ~48s | $0.00036 | Único que retorna receita em R$ |
| GPT-4.1 | Azure Foundry | 6.4s | $0.00054 | **Sweet spot** — metade do custo do 4o |
| Grok 4.1 Fast | Azure Foundry | 8.4s | $0.00054 | Boa formatação em percentuais |
| GPT-4o | Azure Foundry | 7.3s | $0.00108 | Narrativa mais completa, maior custo |
| Gemini 2.5 Flash | Google AI Studio | ~8s | $0.00030 | Tier gratuito: 40 req/dia com rotação de 2 chaves |
| Gemini 2.0 Flash Lite | Google AI Studio | ~5s | $0.00010 | Tier gratuito: 1.500 req/dia — ideal para dev |

**Recomendação para produção:** GPT-4.1 como padrão (equilíbrio custo/qualidade). Gemini 2.0 Flash Lite para desenvolvimento e tooling interno de alto volume.

Cada modelo interpreta o mesmo resultado SQL de forma diferente — GPT-4o enfatiza narrativa, Kimi retorna receita absoluta, Grok usa percentuais. Isso é sinal útil na hora de decidir qual modelo expor aos usuários finais.

---

## Exemplos de Consultas Testadas

As cinco perguntas do enunciado, com o SQL gerado e o resultado real do banco:

---

**1. "Liste os 5 estados com maior número de clientes que compraram via app em maio."**

```sql
SELECT c.estado, COUNT(DISTINCT c.id) AS clientes
FROM clientes c JOIN compras cp ON c.id = cp.cliente_id
WHERE LOWER(cp.canal) = 'app' AND strftime('%m', cp.data_compra) = '05'
GROUP BY c.estado ORDER BY clientes DESC LIMIT 5;
```
```
estado           clientes
São Paulo               6
Santa Catarina          3
Minas Gerais            3
Paraná                  2
Espírito Santo          2
```
→ Visualização: **gráfico de barras horizontal** (intenção: ranking)

---

**2. "Quantos clientes interagiram com campanhas de WhatsApp em 2024?"**

```sql
SELECT COUNT(DISTINCT cliente_id) AS total_clientes
FROM campanhas_marketing
WHERE canal = 'WhatsApp' AND strftime('%Y', data_envio) = '2024' AND interagiu = 1;
```
```
total_clientes
            17
```
→ Visualização: **métrica** (resultado escalar único)

---

**3. "Quais categorias de produto tiveram o maior número de compras em média por cliente?"**

```sql
SELECT categoria, ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT cliente_id), 2) AS media_compras
FROM compras GROUP BY categoria ORDER BY media_compras DESC LIMIT 5;
```
```
categoria    media_compras
Roupas                2.21
Viagens               2.16
Livros                1.98
Serviços              1.96
Eletrônicos           1.92
```
→ Visualização: **gráfico de barras horizontal** (intenção: ranking)

---

**4. "Qual o número de reclamações não resolvidas por canal?"**

```sql
SELECT canal, COUNT(*) AS nao_resolvidas
FROM suporte WHERE resolvido = 0
GROUP BY canal ORDER BY nao_resolvidas DESC;
```
```
canal     nao_resolvidas
Chat                  51
Telefone              46
E-mail                41
```
→ Visualização: **gráfico de barras** (intenção: comparação)

---

**5. "Qual a tendência de reclamações por canal no último ano?"**

```sql
SELECT canal, strftime('%Y-%m', data_contato) AS mes, COUNT(*) AS total
FROM suporte WHERE data_contato >= date('now', '-12 months')
GROUP BY canal, mes ORDER BY mes, canal;
```
```
canal     mes       total
Chat      2025-07       9
E-mail    2025-07       5
Telefone  2025-07       7
...
```
→ Visualização: **gráfico de linha** (intenção: tendência)

> **Nota sobre case sensitivity:** o campo `canal` em `compras` usa capitalização (`'App'`, `'Site'`, `'Loja Física'`). O agente gera `LOWER(canal) = 'app'` automaticamente quando necessário — um exemplo concreto do loop de erro e correção funcionando.

---

## Observabilidade

O LangFuse está conectado ao `graph.invoke()` via callbacks opcionais. Cada run produz:

- Uma trace por query com spans por nó
- Contagem de tokens e latência por nó
- Session IDs tagueados como `{banker_id}:{provider}` (Aba 1) ou `benchmark:{provider}` (Aba 2)

Se as chaves do LangFuse não estiverem configuradas, o agente roda normalmente — a integração degrada graciosamente para `None`.

O uso por banker também é logado localmente em `data/logs/<banker_id>.jsonl` com estimativas de custo, classificação de intenção e contagem de retries. Isso alimenta o painel de métricas na sidebar e foi desenhado para uma trajetória de upgrade Parquet → DuckDB conforme o volume crescer.

---

## Escalabilidade para Produção

Esta demo roda em SQLite + tier gratuito do Google AI Studio. O que muda na escala da Franq:

| Camada | Demo | Produção |
|---|---|---|
| Banco de dados | SQLite (arquivo local) | Cloud SQL (PostgreSQL) ou BigQuery |
| LLM | Google AI Studio free tier | Vertex AI Gemini ou Azure OpenAI com SLAs |
| Autenticação | Nenhuma | IAM + JWT por banker |
| Logging | JSONL → disco local | Parquet → GCS → BigQuery |
| Deploy | Streamlit Community Cloud | Cloud Run (containerizado) |
| Segredos | Arquivo `.env` | Secret Manager |
| Descoberta de schema | PRAGMA no startup | Catálogo de metadados com refresh |

O código do agente em si não muda entre ambientes. Seleção de provedor, conexão com banco e destino de logging são configuração — não lógica.

### Como ficaria um deploy em produção

```
Personal Banker (browser)
        │
        ▼
   Cloud Run (Streamlit)
        │
        ├──▶ Vertex AI Gemini (nós pesados)
        ├──▶ Vertex AI Gemini Flash (nós leves)
        │
        ├──▶ Cloud SQL (execução de queries)
        │
        ├──▶ LangFuse (observabilidade)
        │
        └──▶ GCS / BigQuery (logs de uso → analytics por banker)
```

---

## Limitações Conhecidas

**Memória multi-turn não implementada.** Cada pergunta começa do zero — o agente não tem contexto das perguntas anteriores na mesma sessão. Perguntas de follow-up como "mostre o mesmo só para São Paulo" exigem que a pergunta completa seja refeita.

**Qualidade dependente do schema.** A qualidade do SQL degrada se os nomes das colunas forem ambíguos ou se a pergunta exigir conhecimento de domínio não presente no schema (ex.: "clientes de alto valor" não tem definição no banco).

**Quota do Gemini no tier gratuito.** O Google AI Studio limita o uso gratuito a 40 requisições/dia (com rotação de duas chaves). Os modelos Azure Foundry não têm essa limitação.

**Latência do Kimi K2.5.** O Azure Foundry adiciona overhead de roteamento para modelos de terceiros. O Kimi consistentemente leva 45-55 segundos — acima do timeout de benchmark de 45s. Não é um problema de código; é uma restrição de infraestrutura.

**Sem guardrails contra SQL injection.** O SQL gerado é executado diretamente no banco. Em produção, um usuário de banco read-only e uma allowlist de queries seriam necessários.

---

## Roadmap

**Deploy 1 — Memória multi-turn**
Adicionar `ConversationBufferMemory` ou `RunnableWithMessageHistory` para bankers refinarem queries conversacionalmente. "Mostre o mesmo para SP" deve funcionar.

**Deploy 2 — Hardening para produção**
Usuário DB read-only, timeout de query, cap de linhas no resultado, redação de PII nos logs, container Cloud Run.

**Deploy 3 — Analytics por banker**
Agregar todos os `data/logs/*.jsonl` → Parquet → DuckDB. Surfaçar padrões: quais perguntas são mais feitas, quais mais falham, quais custam mais. Retroalimentar melhorias nos prompts.

---

## Desafio 2 — Arquitetura (para discussão na entrevista)

O segundo desafio (pipeline de classificação e extração de documentos PDF) não foi implementado, conforme instrução do enunciado. A arquitetura proposta está documentada em [`docs/desafio2_arquitetura.md`](docs/desafio2_arquitetura.md).

O design reutiliza os mesmos padrões do Desafio 1: LangGraph para orquestração, Pydantic como contrato de saída, LLM barato para classificação e LLM médio para extração — com estimativa de ~87% de economia versus soluções com modelos pesados.

---

## Estrutura do Projeto

```
personal_banker_copilot/
├── agent/
│   ├── graph.py                   # StateGraph LangGraph + roteamento
│   ├── state.py                   # AgentState TypedDict
│   └── nodes/
│       ├── schema_inspector.py    # Nó 1 — PRAGMA, sem LLM
│       ├── planner.py             # Nó 2 — LLM pesado
│       ├── sql_generator.py       # Nó 3 — LLM pesado
│       ├── sql_executor.py        # Nó 4 — sem LLM
│       ├── error_recovery.py      # Nó 5 — LLM leve
│       ├── visualization_agent.py # Nó 6 — roteamento híbrido
│       └── response_formatter.py  # Nó 7 — LLM leve
├── utils/
│   ├── llm.py                     # Factory agnóstica de provedor (7 provedores)
│   ├── llm_output.py              # extract_text() — trata content blocks do LangChain
│   ├── database.py                # Descoberta dinâmica de schema
│   └── logger.py                  # Logging JSONL por banker + estimativas de custo
├── data/
│   └── franq.db                   # Banco SQLite do desafio
├── docs/
│   └── architecture.png           # Diagrama de fluxo LangGraph
├── devlog/
│   ├── p1.md                      # Log de construção — Dias 1 e 2
│   └── p2.md                      # Log de correções e benchmark — Dias 2 e 3
├── app.py                         # UI Streamlit (abas Deploy Único + Comparação de LLMs)
├── pyproject.toml                 # Dependências compatíveis com uv
├── .env.example                   # Placeholders de chaves — copie para .env
└── .gitignore
```

---

## Stack Tecnológica

| Camada | Escolha | Motivo |
|---|---|---|
| Orquestração do agente | LangGraph | Grafo modela retries, incerteza e trace naturalmente |
| LLM (pesado) | Gemini 2.5 Flash / GPT-4.1 | Forte raciocínio SQL, configurável |
| LLM (leve) | Gemini 2.0 Flash Lite / GPT-5.4 Nano | Custo-eficiente para formatação e viz |
| UI | Streamlit | Iteração rápida, deploy gratuito em cloud |
| Banco de dados | SQLite | Zero infra, compatível com os dados do desafio |
| Observabilidade | LangFuse | Spans por nó, contagem de tokens, latência |
| Gerenciador de pacotes | uv | 10x mais rápido que pip |
| Paralelismo | ThreadPoolExecutor | Benchmark roda todos os modelos simultaneamente |
