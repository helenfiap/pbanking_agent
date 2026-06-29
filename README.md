# Personal Banker Copilot
### Agentic Text-to-SQL Platform built with LangGraph

> O projeto implementa um workflow agêntico capaz de responder perguntas em linguagem natural sobre um banco de dados relacional, considerando práticas modernas de AI Engineering:
- **Orquestração com LangGraph** com estado tipado e roteamento condicional
- **Design modular** com 7 nós independentes e testáveis
- **Visualização automática** com classificação determinística de intenção
- **Benchmark multi-LLM** com 2 provedores e 6 LLMs lado a lado
- **Observabilidade** com rastreamento de latência por nó e integração com LangFuse
- **Estimativa de custo** por query e por LLM
- **Estratégia de retry** com recuperação autônoma de erros
- **Arquitetura orientada à produção** com caminho claro de migração para Cloud Run

![Arquitetura](docs/architecture.png)

---

## O Problema

Para operar um marketplace com mais de 150 produtos financeiros, Personal Bankers precisam entender o comportamento dos clientes: padrões de compra, respostas a campanhas, tendências de reclamações para fazer o match certo com cada produto. Hoje isso significa escrever SQL manualmente ou esperar pelo time de BI. Este copilot elimina os dois gargalos: o banker digita uma pergunta em português e recebe uma resposta estruturada com gráfico, o SQL que a gerou e o raciocínio por trás.

---

## Como Executar

```bash
git clone <repo>
cd personal_banker_copilot
uv sync
cp .env.example .env   # preencha as chaves, veja abaixo o guia
uv run streamlit run app.py
```

Requer Python 3.11+. Altamente recomendado utilizar [uv](https://github.com/astral-sh/uv) para gerenciamento de dependências e o run, para facilitar e agilizar o fluxo.

### Configuração de chaves

**Provedor 1 : Gemini (Google AI Studio, grátis)**
Gere sua chave em [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) e adicione ao `.env`:
```
GOOGLE_API_KEY=sua_chave_aqui
GEMINI_MODEL=gemini-2.5-flash   # 20 req/dia free; use gemini-2.0-flash-lite (1.500 req/dia) para dev
```

**Provedor 2 : Azure AI Foundry (GPT-4o, GPT-4.1, Kimi, Grok, Nano)**
Acesse seu recurso no [Azure AI Foundry](https://ai.azure.com) e copie o endpoint e a chave:
```
AZURE_OPENAI_ENDPOINT=https://<seu-recurso>.services.ai.azure.com/openai/v1
AZURE_OPENAI_API_KEY=sua_chave_aqui
```

Os dois provedores são opcionais, o app roda só com Gemini ou só com Azure. A aba de benchmark usa todos os provedores configurados em paralelo, mas para benchmark rodam em série.

---

## Arquitetura

### Atual

```
Usuário (browser)
       │
       ▼
  Streamlit UI
       │
       ▼
 LangGraph StateGraph
  ├── 1 Schema Inspector   ← PRAGMA, sem LLM, sempre atualizado
  ├── 2 Planner            ← LLM pesado, raciocina antes do SQL
  ├── 3 SQL Generator      ← LLM pesado, regras específicas de SQLite
  ├── 4 SQL Executor       ← Sem LLM, guarda de segurança + limpeza de colunas
  ├── 5 Error Recovery     ← LLM leve, retry com contexto do erro
  ├── 6 Visualization      ← Híbrido: regras por palavras-chave + LLM leve
  └── 7 Response Formatter ← LLM leve, narrativa em PT-BR
       │
       ▼
    SQLite
```

### Caminho para Produção

```
React / Next.js
       │
       ▼
   FastAPI  ──  POST /query
       │
       ▼
 LangGraph (mesmos nós, zero refatoração)
       │
       ├──▶ Cloud SQL (PostgreSQL) ou BigQuery
       ├──▶ Vertex AI Gemini / Azure OpenAI
       ├──▶ LangFuse (observabilidade)
       └──▶ GCS / BigQuery (logs de uso → analytics)
       │
       ▼
  Cloud Run (containerizado, protegido por IAM)
```

O código do agente não muda entre os ambientes. Seleção de provedor, conexão com banco e destino de logs são configuração, não lógica.

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

Em vez de uma chain monolítica de LLM, o projeto modela o workflow como um grafo de execução explícito.

```python
class AgentState(TypedDict):
    question: str       # pergunta original do usuário
    schema:   str       # output vivo do PRAGMA
    plan:     str       # raciocínio do planner
    sql:      str       # SQL atual (mutado pelo error recovery)
    result:   str       # JSON com as linhas da query
    error:    str       # última exceção do SQLite
    retries:  int       # contador de retries
    viz_spec: dict      # {type, x, y, color, orientation, title}
    response: str       # resposta final em PT-BR
    trace:    list[str] # passos por nó com latência
```

Vantagens em relação a uma chain:
- **Nós modulares**: cada nó é testado de forma independente
- **Estado explícito**: cada transição é inspecionável
- **Roteamento condicional**: o loop de retry é uma aresta real no grafo, não um `for` dentro de um monólito
- **Observabilidade**: spans do LangFuse mapeiam 1:1 com os nós
- **Facilidade de teste**: mocka um nó sem tocar nos outros
---
### Observabilidade

O LangFuse está conectado ao `graph.invoke()` via callbacks opcionais. Cada run produz:

- Uma trace por query com spans por nó
- Contagem de tokens e latência por nó
- Session IDs tagueados como `{banker_id}:{provider}` (Aba 1) ou `benchmark:{provider}` (Aba 2)

Se as chaves do LangFuse não estiverem configuradas, o agente roda normalmente — a integração degrada graciosamente para `None`.

O uso por banker também é logado localmente em `data/logs/<banker_id>.jsonl` com estimativas de custo, classificação de intenção e contagem de retries. Isso alimenta o painel de métricas na sidebar e foi desenhado para uma trajetória de upgrade Parquet → DuckDB conforme o volume crescer.

---
### Por que um nó Planejador antes do gerador de SQL?

Perguntas complexas com múltiplos joins falham na geração de SQL sem raciocínio prévio. O planejador força o LLM pesado a identificar quais tabelas são necessárias e por quê antes de escrever qualquer SQL. Nos testes, isso reduziu significativamente nomes de colunas alucinados e JOINs faltando.

### Por que roteamento híbrido de visualização?

Chamar um LLM para decidir "isso deve ser tabela ou gráfico?" custa tokens em cada query. O agente de visualização usa três camadas ordenadas por custo:

Os gráficos são selecionados por um pipeline de três camadas ordenadas por custo:

1. **Filtros determinísticos** (sem LLM): escalar único → `métrica`, linha única → `tabela`
2. **Classificador de intenção por palavras-chave** (sem LLM): detecta `tendência / ranking / contagem / comparação` e mapeia para tipo de gráfico e orientação — cobre ~80% das queries
3. **LLM leve** (apenas casos ambíguos): recebe dica de intenção, tipos de coluna e amostra de dados

Para queries de série temporal, o agente de viz detecta colunas de data e usa `color=coluna_categoria` para gerar gráficos de linha multi-série automaticamente. Se um modelo retorna dados sem dimensão temporal para uma pergunta de tendência, o agente aplica fallback para gráfico de barras horizontal.
~80% das queries nunca chegam à chamada LLM. Isso importa em escala.




## Por que uma aba de LLM benchmark?

Inicialmente o desafio pedia um deploy com modelo único. Durante o desenvolvimento ficou claro que escolher o LLM certo é parte do problema de engenharia em si. Em vez de fazer essa análise offline em um notebook, expus como ferramenta dentro do app, na aba **LLMs Comparison**:

- Todos os modelos rodam  via `ThreadPoolExecutor`
- Resultados aparecem ao vivo conforme cada modelo termina
- Cada run é anexado em `data/llms_comparison.csv` para comparação histórica

O benchmark é uma **ferramenta de desenvolvimento e operações**, não uma feature para o usuário final. Ele responde à pergunta "qual modelo devemos deployar em produção?" com dados reais da carga de trabalho real.
### Análise de Custo

Resultados de execuções reais no benchmark (pergunta sobre efetividade de campanhas):

| Modelo | Latência | Custo estimado/query | Observação |
|---|---|---|---|
| GPT-5.4 Nano | ~9s | **$0,00011** | Melhor relação custo/qualidade |
| Gemini 2.0 Flash Lite | ~5s | $0,00010 | 1.500 req/dia grátis — ideal para desenvolvimento |
| DeepSeek V3.2 | ~9s | ~$0,00000 | Custo praticamente zero em contextos curtos |
| GPT-4.1 | ~8s | $0,00037 | **Ponto ideal** — metade do custo do GPT-4o |
| Grok 4.1 Fast | ~8s | $0,00022 | Bom na formatação de percentuais |
| GPT-4o | ~10s | $0,00056 | Narrativa mais completa |
| Gemini 2.5 Flash | ~9s | $0,00030 | Limite de 20 req/dia no tier gratuito |

>**Recomendação para produção:** GPT-4.1 como padrão (equilíbrio custo/qualidade). GPT-5.4 Nano para tooling interno de alto volume e desenvolvimento. DeepSeek V3.2 se mostrando promissor para mais considerações.
---

### Notas de Avaliação

Resultados para as 5 perguntas do enunciado nos modelos selecionados:

| Pergunta | Modelo | Resultado | Latência | Viz | Observação |
|---|---|---|---|---|---|
| Q1 — Top 5 estados via app em maio | GPT-4.1 | ✅ | ~8s | Barras horizontal | Correto; case-sensitivity resolvida via enriquecimento do schema |
| Q2 — Campanhas WhatsApp 2024 | Todos | ✅ | 7–11s | Métrica escalar | Unânime: 17 clientes |
| Q3 — Categorias por média de compras | GPT-4.1 | ✅ | ~9s | Barras horizontal | Correto com alias `media_compras` |
| Q4 — Reclamações não resolvidas por canal | Todos | ✅ | 7–13s | Barras horizontal | Unânime: Chat > Telefone > E-mail |
| Q5 — Tendência de reclamações por canal | GPT-4.1 | ✅ | ~10s | Linha multi-série | 3 séries coloridas após correção do `color` no `px.line()` |
| Q5 — Tendência de reclamações por canal | GPT-4o | ⚠️ | ~10s | Barras horizontal | SQL sem dimensão temporal; agente de viz aplicou fallback para barra |
| Q5 — Tendência de reclamações por canal | DeepSeek V3.2 | ✅ | ~9s | Linha multi-série | Melhor custo/resultado para esta pergunta |

>**Padrão de falha conhecido:** GPT-4o agrupa Q5 apenas por canal (sem eixo temporal). O agente de viz detecta eixo X sem data e aplica fallback para barra — comportamento correto, mas idealmente o Planner forçaria `GROUP BY mes, canal` para perguntas de tendência.

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
### Limitações Conhecidas

A plataforma foca intencionalmente em demonstrar conceitos de AI Engineering, não em deployment de produção.

| Limitação | Detalhe |
|---|---|
| Frontend local em Streamlit | Sem containerização, sem reverse proxy, sem autenticação real |
| Sem camada FastAPI | O agente é chamado diretamente do Streamlit; produção exporia `POST /query` |
| SQLite | Banco de arquivo único, sem concorrência real. Migração para Cloud SQL é configuração, não código |
| Guarda de SQL básica | Bloqueia DDL/DML mas não é um sandbox completo. Produção: usuário read-only + timeout de query |
| Custos estimados | Contagem de tokens é aproximada (palavras × 1,3). Custos reais dependem do tokenizador de cada modelo |
| LangFuse opcional | Sem as chaves, zero visibilidade em nível de span em produção |
| Quota do tier gratuito Gemini | 20 req/dia por chave (Gemini 2.5 Flash). Produção: Vertex AI ou Azure OpenAI |
| Sem memória conversacional | Cada pergunta começa do zero. Perguntas de acompanhamento exigem reformulação completa |

---

## Roadmap

>**v2 — Memória conversacional**
Adicionar `RunnableWithMessageHistory` para que assessores possam refiná-la de forma conversacional. "Mostre o mesmo para SP" deve funcionar.

>**v3 — Hardening para produção**
Usuário read-only no banco, timeout de query, cap de linhas no resultado, anonimização de PII nos logs, container no Cloud Run.

>**v4 — Camada de analytics**
Agregar `data/logs/*.jsonl` → Parquet → DuckDB. Surfacar padrões: perguntas mais frequentes, maior taxa de falha, maior custo. Alimentar melhorias de volta nos prompts.
---
## Bugs Encontrados e Corrigidos por Componente

Um registro honesto dos problemas reais encontrados durante o desenvolvimento — cada item é um exemplo concreto do ciclo de observar → diagnosticar → corrigir que o agente também executa internamente.

### `utils/llm.py` — Factory de LLM

| Problema | Causa | Correção |
|---|---|---|
| `StopIteration` em todos os nós com Gemini | `itertools.cycle` singleton criava 1 iterator global; `with_fallbacks` instanciava o LLM 2x por chamada, esgotando o ciclo de chaves | Removido singleton e `with_fallbacks`; trocado por `_key_counter` inteiro com lock |
| Erro 401 UNAUTHENTICATED após rotação | Nome de modelo `gemini-3.5-flash` (inexistente) mascarava auth error válido | Corrigido para `gemini-2.5-flash`; `with_fallbacks` removido para expor erros reais |
| Gemini com quota esgotada em poucas queries | 2 instâncias por `get_llm()` call × 7 nós = 14 requisições por pergunta | Instância única por call; quota real: 20 req/dia por chave |

### `utils/database.py` — Schema Inspector

| Problema | Causa | Correção |
|---|---|---|
| Queries retornando 0 linhas para canal | LLM gerava `canal = 'app'` mas DB armazena `'App'` — SQLite é case-sensitive em strings | Schema enriquecido com valores exatos: `canal — valores exatos: ['App', 'Loja Física', 'Site']` |
| "último ano" retornando quase nada | `date('now')` retorna 2026, DB vai até 2025 | SQL prompt proibido de usar `date('now')` para dados históricos; range real exposto no schema |

### `agent/nodes/sql_executor.py` — Executor SQL

| Problema | Causa | Correção |
|---|---|---|
| Colunas com nomes como `COUNT(DISTINCT cm.cliente_id)` na UI | LLM esquecia de usar `AS alias` em funções de agregação | `_clean_columns()` pós-processador: renomeia qualquer coluna que comece com `COUNT(`, `SUM(`, `AVG(` etc. antes de chegar no frontend |

### `agent/nodes/visualization_agent.py` — Agente de Visualização

| Problema | Causa | Correção |
|---|---|---|
| Perguntas de tendência retornando tabela | Checagem `n_rows > 15 → table` ocorria **antes** do classificador de intent; série temporal tem ~36 linhas (12 meses × 3 canais) | Bloco de tendência movido para antes do limite de linhas; line chart retornado para até 200 linhas |
| Gráfico de linha sem separação por canal | `px.line()` não recebia o parâmetro `color` apesar de estar no `viz_spec` | Adicionado `color=color` na chamada `px.line()` — 1 linha de código, 3 horas de debug |
| GPT-4o desenhando linha sobre eixo categórico | Modelo gerou SQL sem dimensão temporal (GROUP BY canal apenas) → viz agent tentou linha sobre 3 categorias | Guarda-chuva: se `x` não contém padrão de data/mês/ano → converte automaticamente para barra horizontal |

### `app.py` — Interface Streamlit

| Problema | Causa | Correção |
|---|---|---|
| `StreamlitDuplicateElementId` no benchmark | 7 modelos gerando `plotly_chart` com mesmo ID auto-gerado (mesmo tipo + mesmo título) | Parâmetro `key=f"viz_{type}_{provider}"` em todos os `st.plotly_chart` |
| `ArrowInvalid` crash na tabela de resumo | Coluna mista com `float` e string `"—"` confunde o encoder PyArrow do Streamlit | `.astype(str)` no DataFrame antes de `st.dataframe` |
| Perguntas clicáveis não preenchendo input | `st.text_input(value=session_state.pop(...))` conflita com widget keyed no Streamlit | Setado `st.session_state["cmp_question"]` diretamente antes do widget |
| Benchmark com 7 colunas muito estreito | `st.columns(len(results))` com 7 modelos → colunas ilegíveis | Grid quebrado em linhas de max 3 colunas |
| Arquivo truncado com byte inválido `0xe2` | Ferramentas de edição de texto cortam arquivos com caracteres UTF-8 multibyte (como `—`) | Todas as reescritas completas passaram a usar `cat > file << 'EOF'` via bash |



---

## Escalabilidade para Produção

Esta demo roda em SQLite + tier gratuito do Google AI Studio. O que muda na escala:

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






