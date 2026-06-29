# Desafio 2 — Arquitetura do Pipeline de Documentos

> Este documento descreve o desenho arquitetural para o Desafio Técnico 2 (não implementado).
> Será o foco de discussão na entrevista técnica presencial.

---

## Problema

Processar um backlog de milhares de PDFs heterogêneos (Notas Fiscais, Contratos, Relatórios de Manutenção),
classificá-los automaticamente e extrair campos estruturados para alimentar um ERP — com custo e latência
otimizados para escala de milhões de documentos.

---

## Arquitetura Proposta

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTÃO                                  │
│                                                                  │
│   data/raw/*.pdf  →  FileWatcher / GCS Trigger                  │
│                           │                                      │
│                    [Fila: Cloud Tasks / Pub/Sub]                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     PIPELINE LANGGRAPH                           │
│                                                                  │
│   ① Extração de Texto                                           │
│      pdfplumber (texto digital) → Gemini Vision (escaneado/OCR) │
│                    │                                             │
│   ② Classificação (Cheap LLM — gemini-2.0-flash-lite)          │
│      Prompt: "Classifique este documento em:                     │
│      nota_fiscal | contrato | relatorio_manutencao"             │
│      → Saída: tipo + score de confiança                         │
│                    │                                             │
│   ③ Roteamento (condicional)                                    │
│      nota_fiscal → ExtratorNF                                    │
│      contrato    → ExtratorContrato                              │
│      relatorio   → ExtratorRelatorio                             │
│      unknown     → FilaRevisaoHumana                            │
│                    │                                             │
│   ④ Extração Estruturada (Cheap LLM + Pydantic)                 │
│      Prompt com schema JSON estrito por tipo                     │
│      → Validação Pydantic antes de persistir                    │
│      → Em caso de falha: retry com prompt mais restritivo       │
│                    │                                             │
│   ⑤ Persistência                                                │
│      BigQuery (analytics) + PostgreSQL (ERP) + GCS (raw JSON)  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Schemas de Extração (Pydantic)

```python
class NotaFiscal(BaseModel):
    fornecedor: str
    cnpj: str
    data_emissao: date
    itens: list[ItemNF]           # descrição, qtd, valor_unit
    valor_total: Decimal

class ContratoServicos(BaseModel):
    contratante: str
    contratado: str
    objeto: str
    vigencia_inicio: date
    vigencia_fim: date
    valor_mensal: Decimal

class RelatorioManutencao(BaseModel):
    data: date
    tecnico: str
    equipamento: str
    problema: str
    solucao: str
```

---

## Decisões de Engenharia

### Classificação com LLM barato antes de extração com LLM pesado
Classificar é mais simples que extrair. Usar `gemini-2.0-flash-lite` ($0.10/1M tokens)
na classificação e só acionar LLM pesado nos casos ambíguos reduz custo em ~70% a escala.

### Pydantic como contrato de saída
O LLM retorna JSON; o Pydantic valida tipo, formato e campos obrigatórios.
Falha de validação → retry com prompt de correção (mesmo pattern do Desafio 1).
Isso garante que nenhum dado malformado chega ao ERP.

### Fila assíncrona para escala
Para milhões de documentos, processamento síncrono é inviável.
Cloud Tasks / Pub/Sub garante: backpressure, retry automático, dead-letter queue para anomalias,
e processamento paralelo com workers stateless (Cloud Run).

### Tratamento de documentos anômalos
- Score de confiança < threshold → rota para fila de revisão humana
- Extração falha após N retries → salva com status `erro` + razão
- Pipeline nunca para: um arquivo ruim não bloqueia os demais

---

## Stack

| Camada | Escolha |
|---|---|
| Orquestração | LangGraph (mesmo padrão do Desafio 1) |
| Extração de texto | pdfplumber (digital) + Gemini Vision (escaneado) |
| LLM classificação | Gemini 2.0 Flash Lite |
| LLM extração | Gemini 2.5 Flash |
| Validação | Pydantic v2 |
| Fila | Cloud Tasks (GCP) |
| Persistência | BigQuery + PostgreSQL + GCS |
| Deploy | Cloud Run (workers stateless) |
| Observabilidade | LangFuse + Cloud Monitoring |

---

## Estimativa de Custo (1M documentos/mês)

| Etapa | Modelo | Tokens est. | Custo est. |
|---|---|---|---|
| Classificação | gemini-2.0-flash-lite | ~500/doc | ~$50 |
| Extração | gemini-2.5-flash | ~2.000/doc | ~$600 |
| **Total LLM** | | | **~$650/mês** |

Comparado com extração manual ou modelos mais pesados (GPT-4o: ~$5.000/mês), a escolha de modelos
por camada (cheap para classificar, médio para extrair) representa ~87% de economia.
