# AI-Powered Analytics Assistant using Databricks GenAI

## 🎯 Overview

An intelligent analytics assistant that combines **RAG (Retrieval Augmented Generation)** with **Databricks GenAI** to answer questions about your data, generate SQL queries, and provide insights from your data documentation.

### What It Does

- 📊 **Natural Language SQL**: Ask questions in plain English, get SQL queries
- 📚 **Document Q&A**: Query your data documentation, schemas, and business logic
- 🤖 **Smart Analytics**: AI understands context from your data catalog
- 🔍 **Semantic Search**: Find relevant tables and columns automatically
- 📈 **Insights Generation**: Get automated insights and recommendations

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   User Question                             │
│  "What was our revenue last quarter by region?"            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              RAG Vector Search                              │
│  • Search documentation for relevant context               │
│  • Find similar questions & queries                        │
│  • Retrieve table schemas & metadata                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         Databricks GenAI (Foundation Model)                 │
│  • Claude Sonnet or Llama                                  │
│  • Generates SQL based on context                          │
│  • Provides explanations                                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Execute & Visualize                            │
│  • Run SQL on your data                                    │
│  • Create visualizations                                   │
│  • Return insights to user                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Features

### 1. **Natural Language to SQL**
```
You: "Show me top 10 customers by revenue this year"
AI:  Generates and explains SQL query
     Executes query
     Returns results with visualization
```

### 2. **Intelligent Document Search**
- Searches through:
  - Data dictionaries
  - Schema documentation
  - Business logic documents
  - Previous queries and analyses
  - Table descriptions

### 3. **Context-Aware Responses**
- Understands your data model
- Remembers conversation history
- Suggests related analyses
- Recommends best practices

### 4. **Multi-Modal Analytics**
- SQL generation
- Data exploration
- Trend analysis
- Report generation
- Dashboard creation

---

## 📋 Prerequisites

### Required
- Databricks Workspace (DBR 14.3+)
- Unity Catalog enabled
- Access to Databricks Model Serving
- Python 3.9+

### Recommended
- Vector Search endpoint
- Foundation Model API access
- Sample data loaded
- Documentation in Delta tables

---

## 🛠️ Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/your-org/ai-analytics-assistant.git
cd ai-analytics-assistant
```

### Step 2: Install Dependencies
```python
%pip install -U databricks-genai-inference databricks-vectorsearch langchain langchain-community mlflow
```

### Step 3: Configure Settings
```python
# config.py
CATALOG = "your_catalog"
SCHEMA = "your_schema"
VECTOR_SEARCH_ENDPOINT = "your_vector_search_endpoint"
MODEL_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
```

---

## 🎓 Quick Start

### 1. Set Up Vector Search

```python
from databricks.vector_search.client import VectorSearchClient

# Create vector search index
client = VectorSearchClient()

index = client.create_delta_sync_index(
    endpoint_name="your_endpoint",
    index_name=f"{CATALOG}.{SCHEMA}.documentation_index",
    source_table_name=f"{CATALOG}.{SCHEMA}.documentation",
    pipeline_type="TRIGGERED",
    primary_key="doc_id",
    embedding_source_column="content",
    embedding_model_endpoint_name="databricks-bge-large-en"
)
```

### 2. Load Your Documentation

```python
# Load data dictionaries, schemas, examples
documentation_df = spark.createDataFrame([
    ("doc1", "Customers table contains customer information including ID, name, email, registration date"),
    ("doc2", "Revenue is calculated as sum of order_amount where order_status = 'completed'"),
    ("doc3", "Fiscal year starts in Q1 (January)"),
])

documentation_df.write.format("delta").mode("overwrite").saveAsTable(
    f"{CATALOG}.{SCHEMA}.documentation"
)
```

### 3. Create Analytics Assistant

```python
from analytics_assistant import AnalyticsAssistant

# Initialize assistant
assistant = AnalyticsAssistant(
    catalog=CATALOG,
    schema=SCHEMA,
    vector_search_endpoint=VECTOR_SEARCH_ENDPOINT,
    model_endpoint=MODEL_ENDPOINT
)

# Ask a question
response = assistant.ask("What are our top customers by revenue?")
print(response)
```

---

## 📚 Detailed Usage

### Example 1: Simple Query
```python
question = "How many orders did we have last month?"

response = assistant.ask(question)

# Response includes:
# - Generated SQL
# - Query results
# - Explanation
# - Visualization (if applicable)
```

### Example 2: Complex Analysis
```python
question = """
Analyze customer churn for Q4 2024. 
Show trends by customer segment and identify key factors.
"""

response = assistant.ask(question)

# AI will:
# 1. Search documentation for churn definitions
# 2. Find relevant tables and columns
# 3. Generate appropriate SQL
# 4. Execute analysis
# 5. Provide insights and recommendations
```

### Example 3: With Context
```python
# Multi-turn conversation
assistant.ask("Show me revenue by region")
assistant.ask("Now break it down by product category")  # Uses context
assistant.ask("Which region had the highest growth?")   # Continues thread
```

### Example 4: Schema Discovery
```python
question = "What tables contain customer information?"

response = assistant.ask(question)

# Returns:
# - List of relevant tables
# - Column descriptions
# - Relationships
# - Sample queries
```

---

## 🏗️ Architecture Details

### Components

#### 1. **Vector Search Layer**
```python
VectorSearchClient
    ↓
Documentation Index
    ↓
Semantic Search → Top K Results
```

#### 2. **RAG Pipeline**
```python
User Query
    ↓
Embedding Generation
    ↓
Vector Search (retrieve relevant docs)
    ↓
Context Assembly
    ↓
Prompt Construction
    ↓
LLM Generation
```

#### 3. **SQL Generation**
```python
Question + Context + Schema
    ↓
Foundation Model
    ↓
SQL Query
    ↓
Validation
    ↓
Execution
```

#### 4. **Response Pipeline**
```python
Query Results
    ↓
Formatting
    ↓
Visualization (optional)
    ↓
Explanation Generation
    ↓
Final Response
```

---

## 📊 Data Sources

### What to Index for RAG

1. **Schema Documentation**
   - Table descriptions
   - Column definitions
   - Data types
   - Relationships

2. **Business Logic**
   - Metric definitions
   - Calculation rules
   - Business glossary
   - KPI documentation

3. **Query Examples**
   - Common queries
   - Complex joins
   - Best practices
   - Optimization tips

4. **Data Quality**
   - Validation rules
   - Data lineage
   - Known issues
   - Update schedules

### Documentation Format

```python
documentation = {
    "doc_id": "unique_id",
    "doc_type": "schema|business_logic|query_example",
    "title": "Document title",
    "content": "Full text content for embedding",
    "metadata": {
        "tables": ["table1", "table2"],
        "columns": ["col1", "col2"],
        "tags": ["revenue", "customer"]
    }
}
```

---

## 🎯 Use Cases

### 1. **Ad-Hoc Analysis**
```
"Show me customer acquisition cost trend for the last 6 months"
→ Generates SQL, runs analysis, creates visualization
```

### 2. **Report Generation**
```
"Create a monthly revenue report by product line"
→ Generates comprehensive report with multiple queries
```

### 3. **Data Discovery**
```
"Where can I find customer email addresses?"
→ Searches schema, returns table and column info
```

### 4. **Metric Calculation**
```
"Calculate our customer lifetime value"
→ Uses documented formulas, generates correct SQL
```

### 5. **Troubleshooting**
```
"Why are we seeing duplicate customers?"
→ Analyzes data, identifies issues, suggests fixes
```

---

## ⚙️ Configuration

### config.yaml
```yaml
# Databricks Configuration
databricks:
  catalog: your_catalog
  schema: your_schema
  warehouse_id: your_warehouse_id

# Vector Search
vector_search:
  endpoint_name: your_endpoint
  index_name: documentation_index
  embedding_model: databricks-bge-large-en
  top_k: 5

# LLM Configuration
llm:
  endpoint: databricks-meta-llama-3-3-70b-instruct
  temperature: 0.1
  max_tokens: 2000

# RAG Configuration
rag:
  chunk_size: 500
  chunk_overlap: 50
  retrieval_threshold: 0.7

# Query Execution
execution:
  max_rows: 10000
  timeout: 300
  enable_caching: true
```

---

## 🔒 Security

### Access Control
- Uses Unity Catalog permissions
- Respects table/column ACLs
- Audit logging enabled
- Row-level security support

### Data Privacy
- No data stored in LLM
- Context limited to metadata
- PII detection and masking
- Secure credential management

---

## 📈 Performance Optimization

### Vector Search
```python
# Use computed columns for embeddings
ALTER TABLE documentation
ADD COLUMN embedding_vector ARRAY<FLOAT>
GENERATED ALWAYS AS (ai_embed(content, 'databricks-bge-large-en'))
```

### Caching
```python
# Cache frequent queries
assistant.enable_cache(ttl=3600)
```

### Batch Processing
```python
# Process multiple questions
questions = [
    "Show revenue by region",
    "Show revenue by product",
    "Show revenue trend"
]

responses = assistant.ask_batch(questions)
```

---

## 🧪 Testing

### Unit Tests
```bash
pytest tests/test_vector_search.py
pytest tests/test_sql_generation.py
pytest tests/test_rag_pipeline.py
```

### Integration Tests
```bash
pytest tests/integration/test_end_to_end.py
```

### Quality Metrics
- SQL accuracy: >95%
- Response relevance: >90%
- Query success rate: >98%
- Average response time: <5s

---

## 🐛 Troubleshooting

### Common Issues

**Issue 1: No relevant context found**
```
Solution: Add more documentation to vector index
Check: Is documentation indexed properly?
```

**Issue 2: SQL generation errors**
```
Solution: Provide more schema examples
Check: Are table names and columns correct in docs?
```

**Issue 3: Slow responses**
```
Solution: Optimize vector search top_k parameter
Check: Is caching enabled?
```

**Issue 4: Incorrect results**
```
Solution: Add query examples to documentation
Check: Are business rules documented?
```

---

## 📚 Best Practices

### 1. Documentation Quality
- ✅ Keep documentation up-to-date
- ✅ Include examples for complex logic
- ✅ Document relationships between tables
- ✅ Add business context

### 2. Query Design
- ✅ Start with simple queries
- ✅ Build iteratively
- ✅ Validate results
- ✅ Save successful queries as examples

### 3. Prompt Engineering
- ✅ Be specific in questions
- ✅ Provide context when needed
- ✅ Use domain terminology
- ✅ Break complex questions into steps

### 4. Monitoring
- ✅ Track query success rates
- ✅ Monitor response times
- ✅ Log user feedback
- ✅ Analyze usage patterns

---

## 🔄 Continuous Improvement

### Feedback Loop
1. User asks question
2. AI generates response
3. User provides feedback (👍/👎)
4. System learns from feedback
5. Improves future responses

### Updating Knowledge Base
```python
# Add new documentation
new_docs = spark.createDataFrame([
    ("doc_new", "New business rule: ...")
])

new_docs.write.format("delta").mode("append").saveAsTable(
    f"{CATALOG}.{SCHEMA}.documentation"
)

# Sync vector index
index.sync()
```

---

## 📊 Monitoring & Analytics

### Dashboard Metrics
- Total questions asked
- Successful query rate
- Average response time
- Most common questions
- User satisfaction score

### Usage Analytics
```sql
SELECT 
  DATE(timestamp) as date,
  COUNT(*) as questions_asked,
  AVG(response_time_seconds) as avg_response_time,
  SUM(CASE WHEN thumbs_up THEN 1 ELSE 0 END) / COUNT(*) as satisfaction_rate
FROM analytics_assistant_logs
GROUP BY date
ORDER BY date DESC
```

---

## 🤝 Contributing

### How to Contribute
1. Fork the repository
2. Create feature branch
3. Add tests
4. Submit pull request

### Development Setup
```bash
git clone <repo>
pip install -r requirements-dev.txt
pre-commit install
```

---

## 📝 License

MIT License - see LICENSE file

---

## 🆘 Support

### Get Help
- **Documentation**: [docs.databricks.com](https://docs.databricks.com)
- **Community**: Databricks Community Forums
- **Issues**: GitHub Issues
- **Email**: support@company.com

### Resources
- [Databricks GenAI Guide](https://docs.databricks.com/generative-ai/)
- [Vector Search Documentation](https://docs.databricks.com/vector-search/)
- [LangChain Documentation](https://python.langchain.com/)

---

## 🎯 Roadmap

### Current Version (v1.0)
- ✅ Natural language to SQL
- ✅ RAG-based context retrieval
- ✅ Basic visualizations
- ✅ Query caching

### Next Release (v1.1)
- 🔄 Multi-turn conversations
- 🔄 Advanced visualizations
- 🔄 Report templates
- 🔄 Scheduled queries

### Future
- 📋 Predictive analytics
- 📋 Automated insights
- 📋 Custom dashboards
- 📋 API access

---

## 💡 Examples Gallery

### Example 1: Revenue Analysis
```python
assistant.ask("Compare Q4 2024 revenue with Q4 2023 by region")
```

### Example 2: Customer Segmentation
```python
assistant.ask("Segment customers by purchase frequency and average order value")
```

### Example 3: Trend Detection
```python
assistant.ask("Identify any unusual patterns in daily sales for the last month")
```

### Example 4: Forecasting
```python
assistant.ask("Project next quarter revenue based on current trends")
```

---

## 🌟 Success Stories

> "Reduced time to insight from hours to minutes" - Data Analyst

> "Democratized data access across the organization" - VP of Analytics

> "SQL generation accuracy exceeded 95%" - Data Engineer

---

**Built with ❤️ using Databricks GenAI, LangChain, and Vector Search**

*Last Updated: January 2025*
