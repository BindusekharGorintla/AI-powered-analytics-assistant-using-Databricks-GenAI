# Databricks notebook source
# MAGIC %md
# MAGIC # AI-Powered Analytics Assistant using Databricks GenAI + RAG
# MAGIC 
# MAGIC **Complete Implementation**
# MAGIC 
# MAGIC This notebook implements a RAG-based analytics assistant that:
# MAGIC - Answers questions in natural language
# MAGIC - Generates SQL queries automatically
# MAGIC - Retrieves relevant context from documentation
# MAGIC - Executes queries and returns insights
# MAGIC 
# MAGIC **Architecture:** Vector Search (RAG) + Foundation Models + SQL Execution

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Setup & Installation

# COMMAND ----------

# DBTITLE 1,Install Dependencies
%pip install -U -qqq databricks-genai-inference databricks-vectorsearch langchain langchain-community mlflow-skinny pandas

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import Libraries
from databricks.vector_search.client import VectorSearchClient
from databricks_genai_inference import ChatCompletion
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from delta.tables import DeltaTable
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import mlflow

# Initialize
spark = SparkSession.builder.getOrCreate()
mlflow.langchain.autolog()

print("✓ All libraries imported successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Configuration

# COMMAND ----------

# DBTITLE 1,Configuration Settings
# =============================================================================
# CONFIGURATION
# =============================================================================

# Databricks Configuration
CATALOG = "your_catalog"
SCHEMA = "your_schema"
WAREHOUSE_ID = "your_warehouse_id"

# Vector Search Configuration
VECTOR_SEARCH_CONFIG = {
    "endpoint_name": "your_vector_search_endpoint",
    "index_name": f"{CATALOG}.{SCHEMA}.analytics_docs_index",
    "embedding_model": "databricks-bge-large-en"
}

# LLM Configuration
LLM_CONFIG = {
    "endpoint": "databricks-meta-llama-3-3-70b-instruct",  # or databricks-dbrx-instruct
    "temperature": 0.1,
    "max_tokens": 2000
}

# RAG Configuration
RAG_CONFIG = {
    "top_k": 5,  # Number of documents to retrieve
    "relevance_threshold": 0.7,  # Minimum similarity score
    "chunk_size": 500,
    "chunk_overlap": 50
}

print("✓ Configuration loaded")
print(f"  Catalog: {CATALOG}")
print(f"  Schema: {SCHEMA}")
print(f"  LLM: {LLM_CONFIG['endpoint']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: Data Setup

# COMMAND ----------

# DBTITLE 1,Create Documentation Table
# =============================================================================
# STEP 1: Create table for storing documentation
# =============================================================================

# Create schema if it doesn't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# Create documentation table
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.analytics_documentation (
  doc_id STRING,
  doc_type STRING,  -- 'schema', 'business_logic', 'query_example', 'metric'
  title STRING,
  content STRING,
  metadata STRING,  -- JSON string
  tables ARRAY<STRING>,
  columns ARRAY<STRING>,
  tags ARRAY<STRING>,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
)
USING DELTA
""")

print(f"✓ Documentation table created: {CATALOG}.{SCHEMA}.analytics_documentation")

# COMMAND ----------

# DBTITLE 1,Load Sample Documentation
# =============================================================================
# STEP 2: Load sample documentation (replace with your actual docs)
# =============================================================================

sample_docs = [
    {
        "doc_id": "schema_customers",
        "doc_type": "schema",
        "title": "Customers Table Schema",
        "content": """
        Table: customers
        Description: Contains customer master data including contact information and registration details.
        
        Columns:
        - customer_id (BIGINT): Unique customer identifier, primary key
        - customer_name (STRING): Full name of the customer
        - email (STRING): Customer email address
        - phone (STRING): Contact phone number
        - registration_date (DATE): Date when customer registered
        - customer_segment (STRING): Business segment (Enterprise, SMB, Individual)
        - region (STRING): Geographic region (North, South, East, West)
        - is_active (BOOLEAN): Whether customer is currently active
        - lifetime_value (DOUBLE): Total revenue from customer
        """,
        "tables": ["customers"],
        "columns": ["customer_id", "customer_name", "email", "registration_date", "customer_segment", "region"],
        "tags": ["customer", "master_data", "crm"]
    },
    {
        "doc_id": "schema_orders",
        "doc_type": "schema",
        "title": "Orders Table Schema",
        "content": """
        Table: orders
        Description: Stores all customer orders and transactions.
        
        Columns:
        - order_id (BIGINT): Unique order identifier, primary key
        - customer_id (BIGINT): Foreign key to customers table
        - order_date (DATE): Date order was placed
        - order_amount (DOUBLE): Total order value in USD
        - order_status (STRING): Current status (pending, completed, cancelled)
        - product_category (STRING): Main product category
        - quantity (INT): Number of items
        - discount_applied (DOUBLE): Discount percentage applied
        - shipping_region (STRING): Destination region
        """,
        "tables": ["orders"],
        "columns": ["order_id", "customer_id", "order_date", "order_amount", "order_status"],
        "tags": ["orders", "transactions", "sales"]
    },
    {
        "doc_id": "metric_revenue",
        "doc_type": "metric",
        "title": "Revenue Calculation",
        "content": """
        Metric: Total Revenue
        Definition: Sum of all completed orders
        
        Calculation:
        SELECT SUM(order_amount) as revenue
        FROM orders
        WHERE order_status = 'completed'
        
        Business Rules:
        - Only completed orders count toward revenue
        - Cancelled orders are excluded
        - Refunds are handled separately in refunds table
        - Revenue is reported in USD
        """,
        "tables": ["orders"],
        "columns": ["order_amount", "order_status"],
        "tags": ["revenue", "kpi", "financial"]
    },
    {
        "doc_id": "query_top_customers",
        "doc_type": "query_example",
        "title": "Top Customers by Revenue",
        "content": """
        Query: Find top customers by revenue
        
        SQL Example:
        SELECT 
          c.customer_id,
          c.customer_name,
          c.customer_segment,
          SUM(o.order_amount) as total_revenue,
          COUNT(o.order_id) as order_count
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        WHERE o.order_status = 'completed'
        GROUP BY c.customer_id, c.customer_name, c.customer_segment
        ORDER BY total_revenue DESC
        LIMIT 10
        """,
        "tables": ["customers", "orders"],
        "columns": ["customer_id", "customer_name", "order_amount"],
        "tags": ["customer_analysis", "revenue", "top_n"]
    },
    {
        "doc_id": "business_fiscal_year",
        "doc_type": "business_logic",
        "title": "Fiscal Year Definition",
        "content": """
        Business Logic: Fiscal Year
        
        Our fiscal year starts on January 1st and ends on December 31st.
        
        Quarters:
        - Q1: January - March
        - Q2: April - June
        - Q3: July - September
        - Q4: October - December
        
        When calculating year-over-year metrics, compare same quarters.
        """,
        "tables": [],
        "columns": [],
        "tags": ["fiscal_year", "calendar", "business_rules"]
    },
    {
        "doc_id": "business_customer_churn",
        "doc_type": "business_logic",
        "title": "Customer Churn Definition",
        "content": """
        Business Logic: Customer Churn
        
        Definition: A customer is considered churned if they have not placed an order in the last 90 days.
        
        Calculation:
        - Active: Last order within 90 days
        - At Risk: Last order between 60-90 days
        - Churned: Last order more than 90 days ago
        
        Churn Rate = (Churned Customers / Total Customers) * 100
        """,
        "tables": ["customers", "orders"],
        "columns": ["customer_id", "order_date"],
        "tags": ["churn", "retention", "kpi"]
    }
]

# Convert to DataFrame
docs_data = []
for doc in sample_docs:
    docs_data.append({
        "doc_id": doc["doc_id"],
        "doc_type": doc["doc_type"],
        "title": doc["title"],
        "content": doc["content"],
        "metadata": json.dumps({"source": "sample"}),
        "tables": doc["tables"],
        "columns": doc["columns"],
        "tags": doc["tags"],
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    })

docs_df = spark.createDataFrame(docs_data)

# Write to Delta table
docs_df.write.format("delta").mode("overwrite").saveAsTable(
    f"{CATALOG}.{SCHEMA}.analytics_documentation"
)

print(f"✓ Loaded {len(sample_docs)} sample documents")
print(f"  - Schema docs: {sum(1 for d in sample_docs if d['doc_type'] == 'schema')}")
print(f"  - Business logic: {sum(1 for d in sample_docs if d['doc_type'] == 'business_logic')}")
print(f"  - Query examples: {sum(1 for d in sample_docs if d['doc_type'] == 'query_example')}")
print(f"  - Metrics: {sum(1 for d in sample_docs if d['doc_type'] == 'metric')}")

# COMMAND ----------

# DBTITLE 1,View Sample Documentation
# Display the documentation
display(spark.table(f"{CATALOG}.{SCHEMA}.analytics_documentation"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Vector Search Setup

# COMMAND ----------

# DBTITLE 1,Create Vector Search Index
# =============================================================================
# STEP 3: Create Vector Search Index
# =============================================================================

# Initialize Vector Search Client
vsc = VectorSearchClient()

# Create endpoint if it doesn't exist (run once)
try:
    vsc.create_endpoint(
        name=VECTOR_SEARCH_CONFIG["endpoint_name"],
        endpoint_type="STANDARD"
    )
    print(f"✓ Created endpoint: {VECTOR_SEARCH_CONFIG['endpoint_name']}")
except Exception as e:
    print(f"Endpoint may already exist: {e}")

# COMMAND ----------

# DBTITLE 1,Create Delta Sync Index
# Create the vector search index
try:
    index = vsc.create_delta_sync_index(
        endpoint_name=VECTOR_SEARCH_CONFIG["endpoint_name"],
        index_name=VECTOR_SEARCH_CONFIG["index_name"],
        source_table_name=f"{CATALOG}.{SCHEMA}.analytics_documentation",
        pipeline_type="TRIGGERED",
        primary_key="doc_id",
        embedding_source_column="content",
        embedding_model_endpoint_name=VECTOR_SEARCH_CONFIG["embedding_model"]
    )
    
    print(f"✓ Vector search index created: {VECTOR_SEARCH_CONFIG['index_name']}")
    print("  Waiting for index to sync...")
    
except Exception as e:
    print(f"Index may already exist: {e}")
    index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_CONFIG["endpoint_name"],
        index_name=VECTOR_SEARCH_CONFIG["index_name"]
    )

# Sync the index
index.sync()
print("✓ Index synced successfully")

# COMMAND ----------

# DBTITLE 1,Test Vector Search
# Test the vector search
test_query = "How do I calculate revenue?"

search_results = index.similarity_search(
    query_text=test_query,
    columns=["doc_id", "doc_type", "title", "content"],
    num_results=3
)

print(f"Test query: '{test_query}'")
print(f"\nTop 3 results:")
print(json.dumps(search_results, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 5: Analytics Assistant Implementation

# COMMAND ----------

# DBTITLE 1,Analytics Assistant Class
class AnalyticsAssistant:
    """
    AI-Powered Analytics Assistant using RAG + Databricks GenAI.
    """
    
    def __init__(
        self,
        catalog: str,
        schema: str,
        vector_search_client: VectorSearchClient,
        vector_index_name: str,
        llm_endpoint: str,
        vector_endpoint: str
    ):
        """Initialize the analytics assistant."""
        self.catalog = catalog
        self.schema = schema
        self.vsc = vector_search_client
        self.vector_index_name = vector_index_name
        self.vector_endpoint = vector_endpoint
        self.llm_endpoint = llm_endpoint
        self.conversation_history = []
        
        # Get vector search index
        self.index = self.vsc.get_index(
            endpoint_name=vector_endpoint,
            index_name=vector_index_name
        )
        
        print("✓ Analytics Assistant initialized")
    
    def retrieve_context(self, question: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant context from documentation using vector search.
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        try:
            # Search vector index
            results = self.index.similarity_search(
                query_text=question,
                columns=["doc_id", "doc_type", "title", "content", "tables", "tags"],
                num_results=top_k
            )
            
            # Extract results
            documents = []
            if results and 'result' in results and 'data_array' in results['result']:
                for item in results['result']['data_array']:
                    documents.append({
                        "doc_id": item[0],
                        "doc_type": item[1],
                        "title": item[2],
                        "content": item[3],
                        "tables": item[4] if len(item) > 4 else [],
                        "tags": item[5] if len(item) > 5 else []
                    })
            
            return documents
            
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []
    
    def generate_response(self, question: str, context: List[Dict]) -> str:
        """
        Generate response using LLM with retrieved context.
        
        Args:
            question: User's question
            context: Retrieved context documents
            
        Returns:
            AI-generated response
        """
        # Build context string
        context_text = "\n\n".join([
            f"Document: {doc['title']}\nType: {doc['doc_type']}\n{doc['content']}"
            for doc in context
        ])
        
        # Create prompt
        prompt = f"""You are an expert data analyst assistant. Use the following documentation to answer the user's question.

DOCUMENTATION:
{context_text}

USER QUESTION:
{question}

INSTRUCTIONS:
1. If the question asks for a SQL query, generate proper Databricks SQL
2. Use only the tables and columns mentioned in the documentation
3. Follow any business rules provided in the documentation
4. If you need more information, ask clarifying questions
5. Be concise but thorough
6. Include explanations with your SQL queries

ANSWER:"""
        
        # Call LLM
        try:
            response = ChatCompletion.create(
                model=self.llm_endpoint,
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_CONFIG["temperature"],
                max_tokens=LLM_CONFIG["max_tokens"]
            )
            
            answer = response.choices[0].message.content
            return answer
            
        except Exception as e:
            return f"Error generating response: {e}"
    
    def ask(self, question: str, top_k: int = 5, return_context: bool = False) -> Dict:
        """
        Ask a question to the analytics assistant.
        
        Args:
            question: User's question
            top_k: Number of context documents to retrieve
            return_context: Whether to return retrieved context
            
        Returns:
            Dictionary with answer and optionally context
        """
        print(f"\n{'='*80}")
        print(f"Question: {question}")
        print(f"{'='*80}\n")
        
        # Step 1: Retrieve context
        print("🔍 Retrieving relevant context...")
        context = self.retrieve_context(question, top_k)
        print(f"✓ Retrieved {len(context)} relevant documents")
        
        if context:
            print("\nRelevant documents:")
            for i, doc in enumerate(context, 1):
                print(f"  {i}. {doc['title']} ({doc['doc_type']})")
        
        # Step 2: Generate response
        print("\n🤖 Generating response...")
        answer = self.generate_response(question, context)
        print("✓ Response generated")
        
        # Store in conversation history
        self.conversation_history.append({
            "question": question,
            "answer": answer,
            "context_docs": [doc['doc_id'] for doc in context],
            "timestamp": datetime.now()
        })
        
        # Prepare result
        result = {
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }
        
        if return_context:
            result["context"] = context
        
        return result
    
    def execute_sql(self, sql_query: str, limit: int = 100) -> pd.DataFrame:
        """
        Execute a SQL query and return results.
        
        Args:
            sql_query: SQL query to execute
            limit: Maximum rows to return
            
        Returns:
            Pandas DataFrame with results
        """
        try:
            # Add LIMIT if not present
            if "LIMIT" not in sql_query.upper():
                sql_query = f"{sql_query.rstrip(';')} LIMIT {limit}"
            
            print(f"\n📊 Executing SQL query...")
            print(f"Query: {sql_query[:200]}...")
            
            # Execute query
            result_df = spark.sql(sql_query).toPandas()
            
            print(f"✓ Query executed successfully")
            print(f"  Rows returned: {len(result_df)}")
            
            return result_df
            
        except Exception as e:
            print(f"❌ Error executing query: {e}")
            return pd.DataFrame()
    
    def ask_and_execute(self, question: str) -> Dict:
        """
        Ask a question, get SQL, and execute it.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer, SQL, and results
        """
        # Get response
        response = self.ask(question)
        answer = response["answer"]
        
        # Try to extract and execute SQL
        if "SELECT" in answer.upper():
            # Extract SQL (simple extraction)
            sql_start = answer.upper().find("SELECT")
            sql_query = answer[sql_start:]
            
            # Clean up SQL (remove markdown, explanations after query)
            if "```" in sql_query:
                sql_query = sql_query.split("```")[0]
            
            sql_query = sql_query.split("\n\n")[0].strip()
            
            # Execute
            results = self.execute_sql(sql_query)
            
            return {
                "question": question,
                "answer": answer,
                "sql": sql_query,
                "results": results,
                "rows_returned": len(results)
            }
        
        return response
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history."""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("✓ Conversation history cleared")


print("✓ AnalyticsAssistant class defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 6: Initialize Assistant

# COMMAND ----------

# DBTITLE 1,Create Assistant Instance
# Initialize the analytics assistant
assistant = AnalyticsAssistant(
    catalog=CATALOG,
    schema=SCHEMA,
    vector_search_client=vsc,
    vector_index_name=VECTOR_SEARCH_CONFIG["index_name"],
    vector_endpoint=VECTOR_SEARCH_CONFIG["endpoint_name"],
    llm_endpoint=LLM_CONFIG["endpoint"]
)

print("\n" + "="*80)
print("🎉 Analytics Assistant Ready!")
print("="*80)
print("\nYou can now ask questions like:")
print("  • 'How do I calculate total revenue?'")
print("  • 'Show me the top 10 customers by revenue'")
print("  • 'What tables contain customer information?'")
print("  • 'How is churn defined in our business?'")
print("="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 7: Usage Examples

# COMMAND ----------

# DBTITLE 1,Example 1: Simple Question
# Ask a simple question
response = assistant.ask("How do I calculate total revenue?")

print("\n" + "="*80)
print("ANSWER:")
print("="*80)
print(response["answer"])

# COMMAND ----------

# DBTITLE 1,Example 2: SQL Generation
# Ask for SQL query
response = assistant.ask("Write a SQL query to find the top 10 customers by revenue")

print("\n" + "="*80)
print("ANSWER:")
print("="*80)
print(response["answer"])

# COMMAND ----------

# DBTITLE 1,Example 3: Schema Question
# Ask about schema
response = assistant.ask("What columns are in the customers table?")

print("\n" + "="*80)
print("ANSWER:")
print("="*80)
print(response["answer"])

# COMMAND ----------

# DBTITLE 1,Example 4: Business Logic
# Ask about business logic
response = assistant.ask("How is customer churn defined?")

print("\n" + "="*80)
print("ANSWER:")
print("="*80)
print(response["answer"])

# COMMAND ----------

# DBTITLE 1,Example 5: Complex Analysis
# Complex question
response = assistant.ask("""
Generate SQL to analyze revenue trends by region for the last 12 months.
Include month, region, and total revenue.
""")

print("\n" + "="*80)
print("ANSWER:")
print("="*80)
print(response["answer"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 8: Interactive Q&A

# COMMAND ----------

# DBTITLE 1,Interactive Question-Answer
# Function for interactive Q&A
def interactive_qa():
    """Interactive Q&A loop."""
    print("="*80)
    print("INTERACTIVE Q&A MODE")
    print("="*80)
    print("Type your questions below. Type 'quit' to exit.\n")
    
    while True:
        question = input("\n💬 Your question: ")
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not question.strip():
            continue
        
        # Get response
        response = assistant.ask(question)
        
        print("\n🤖 Answer:")
        print(response["answer"])
        print("\n" + "-"*80)

# Uncomment to run interactive mode:
# interactive_qa()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 9: Monitoring & Analytics

# COMMAND ----------

# DBTITLE 1,View Conversation History
# View conversation history
history = assistant.get_conversation_history()

if history:
    print(f"Conversation History ({len(history)} interactions):\n")
    
    for i, interaction in enumerate(history, 1):
        print(f"\n{'='*80}")
        print(f"Interaction {i}")
        print(f"{'='*80}")
        print(f"Time: {interaction['timestamp']}")
        print(f"\nQuestion: {interaction['question']}")
        print(f"\nAnswer: {interaction['answer'][:200]}...")
        print(f"\nContext docs: {', '.join(interaction['context_docs'])}")
else:
    print("No conversation history yet")

# COMMAND ----------

# DBTITLE 1,Usage Statistics
# Calculate statistics
if history:
    total_questions = len(history)
    unique_doc_types = set()
    doc_usage_count = {}
    
    for interaction in history:
        for doc_id in interaction['context_docs']:
            doc_usage_count[doc_id] = doc_usage_count.get(doc_id, 0) + 1
    
    print("="*80)
    print("USAGE STATISTICS")
    print("="*80)
    print(f"\nTotal questions asked: {total_questions}")
    print(f"Unique documents used: {len(doc_usage_count)}")
    
    print("\n\nMost frequently used documents:")
    sorted_docs = sorted(doc_usage_count.items(), key=lambda x: x[1], reverse=True)
    for doc_id, count in sorted_docs[:5]:
        print(f"  • {doc_id}: {count} times")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 10: Advanced Features

# COMMAND ----------

# DBTITLE 1,Add New Documentation
def add_documentation(doc_id: str, doc_type: str, title: str, content: str,
                     tables: List[str] = [], columns: List[str] = [],
                     tags: List[str] = []):
    """
    Add new documentation to the knowledge base.
    
    Args:
        doc_id: Unique document ID
        doc_type: Type of document
        title: Document title
        content: Document content
        tables: Related tables
        columns: Related columns
        tags: Tags for categorization
    """
    new_doc = spark.createDataFrame([{
        "doc_id": doc_id,
        "doc_type": doc_type,
        "title": title,
        "content": content,
        "metadata": json.dumps({"source": "user_added"}),
        "tables": tables,
        "columns": columns,
        "tags": tags,
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }])
    
    # Append to table
    new_doc.write.format("delta").mode("append").saveAsTable(
        f"{CATALOG}.{SCHEMA}.analytics_documentation"
    )
    
    # Sync vector index
    assistant.index.sync()
    
    print(f"✓ Added new documentation: {doc_id}")
    print("✓ Vector index synced")

# Example: Add new documentation
add_documentation(
    doc_id="metric_conversion_rate",
    doc_type="metric",
    title="Conversion Rate Calculation",
    content="""
    Metric: Conversion Rate
    Definition: Percentage of customers who made a purchase
    
    Calculation:
    (Number of customers with orders / Total number of customers) * 100
    
    Formula:
    SELECT 
      (COUNT(DISTINCT o.customer_id) * 100.0 / COUNT(DISTINCT c.customer_id)) as conversion_rate
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.order_status = 'completed'
    """,
    tables=["customers", "orders"],
    columns=["customer_id", "order_status"],
    tags=["conversion", "kpi", "metrics"]
)

# COMMAND ----------

# DBTITLE 1,Batch Questions
# Process multiple questions at once
questions = [
    "What is the conversion rate formula?",
    "How many columns are in the orders table?",
    "What is our fiscal year schedule?"
]

print("Processing batch questions...\n")

batch_results = []
for question in questions:
    result = assistant.ask(question)
    batch_results.append(result)
    print(f"✓ {question}")

print(f"\n✓ Processed {len(batch_results)} questions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary & Next Steps

# COMMAND ----------

displayHTML("""
<div style="font-family: Arial; padding: 20px; background-color: #f8f9fa; border-radius: 8px;">
    <h2 style="color: #2c3e50;">🎉 AI-Powered Analytics Assistant</h2>
    
    <h3>✅ What We've Built:</h3>
    <ul>
        <li>✅ RAG-based analytics assistant</li>
        <li>✅ Vector Search for context retrieval</li>
        <li>✅ Natural language to SQL generation</li>
        <li>✅ Documentation knowledge base</li>
        <li>✅ Conversation history tracking</li>
    </ul>
    
    <h3>🎯 Key Features:</h3>
    <ul>
        <li><strong>Semantic Search:</strong> Finds relevant documentation automatically</li>
        <li><strong>SQL Generation:</strong> Creates queries from natural language</li>
        <li><strong>Context-Aware:</strong> Uses your schema and business rules</li>
        <li><strong>Extensible:</strong> Easy to add new documentation</li>
    </ul>
    
    <h3>📚 How to Use:</h3>
    <pre style="background-color: #f4f4f4; padding: 15px; border-radius: 5px;">
# Ask a question
response = assistant.ask("How do I calculate revenue?")

# Get SQL and execute
result = assistant.ask_and_execute("Show top 10 customers")

# Add new documentation
add_documentation(doc_id, doc_type, title, content)
    </pre>
    
    <h3>🚀 Next Steps:</h3>
    <ol>
        <li>Add your actual documentation and schemas</li>
        <li>Create sample tables with real data</li>
        <li>Test with your business questions</li>
        <li>Deploy as a Streamlit app or API</li>
        <li>Add monitoring and feedback loop</li>
    </ol>
    
    <h3>📊 Architecture:</h3>
    <p style="background-color: #e3f2fd; padding: 15px; border-left: 4px solid #2196f3; margin: 15px 0;">
        <strong>User Question</strong> → <strong>Vector Search</strong> → <strong>Context Retrieval</strong> → 
        <strong>LLM (GenAI)</strong> → <strong>SQL Generation</strong> → <strong>Execution</strong> → <strong>Results</strong>
    </p>
    
    <p style="margin-top: 30px; padding: 15px; background-color: #d4edda; border-left: 4px solid #28a745;">
        <strong>✅ Your AI Analytics Assistant is ready to use!</strong><br>
        Start asking questions and let AI help you analyze your data.
    </p>
</div>
""")
