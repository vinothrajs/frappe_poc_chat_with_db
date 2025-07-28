import os
import json
import uuid
import backoff
import pandas as pd
from sqlalchemy import create_engine, text
from pymysql.err import OperationalError
from openai import RateLimitError

from typing import Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Configuration
os.environ["OPENAI_API_KEY"] =  "sk-proj----"
DB_URI = "mysql+pymysql://root:admin@localhost:3306/_35f6ff329580de8b"
PERSIST_DIR = "./frappe_chroma_db"
COLLECTION_NAME = "frappe_schema_full"

frappe_tables = [
    "tabPatient", "tabPatient Encounter", "tabPatient Appointment",
    "tabPatient Medical Record", "tabPatient Assessment Parameter",
    "tabPatch Log", "tabPatient Assessment Sheet", "tabPatient Assessment Template",
    "tabPatient Assessment", "tabPatient Encounter Diagnosis", "tabPatient Assessment Detail",
    "tabPatient Encounter Symptom", "tabPatient History Custom Document Type",
    "tabPatient Care Type", "tabPatient History Standard Document Type"
]

class FrappeSchemaMemory:
    def __init__(self, persist_dir="./frappe_chroma_db", collection_name=COLLECTION_NAME):
        self.embeddings = OpenAIEmbeddings()
        self.collection = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_dir
        )

    def _make_id(self, content: str, suffix: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, content)) + f"-{suffix}"

    def add_sql(self, question: str, sql: str):
        payload = json.dumps({"question": question, "sql": sql}, ensure_ascii=False)
        doc = Document(page_content=payload, metadata={"type": "sql"})
        self.collection.add_documents([doc])
        return self._make_id(payload, "sql")

    def add_ddl(self, ddl: str):
        doc = Document(page_content=ddl, metadata={"type": "ddl"})
        self.collection.add_documents([doc])
        return self._make_id(ddl, "ddl")

    def add_doc(self, doc_text: str):
        doc = Document(page_content=doc_text, metadata={"type": "doc"})
        self.collection.add_documents([doc])
        return self._make_id(doc_text, "doc")

    def add_relation(self, relation_text: str, relation_type: str):
        doc = Document(
            page_content=relation_text,
            metadata={"type": "relation", "relation_type": relation_type}
        )
        self.collection.add_documents([doc])
        return self._make_id(relation_text, relation_type)

    def add_sample_data(self, table: str, sample_rows: list):
        if not sample_rows:
            print(f"⚠️ Skipping {table} (no sample data)")
            return None

        content = f"SAMPLE DATA from {table}:\n"
        for row in sample_rows:
            content += json.dumps(row, ensure_ascii=False, default=str) + "\n"

        doc = Document(
            page_content=content,
            metadata={"type": "sample_data", "table": table}
        )
        self.collection.add_documents([doc])
        return self._make_id(content, "sample")


    def retrieve(self, query: str, k=5, filter_type: Optional[str] = None):
        retriever = self.collection.as_retriever(search_kwargs={"k": k})
        if filter_type:
            retriever.search_kwargs["filter"] = {"type": filter_type}
        docs = retriever.get_relevant_documents(query)

        print("\n🔍 Retrieved Top-K Embeddings:")
        for idx, doc in enumerate(docs, 1):
            print(f"--- Document {idx} ---")
            print(f"Content:\n{doc.page_content}\n")
            print(f"Metadata: {doc.metadata}\n")

        return docs

    def get_all(self) -> pd.DataFrame:
        data = self.collection.get()
        if not data["ids"]:
            return pd.DataFrame()
        return pd.DataFrame({
            "id": data["ids"],
            "content": data["documents"],
            "metadata": data["metadatas"]
        })

    def remove(self, id: str):
        self.collection._collection.delete(ids=[id])

    def reset(self):
        self.collection.delete_collection()

    def as_retriever(self, k=5, filter_type=None):
        retriever = self.collection.as_retriever(search_kwargs={"k": k})
        if filter_type:
            retriever.search_kwargs["filter"] = {"type": filter_type}
        return retriever

# Auto-Training from Frappe DB
def train_from_frappe(memory: FrappeSchemaMemory):
    engine = create_engine(DB_URI)

    with engine.connect() as conn:
        for tbl in frappe_tables:
            # DDL
            ddl = conn.execute(text(f"SHOW CREATE TABLE `{tbl}`")).fetchone()
            if ddl:
                memory.add_ddl(f"DDL: {ddl[1]}")

            # Columns (optional as doc)
            cols = conn.execute(text(f"SHOW COLUMNS FROM `{tbl}`")).fetchall()
            col_list = ", ".join(c[0] for c in cols)
            memory.add_doc(f"{tbl}: columns: {col_list}")

            # Link relations
            doctype = tbl.replace("tab", "")
            links = conn.execute(
                text("SELECT fieldname, options FROM tabDocField WHERE parent=:dt AND fieldtype='Link'"),
                {"dt": doctype}
            ).fetchall()
            for fieldname, opt in links:
                rel = f"RELATION (Link): {tbl}.{fieldname} → tab{opt}.name"
                memory.add_relation(rel, "link")

            # Primary Key
            pkey_rel = f"{tbl}.name is PRIMARY KEY"
            memory.add_relation(pkey_rel, "pkey")

            # Foreign keys via Link (simplified mapping)
            for fieldname, opt in links:
                fkey = f"FOREIGN KEY: {tbl}.{fieldname} references tab{opt}.name"
                memory.add_relation(fkey, "fkey")

            # Sample Data
            rows = conn.execute(text(f"SELECT * FROM `{tbl}` LIMIT 5")).fetchall()
            if rows:
                row_dicts = [dict(row._mapping) for row in rows]
                memory.add_sample_data(tbl, row_dicts)


    print("✅ Frappe schema + relations trained into vector memory.")

# SQL Execution
def execute_sql(sql_query: str) -> str:
    engine = create_engine(DB_URI)
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql_query)).fetchall()
            if not result:
                return "✅ Query executed successfully, but no data found."
            return "\n".join(str(row) for row in result)
    except Exception as e:
        return f"❌ SQL Execution Error: {e}"

# Follow-up Suggestion
def suggest_follow_up(original_q: str, memory: FrappeSchemaMemory) -> list:
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an assistant skilled at asking relevant follow-up questions. "
         "Use the schema context below.\n{context}"),
        ("user",
         "User asked: \"{input}\". "
         "Suggest up to 3 follow-up questions. Present each on a new line.")
    ])

    llm = ChatOpenAI(temperature=0.3)

    combine_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    retrieval_chain = create_retrieval_chain(
        retriever=memory.as_retriever(k=3),
        combine_docs_chain=combine_chain
    )

    resp = retrieval_chain.invoke({
        "input": original_q,
        "context": ""
    })["answer"]

    return [line.strip(" -•") for line in resp.split("\n") if line.strip()]

# ask_data
@backoff.on_exception(backoff.expo, RateLimitError, max_time=60, max_tries=5)
def ask_data(question: str, memory: FrappeSchemaMemory):
    similar_docs = memory.retrieve(question, k=5)

    examples = ""
    for doc in similar_docs:
        examples += f"{doc.page_content}\n\n"

    prompt_template = (
        "You are a MySQL SQL generator for Frappe Healthcare.\n"
        "Here is relevant schema, DDL, docs, and relationships:\n\n"
        "{examples}\n"
        "User Question: {user_question}\n"
        "Generate a syntactically correct SQL query.\n"
        "SQL Query:"
    )

    prompt_text = prompt_template.format(examples=examples, user_question=question)

    llm = ChatOpenAI(temperature=0)
    response = llm.invoke(prompt_text)

    generated_sql = response.content.strip()
    print(f"\n📝 Generated SQL:\n{generated_sql}\n")

    sql_result = execute_sql(generated_sql)

    follow_ups = suggest_follow_up(question, memory)

    return sql_result, follow_ups

# CLI
if __name__ == "__main__":
    memory = FrappeSchemaMemory()

    print("[1] Train from Frappe DB [2] Ask Data [3] Reset Memory [4] View Memory")
    choice = input("Choose option: ").strip()

    if choice == "1":
        train_from_frappe(memory)

    elif choice == "2":
        q = input("Data question: ").strip()
        result, follow_ups = ask_data(q, memory)
        print("\n📊 SQL Execution Result:\n", result)
        if follow_ups:
            print("\n💡 Follow-up suggestions:")
            for f in follow_ups:
                print(" •", f)

    elif choice == "3":
        memory.reset()
        print("✅ Memory reset.")

    elif choice == "4":
        df = memory.get_all()
        print(df)

    else:
        print("❌ Invalid choice.")
