import frappe
from frappe import _
import os
from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseSequentialChain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

from sqlalchemy import create_engine, text

# Configuration
# OPENAI_API_KEY = frappe.conf.openai_api_key or os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = "sk-proj-WgMkvtPmd---"
DB_URI = frappe.conf.db_uri or "mysql+pymysql://root:admin@localhost:3306/" + frappe.conf.db_name
PERSIST_DIR = "./frappe_chroma_db"
VECTOR_COLLECTION = "frappe_schema"

frappe_tables = [
    "tabPatient", "tabPatient Encounter", "tabPatient Appointment",
    "tabPatient Medical Record", "tabPatient Assessment Parameter",
    "tabPatch Log", "tabPatient Assessment Sheet", "tabPatient Assessment Template",
    "tabPatient Assessment", "tabPatient Encounter Diagnosis", "tabPatient Assessment Detail",
    "tabPatient Encounter Symptom", "tabPatient History Custom Document Type",
    "tabPatient Care Type", "tabPatient History Standard Document Type"
]

@frappe.whitelist(allow_guest=True)
def ping():
    return {"result": "pong"}

@frappe.whitelist(allow_guest=True)
def embed_schema():
    engine = create_engine(DB_URI)
    docs = []

    with engine.connect() as conn:
        for tbl in frappe_tables:
            res = conn.execute(text(
                "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
                "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=:tbl"
            ), {"tbl": tbl}).fetchall()

            if not res: continue
            actual_tbl = res[0][0]

            create = conn.execute(text(f"SHOW CREATE TABLE `{actual_tbl}`")).fetchone()
            if create:
                docs.append(Document(page_content=f"DDL: {create[1]}", metadata={"table": actual_tbl}))

            cols = conn.execute(text(f"SHOW COLUMNS FROM `{actual_tbl}`")).fetchall()
            docs.append(Document(page_content=f"{actual_tbl}: columns: {', '.join(c[0] for c in cols)}", metadata={"table": actual_tbl}))

            links = conn.execute(text(
                "SELECT fieldname, options FROM tabDocField "
                "WHERE parent=:dt AND fieldtype='Link'"
            ), {"dt": actual_tbl.replace("tab", "")}).fetchall()

            for fn, opt in links:
                docs.append(Document(
                    page_content=f"RELATION: {actual_tbl}.{fn} → tab{opt}.name",
                    metadata={"table": actual_tbl}
                ))

    chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(docs)

    vectordb = Chroma(collection_name=VECTOR_COLLECTION, embedding_function=OpenAIEmbeddings(), persist_directory=PERSIST_DIR)
    vectordb.add_documents(chunks)

    return {"message": "✅ Frappe schema embedded successfully."}

@frappe.whitelist(allow_guest=True)
def ask_data(question):
    db = SQLDatabase.from_uri(DB_URI)

    sql_prompt = PromptTemplate(
        input_variables=["input", "table_info", "top_k", "dialect"],
        template=(
            "You are a MySQL expert. Generate a syntactically correct {dialect} query.\n"
            "- Wrap table/column names with spaces or special chars in backticks.\n"
            "- Use only the following tables:\n{table_info}\n"
            "- Limit results to {top_k} unless specified otherwise.\n"
            "- No INSERT, UPDATE, DELETE, DROP.\n\n"
            "User: {input}\nSQLQuery:"
        )
    )

    chain = SQLDatabaseSequentialChain.from_llm(
        llm=ChatOpenAI(temperature=0),
        db=db,
        query_prompt=sql_prompt,
        return_direct=True,
        top_k=5
    )

    result_dict = chain.invoke({"query": question})
    result = result_dict["result"] if isinstance(result_dict, dict) else str(result_dict)
    follow_ups = suggest_follow_up(question)

    return {"result": result, "follow_ups": follow_ups}

def suggest_follow_up(original_q: str) -> List[str]:
    vectordb = Chroma(VECTOR_COLLECTION, OpenAIEmbeddings(), PERSIST_DIR)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant suggesting relevant follow-up questions. Here is schema context:\n{context}"),
        ("user", "User asked: \"{input}\". Suggest up to 3 follow-up questions. One per line.")
    ])

    combine_chain = create_stuff_documents_chain(llm=ChatOpenAI(temperature=0.3), prompt=prompt)

    retrieval_chain = create_retrieval_chain(
        retriever=vectordb.as_retriever(k=3),
        combine_docs_chain=combine_chain
    )

    resp = retrieval_chain.invoke({
        "input": original_q,
        "context": ""
    })["answer"]

    return [line.strip(" -•") for line in resp.split("\n") if line.strip()]