### Chat With Db

Connect to any db and connect with LLMA and ask question RAG

### Installation Pre Request

You can install this app using the [bench](https://github.com/frappe/bench) CLI:

### Dev Setup

bench --site demo.localhost install-app chat_with_db

bench pip install -r apps/chat_with_db/requirements.txt

bench --site demo.localhost uninstall-app chat_with_db

### configs - keep necessay in .env file

os.environ["OPENAI_API_KEY"] = "sk-proj-WgMkvtPmd---"

DB_URI =  "mysql+pymysql://root:admin@localhost:3306/_dbname" 

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

### API call

## Train the schema

## API Call - Trian data 
```
curl --request POST 'http://localhost:8000/api/method/chat_with_db.api.chatbot.embed_schema'

```
## API Call - Trian response 
```
{
  "message": "✅ Frappe schema embedded successfully."
}


```

## API Call - Ask Data
```
curl -X POST 'http://localhost:8000/api/method/chat_with_db.api.chatbot.ask_data' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  --data-urlencode 'question=list of patient information'

```

## API Response - Ask Data

```
{
  "message": {
    "result": [
      ["Anitha", "Anitha", "Female", "2025-07-01", "", "Active", null, null],
      ["Kiruba", "Kiruba", "Male", "2025-07-09", "", "Active", null, null],
      ["Kumar", "Kumar", "Male", "2025-06-30", "", "Active", null, null],
      ["Raj", "Raj ", "Male", null, "", "Active", null, null],
      ["Vinoth", "Vinoth", "Male", "2025-07-08", "", "Active", null, null]
    ],
    "follow_ups": [
      "1. Are you looking for a specific patient's information or a general list of all patients?",
      "2. Would you like details such as contact information, medical history, or appointment records included in the patient information list?",
      "3. Are there any specific criteria or filters you would like to apply to the patient information list, such as by age group or medical condition?"
    ]
  }
}

```

### License

mit
