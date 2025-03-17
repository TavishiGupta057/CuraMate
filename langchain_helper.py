import threading
import pandas as pd
from langchain_google_genai import GoogleGenerativeAI  # LLM for text generation
from langchain_community.vectorstores import FAISS  # FAISS for storing embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Embeddings for text search
from langchain_community.document_loaders import CSVLoader  # CSV Loader for dataset processing
from langchain.chains import RetrievalQA  # To answer multiple questions at a time
from langchain.prompts import PromptTemplate  # For structuring query prompts
from langchain.schema import Document

#  Replace with your actual API key
api1 = "AIzaSyBHhYPFiPus3Vr31z51bN3XHSKrMjqCE3A"
api2 = "AIzaSyCt3qXHjSc5gP8b2ONhuL6kfl8g7xvG380"
api3 = "AIzaSyCNWT0ImoD7NXj4XeVP6m4dn0leX6KK918"
api_key = api3

#  Initialize Google Gemini LLM (Language Model)
llm = GoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key)

#  Define the FAISS storage location (saves the vector database)
file_path = "faiss_index"
csv_file = "C:/Users/sigin/python_LLM_project/ds.csv"  # CSV file where prompts and responses are stored

#  Initialize Google Gemini Embeddings (For vectorizing text)
instr_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)


def create_vector_db():
    """
    Function to create and save a FAISS vector database from a CSV file.

    - Loads the dataset from `ds.csv`
    - Converts text data into embeddings
    - Stores the embeddings in a FAISS vector database
    - Saves the FAISS index locally for future use (avoids recomputation)
    """
    print("⏳ Updating FAISS vector database...")

    #  Load data from CSV (using 'Prompt' column as text input)
    loader = CSVLoader(file_path=csv_file, source_column="Prompt")
    data = loader.load()

    #  Convert text data into a FAISS vector database (FAISS stores embeddings for fast retrieval)
    v_db = FAISS.from_documents(documents=data, embedding=instr_embeddings)

    #  Save FAISS vector database locally (so it can be loaded later)
    v_db.save_local(file_path)

    print("✅ Vector database successfully updated!")


def get_qa_chain():
    """
    Creates a RetrievalQA chain using FAISS and ensures multiple responses are passed to LLM.
    """

    # ✅ Load FAISS vector database from local storage
    v_db = FAISS.load_local(file_path, instr_embeddings, allow_dangerous_deserialization=True)

    # ✅ Retrieve the top 2 most relevant responses
    retriever = v_db.as_retriever(search_kwargs={"k": 2 ,"score_threshold": 0.99})  # ✅ Fetch top 2 relevant docs thats the value of cosine similarity

    # ✅ Define a structured prompt template for responses
    prompt_template = '''
Given the retrieved medical context and a question, generate a **clear, structured medical response**.

**Rules:**
- Use **only relevant details**.
- Keep it **concise and professional**.
- **Skip to the next point if `*n*` is present.**
- Each symptom should be **separated and numbered (1, 2, 3, etc.).**
- Highlight **key medical terms in bold**.
- **Avoid using tables**.
- If no relevant answer is found, return:  
  **"!!!! PLEASE check with the respective Doctor!!"**

---

### **Medical Context:**  
{context}  

---

### **Question:**  
{question}  

---

### **🔹 Final Answer (Compact Medical Response per Symptom)**  

#### **1. Symptom: [Symptom Name]**
- **Condition:** [Diagnosis]  
- **Cause:** Possible causes - [Main cause], [Alternative]  
- **Treatment:** [Primary treatment], [Alternative option]  
- **Medication:** **[Drug Name] ([Dosage])** – [Usage]  
- **Precautions:** [Do this] *n* [Avoid this]  
- **Doctor Visit:** If [Worsens] *n* If [Emergency signs appear]  

Make a table purely for medication :

    |symptom |medication dosage |
    |symptom | medication  dosage |
    Make this look professional best hospital


    '''



    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=['context', 'question']
    )

    # ✅ Initialize RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm.with_config({"memory" : False}),  # Google Gemini LLM SOOO THIS DOENST REMEMBER PREVIOUS patient details
        chain_type="stuff",  # Options: "stuff", "map_reduce", "refine"
        retriever=retriever,  # ✅ Retrieves multiple relevant documents
        return_source_documents=True,  # ✅ Ensures the LLM receives multiple responses
        chain_type_kwargs={'prompt': PROMPT}
    )

    return qa_chain  # ✅ Now retrieves and formats multiple relevant responses


def add_to_faiss(prompt, response):
    """
    Adds a new prompt-response pair to FAISS without reprocessing the entire dataset.
    """

    # Try loading the existing FAISS index
    try:
        v_db = FAISS.load_local(file_path, instr_embeddings, allow_dangerous_deserialization=True)
        print("✅ Loaded existing FAISS index.")
    except:
        v_db = None  # If FAISS index does not exist
        print("⚠️ No existing FAISS index found. Creating a new one.")

    # Convert new data to Document format
    new_doc = [Document(page_content=prompt, metadata={"response": response})]

    if v_db:
        v_db.add_documents(new_doc)  # ✅ Append new entry without affecting previous embeddings
        print("✅ New embedding added to FAISS.")
    else:
        v_db = FAISS.from_documents(new_doc, instr_embeddings)  # Create new FAISS index
        print("✅ New FAISS index created.")

    # Save the updated FAISS index
    v_db.save_local(file_path)
    print("💾 FAISS vector database updated and saved.")


def learn(prompt, response):
    """
    Function to add a new Prompt-Response pair to the dataset.

    - Appends the new data to the CSV file.
    - Runs `add_to_faiss()` to update the FAISS vector database asynchronously.
    """
    df = pd.read_csv(csv_file)
    print("\n\n\n",df,"\n\n\n");
    # ✅ Append the new data directly to the CSV without reloading everything
    new_data = pd.DataFrame([[prompt, response]], columns=["Prompt", "Response"])
    
    # ✅ Append mode, avoid re-writing the entire file
    new_data.to_csv(csv_file, mode='a', header=False, index=False)

    print(f"✅ New Prompt-Response added: '{prompt}' → '{response}'")

    # ✅ Efficiently update FAISS with only the new entry
    add_to_faiss(prompt, response)

    print("✅ FAISS vector database updated successfully!")


# ✅ Run the function only when this script is executed (prevents redundant execution when imported)
if __name__ == "__main__":
    create_vector_db()  # ✅ Generates and saves the FAISS vector database
    # chain = get_qa_chain()
    
    # ✅ Example usage of `learn`
    # learn("What should I take for mild headaches?", "('Paracetamol', 'Helps relieve mild to moderate headaches.')")

    # ✅ Test query
    # print(chain("What is the best medicine for headache?"))
