Here's your entire **Frank Solar Project – Work Summary**, with all entries standardized into a consistent and professional format, including proper headings, time blocks, and bullet formatting:

---

## **Frank Solar Project – Work Summary**

---

### 📅 **1st July 2025 – Work Log**

#### 🔹 **Task 1 (10:30 AM – 1:30 PM)**

* Worked on **Version 1** of the chatbot.
* Passed all necessary resources to the system prompt to build the LLM knowledge base:

  * Data from **7 sites**
  * **Summary table**
  * **3 conversation logs**
* Developed a **Python script** for chatbot testing to validate prompt quality and model responses.

#### 🔹 **Task 2 (2:30 PM – 7:30 PM)**

* Optimized the **summary table**:

  * Removed redundant data from the JSON structure.
  * Combined multiple subtables into one optimized JSON table.
* Updated the system prompt (based on suggestions from **@Kunj Gandhi**):

  * Removed the **conversation logs** to reduce token size.
  * Cleaned and refactored the prompt accordingly.
* Saved this version as **Version 2**.

---

### 📅 **2nd July 2025 – Work Log**

#### 🔹 **Task 1 (10:30 AM – 1:30 PM)**

* Optimized the **site data (7 sites)** by removing irrelevant content.
* Enhanced the **system prompt logic** to handle dynamic user responses.
* Implemented output formatting for ranked recommendations:

  * Displayed **6 water heater system types** based on user needs and preferences.

#### 🔹 **Task 2 (2:30 PM – 7:30 PM)**

* Built a **sample UI** for chatbot interaction:

  * Designed a basic **HTML frontend**.
  * Created a **Flask-based Python API** for backend responses.
* Performed **end-to-end testing** of the chatbot prototype.

---

### 📅 **3rd July 2025 – Work Log**

#### 🔹 **Morning (10:30 AM – 1:30 PM)**

* Updated the **UI** for improved layout and usability.
* Fixed a **bug** causing repeated questions during conversations.
* Refined and updated the **system prompt** to improve contextual understanding.

#### 🔹 **Afternoon (2:30 PM – 7:30 PM)**

* Implemented **streaming response** capability for real-time answers.
* Began debugging logic for **utility prioritization**.
* Fixed a key logic issue:

  * **Problem**: User without gas was recommended gas heater due to budget priority.
  * **Solution**: Prioritized utility compatibility over cost.
* Updated **system prompt logic** for better conditional ranking.

---

### 📅 **4th July 2025 – Work Log**

#### 🔹 **Morning (10:30 AM – 1:30 PM)**

* Tested and refined the **system prompt** for improved context handling.
* Reorganized **site data** into folders and topic-wise files.
* Prepared the data for **RAG (Retrieval-Augmented Generation)** integration.

#### 🔹 **Afternoon (2:30 PM – 7:30 PM)**

* Created a **Vector Database** using embedding models.
* Added a **technical summary** file for water heaters to the RAG base.
* Conducted full pipeline testing and identified bottlenecks.
* Optimized performance:

  * Refined **prompt formatting**.
  * Improved **vector retrieval logic**.

---

### 📅 **7th July 2025 – Work Log**

#### 🔹 **Task 1**

* Verified that the bot performs well **without full site files** in the system prompt.
* Created **3–5 line summaries** for each of the 7 sites.
* Added summaries to the **system prompt**, preserving context.

#### 🔹 **Task 2**

* Integrated **RAG** for user-question-based retrieval.
* Added **file-based RAG logic** to the main application to support contextual lookup when needed.

---

### 📅 **8th July 2025 – Work Log**

#### 🔹 **Morning (10:30 AM – 1:30 PM)**

* Removed all **site-specific content** from the system prompt.
* Retained only the **`summary_table` JSON**, reducing token usage from \~10,000 to \~2,900.

#### 🔹 **Afternoon (2:30 PM – 7:30 PM)**

* Verified that the LLM maintained performance after prompt reduction.
* Integrated **RAG** for external information retrieval.
* Added a **conditional agent**:

  * Continues chat if user input is an **answer**.
  * Uses RAG if it's a **new question**.

---

### 📅 **9th July 2025 – Work Log**

#### 🔹 **Morning (10:30 AM – 1:30 PM)**

* Performed **RAG pipeline testing** to validate accurate contextual responses.

#### 🔹 **Afternoon (2:30 PM – 7:30 PM)**

* Started migrating from **LangChain to LangGraph** for more modular, workflow-driven logic.

---

### 📅 **10th July 2025 – Work Log**

#### 🔹 **Morning (10:30 AM – 1:30 PM)**

* Migrated the RAG system from **Ollama** to **OpenAI**:

  * Switched embeddings to `text-embedding-3-small`.
  * Updated LLM to `gpt-4o-mini`.

#### 🔹 **Afternoon (2:30 PM – 7:30 PM)**

* **Dockerized the application** for deployment.
* Deployed to an **Azure Virtual Machine (VM)** and verified functionality.

---

### 📅 **11th July 2025 – Work Log**

#### 🔹 **Morning (10:30 AM – 1:30 PM)**

* Fixed **port binding errors** during deployment.
* Successfully tested the **live Azure deployment**.

#### 🔹 **Afternoon (2:30 PM – 7:30 PM)**

* Continued migrating core logic from **LangChain to LangGraph** for improved workflow structuring.

---
