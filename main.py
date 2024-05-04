from src.localllm import LocalLLM

phi3_llm = LocalLLM("phi3")

# documents = phi3_llm.read_document("my_pdf.pdf")
# phi3_llm.document_to_embeddings()

# query_input = input("Enter your query: ")
query_input = "What are coding conferences?"
phi3_llm.run(query_input)
