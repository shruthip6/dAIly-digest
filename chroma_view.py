import chromadb

# Connect to persistent ChromaDB

client = chromadb.PersistentClient(path="./chroma_db")

# Load collection

collection = client.get_collection("ai_digest_knowledge")

# Fetch all documents

results = collection.get()

print("\n========== CHROMADB DOCUMENTS ==========\n")

for i in range(len(results["documents"])):


    print(f"DOCUMENT {i+1}")
    print("-" * 50)

    print("ID:")
    print(results["ids"][i])

    print("\nDOCUMENT:")
    print(results["documents"][i])

    print("\nMETADATA:")
    print(results["metadatas"][i])

    print("\n\n")

