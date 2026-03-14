from src.generation.generator import run_rag_pipeline

# print('=== TEST 1: Text-based question ===')
# result = run_rag_pipeline('What is the main contribution of the attention mechanism paper?')
# print('ANSWER:')
# print(result['answer'])
# print()
# print('SOURCES:')
# for s in result['sources'][:3]:
#     print(f'- {s["document_id"]} | Page {s["page_number"]} | {s["content_type"]}')
# print(f'Response time: {result["response_time_seconds"]}s')


# print('=== TEST 2: Visual question ===')
# result = run_rag_pipeline('Describe any architecture diagram or figure found in the papers')
# print('ANSWER:')
# print(result['answer'])
# print()
# print(f'Images used: {result["images_used"]}')
# print(f'Response time: {result["response_time_seconds"]}s')

# from src.retrieval.retriever import retrieve

# result = retrieve('architecture diagram transformer model', n_results=5)

# print('=== RAW RESULTS ===')
# for r in result['results']:
#     print(f'Type: {r["content_type"]}')
#     print(f'Metadata keys: {list(r["metadata"].keys())}')   # Fixed: r["metadata"]
#     if r['content_type'] == 'image':
#         print(f'image_path in metadata: {r["metadata"].get("image_path", "NOT FOUND")}')  # Fixed: r["metadata"]
#     print()

# from src.retrieval.retriever import retrieve, format_context_for_generation
# import os

# result = retrieve("architecture diagram transformer model", n_results=5)

# print("Breakdown:", result.get("breakdown"))
# print()
# print("=== ALL TOP 5 RESULTS ===")

# for i, r in enumerate(result.get("results", []), start=1):
#     metadata = r.get("metadata", {})
#     page = metadata.get("page_number", "NA")
#     print(f"{i}. [{r.get('content_type')}] score={r.get('similarity_score')} page={page}")

#     if r.get("content_type") == "image":
#         path = metadata.get("image_path", "MISSING")
#         exists = os.path.exists(path) if path != "MISSING" else False
#         print(f"   image_path: {path}")
#         print(f"   image_path exists: {exists}")

# print()
# context = format_context_for_generation(result)
# image_paths = context.get("image_paths", [])
# print("Images collected:", len(image_paths))
# for p in image_paths:
#     print(" -", os.path.basename(p)) 

from src.generation.generator import run_rag_pipeline

result = run_rag_pipeline("Describe the transformer architecture diagram and its components")

print("ANSWER:")
print(result["answer"])
print()
print(f"Images used: {result['images_used']}")
print(f"Response time: {result['response_time_seconds']}s")
print("Sources:")
for s in result["sources"]:
    print(f"  [{s['content_type']}] {s['document_id']} page {s['page_number']}")