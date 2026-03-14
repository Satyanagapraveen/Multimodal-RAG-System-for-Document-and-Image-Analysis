from src.retrieval.retriever import retrieve, format_context_for_generation

print('=== RETRIEVAL TEST 1 — Text Query ===')
result = retrieve('What is the fragmentation of B-trees?', n_results=5)
print('Total found:', result['total_found'])
print('Breakdown:', result['breakdown'])
print()
for i, r in enumerate(result['results']):
    print(f"Result {i+1}: [{r['content_type']}] score={r['similarity_score']} | page={r['metadata']['page_number']}")
    print(f"  Preview: {r['content'][:100]}")
    print()

print('=== RETRIEVAL TEST 2 — Image-targeting Query ===')
result2 = retrieve('show me the chart or figure in the paper', n_results=5)
print('Breakdown:', result2['breakdown'])
for r in result2['results']:
    print(f"[{r['content_type']}] score={r['similarity_score']} | page={r['metadata']['page_number']}")

print()
print('=== FORMAT CONTEXT TEST ===')
context = format_context_for_generation(result)
print('Image paths found:', context['image_paths'])
print('Source references count:', len(context['source_references']))
print('Text context preview:')
print(context['text_context'][:300])
