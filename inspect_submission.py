import json, sys
sys.stdout.reconfigure(encoding='utf-8')

lines = [json.loads(l) for l in open('submission.jsonl', encoding='utf-8')]
print(f'Total: {len(lines)} pairs\n')

fallback_count = 0
llm_count = 0
for l in lines:
    body = l['body']
    is_fallback = 'Quick update from Vera' in body or 'update for you' in body
    tag = 'FALLBACK' if is_fallback else 'LLM_OK '
    if is_fallback:
        fallback_count += 1
    else:
        llm_count += 1
    tid = l.get('test_id','?')
    tmpl = l.get('template_name','')[:30]
    preview = body[:90].replace('\n', ' ')
    print(f"[{tag}] {tid} {tmpl:30} {len(body):4}ch | {preview}")

print(f'\nLLM OK:   {llm_count}/30')
print(f'Fallback: {fallback_count}/30')
