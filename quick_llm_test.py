"""quick_llm_test.py — verify the Gemini API key works"""
import asyncio, sys, os
sys.path.insert(0, '.')

async def test():
    from llm_client import call_gemini, extract_json
    system = "You are Vera, a merchant AI. Respond ONLY with valid JSON: {\"body\": \"<message>\", \"cta\": \"binary_yes_stop\"}"
    user = "Compose a 1-line test message for Dr. Meera, dentist in Delhi. Include the word FLUORIDE and cite JIDA Oct 2026."
    try:
        r = await call_gemini(system, user)
        print("RAW:", r[:300])
        try:
            j = extract_json(r)
            print("\nJSON OK:")
            print("  body:", j.get("body","")[:150])
            print("  cta:", j.get("cta",""))
            print("\nAPI KEY WORKS. LLM IS FUNCTIONAL.")
        except Exception as e:
            print("JSON parse issue:", e)
            print("But raw response received — API key IS valid.")
    except Exception as e:
        print("FAIL:", e)

asyncio.run(test())
