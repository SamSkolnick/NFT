import httpx
import asyncio
from a2a.client import ClientConfig

async def test():
    urls = [
        "http://localhost:8000/.well-known/agent-card.json",
        "http://127.0.0.1:8000/.well-known/agent-card.json"
    ]
    
    for url in urls:
        print(f"Testing {url}...")
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url)
                print(f"Success! Status: {resp.status_code}")
                print(f"Content preview: {resp.text[:50]}...")
        except Exception as e:
            print(f"Failed: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test())
