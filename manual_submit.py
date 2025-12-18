import asyncio
import uuid
import json
import requests
import sys

# Mock imports or minimal imports if possible
from a2a.client import ClientFactory, ClientConfig
from a2a.types import AgentCard, Message, Role, TextPart, TaskStatusUpdateEvent, TaskArtifactUpdateEvent

async def submit_task_manual(green_url, solver_url):
    print(f"Manual Submit: Green={green_url}, Solver={solver_url}")
    
    # 1. Fetch Card
    print("Fetching card...")
    try:
        resp = requests.get(f"{green_url.rstrip('/')}/.well-known/agent-card.json")
        if resp.status_code != 200:
             print(f"Error: Failed to fetch card. Status: {resp.status_code}")
             return
        card_data = resp.json()
        # Correctly replace host/scheme while preserving path
        original_url = card_data.get("url", "")
        if original_url:
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(original_url)
            # parsed.netloc includes port if present
            # We want to replace it with localhost:8000 (from green_url)
            green_parsed = urlparse(green_url)
            
            # Reconstruct
            new_url = urlunparse((
                green_parsed.scheme, 
                green_parsed.netloc, 
                parsed.path, 
                parsed.params, 
                parsed.query, 
                parsed.fragment
            ))
            card_data["url"] = new_url
            print(f"Overridden Agent URL to local: {card_data['url']}")
        
        agent_card = AgentCard(**card_data)
        print("Card fetched successfully.")
    except Exception as e:
        print(f"Card Fetch Error: {e}")
        return

    # 2. Setup Client
    config = ClientConfig()
    factory = ClientFactory(config=config)
    client = factory.create(agent_card)
    
    # 3. Submit
    submission_data = {
        "agent_url": solver_url,
        "research_artifacts": "/path/to/research", 
        "docker_image": "placeholder"
    }
    submission_json = json.dumps(submission_data)
    
    message = Message(
        message_id=str(uuid.uuid4()),
        role=Role.user,
        parts=[TextPart(text=submission_json)],
        context_id=f"manual_sub_{uuid.uuid4().hex}",
        metadata={"submission": submission_data}
    )
    
    print("Sending request to AgentBeats agent...")
    try:
        async for item in client.send_message(request=message):
             if isinstance(item, tuple):
                 task, event = item
                 if event:
                     if isinstance(event, TaskStatusUpdateEvent):
                         msg = ""
                         if event.status and event.status.message and event.status.message.parts:
                              part = event.status.message.parts[0]
                              p = part.root if hasattr(part, 'root') else part
                              if hasattr(p, 'text'): msg = p.text
                         if msg:
                             print(f"[STATUS] {msg}")
                     
                     elif isinstance(event, TaskArtifactUpdateEvent):
                         print(f"[ARTIFACT] {event.artifact.name}")
                         if event.artifact.name == "evaluation_summary":
                             print("VICTORY! Evaluation Summary Received.")
                             # Dump content
                             if event.artifact.parts:
                                  p = event.artifact.parts[0]
                                  root = p.root if hasattr(p, 'root') else p
                                  if hasattr(root, 'data'):
                                       print(json.dumps(root.data, indent=2))
        print("Done.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during client interaction: {e}")
        
        # DEBUG: Send raw POST to see the error message
        print("\n[DEBUG] Sending raw POST to inspect response...")
        try:
             # Construct A2A request body manually-ish or just simple JSON
             # A2A V1 uses specific JSON structure. 
             # But the error implies the SERVER rejected the connection or internal error.
             # We'll try to POST the same body the client would.
             headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
             payload = {
                 "id": str(uuid.uuid4()),
                 "role": "user",
                 "parts": [{"text": submission_json}],
                 "context_id": f"manual_sub_{uuid.uuid4().hex}",
                 "metadata": {"submission": submission_data}
             }
             resp = requests.post(agent_card.url, json=payload, headers=headers)
             print(f"Raw Response Status: {resp.status_code}")
             print(f"Raw Response Headers: {resp.headers}")
             print(f"Raw Response Body: {resp.text}")
        except Exception as ex:
             print(f"Raw POST failed: {ex}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python manual_submit.py <green_url> <solver_url>")
        sys.exit(1)
    asyncio.run(submit_task_manual(sys.argv[1], sys.argv[2]))
