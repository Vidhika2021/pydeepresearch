import requests
import time
import subprocess
import sys
import os

def test_ica_payload():
    url = "http://127.0.0.1:8000/deep-research"
    payload = {
        "query": "Test Prompt for ICA",
        "stream": True,
        "context": "Some context",
        "use_context": True,
        "prompt_template": "Some template"
    }

    print(f"Sending payload: {payload}")
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response Body: {response.text}")
        
        if response.status_code == 200:
            print("✅ Request successful")
            data = response.json()
            
            if "status" in data and data["status"] == "success":
                print(f"✅ Status: {data['status']}")
            else:
                print("⚠️ Missing or invalid 'status' field")
                
            if "invocationId" in data:
                 print(f"✅ Found invocationId: {data['invocationId']}")
            else:
                 print("⚠️ Missing 'invocationId' field")

            if "response" in data and isinstance(data["response"], list):
                block = data["response"][0]
                if "message" in block and block.get("type") == "text":
                    print(f"✅ Found response message: {block['message'][:50]}...")
                else:
                    print("⚠️ Invalid response block format")
            else:
                print("⚠️ Missing or invalid 'response' field")
        else:
            print("❌ Request failed")
    except Exception as e:
        print(f"❌ Connection error: {e}")

if __name__ == "__main__":
    test_ica_payload()
