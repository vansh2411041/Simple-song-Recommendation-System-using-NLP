import requests
import json

class GroqLyricsFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def get_lyrics(self, song_name, artist):
        """Fetch lyrics using song name and artist"""
        prompt = f"""Provide ONLY the lyrics for "{song_name}" by {artist}. 
        Return plain text without any additional commentary or metadata."""
        
        payload = {
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }
        
        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Error fetching lyrics: {e}")
            return None