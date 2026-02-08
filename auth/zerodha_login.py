"""
Zerodha Kite Connect daily authentication helper.

Purpose:
- Generate a fresh access token using request_token + api_secret
- Validate it
- Persist it for the trading engine

Run once per trading day.
"""

from kiteconnect import KiteConnect
from dotenv import load_dotenv, set_key
import os
import sys
from pathlib import Path

# Add project root to path so we can import config if needed
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

ENV_PATH = project_root / ".env"

def main():
    # Load env, if .env doesn't exist, we'll create it via set_key later if needed
    # but for now we expect it to exist
    load_dotenv(dotenv_path=ENV_PATH)

    api_key = os.getenv("TRADING_BROKER_API_KEY") or os.getenv("TRADING_API_KEY")
    api_secret = os.getenv("TRADING_API_SECRET")

    if not api_key or not api_secret:
        print("❌ Error: TRADING_BROKER_API_KEY or TRADING_API_SECRET missing in .env")
        print("Please ensure your .env has:")
        print("TRADING_BROKER_API_KEY=your_api_key")
        print("TRADING_API_SECRET=your_api_secret")
        return

    kite = KiteConnect(api_key=api_key)

    print("\n🔐 1. Open this URL in your browser and login:")
    print("-" * 50)
    print(kite.login_url())
    print("-" * 50)

    print("\n🔐 2. After login, you will be redirected to your callback URL.")
    print("Example: http://127.0.0.1:5000/callback?request_token=XXXXXX...")
    
    request_token = input("\nPaste the 'request_token' from that URL: ").strip()

    if not request_token:
        print("❌ Error: No request token provided.")
        return

    try:
        print("\n⏳ Exchanging request token for session...")
        data = kite.generate_session(request_token, api_secret)
        access_token = data["access_token"]
        user_name = data.get("user_name", "User")

        # Validate immediately
        kite.set_access_token(access_token)
        profile = kite.profile()

        print(f"\n✅ SUCCESS! Authenticated as: {profile.get('user_name', user_name)}")

        # Persist to .env
        # Note: set_key handles the file I/O safely
        set_key(str(ENV_PATH), "TRADING_BROKER_ACCESS_TOKEN", access_token)

        print(f"\n💾 Access token saved to: {ENV_PATH}")
        print("🚀 You can now start the trading engine (python main.py)")

    except Exception as e:
        print(f"\n❌ Failed to authenticate: {e}")
        print("\nPossible issues:")
        print("- Request token expired (they are extremely short-lived)")
        print("- API Secret is incorrect")
        print("- You used a request token from a different API Key")

if __name__ == "__main__":
    main()
