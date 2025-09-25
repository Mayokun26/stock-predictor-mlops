#!/usr/bin/env python3
"""
Real-Time Streaming ML Demo

Quick demonstration of the WebSocket-based streaming ML prediction system.
Shows live predictions updating every 30 seconds with formatted output.
"""

import asyncio
import json
import websockets
import subprocess
import sys
import os
import time
from datetime import datetime

async def demo_streaming_system():
    """
    Interactive demo of the real-time streaming ML system
    
    1. Starts the streaming server
    2. Connects a demo client
    3. Shows live predictions for 2 minutes
    4. Cleans up automatically
    """
    
    print("🚀" + "="*58 + "🚀")
    print("📊 REAL-TIME STREAMING ML PREDICTION DEMO")  
    print("🚀" + "="*58 + "🚀")
    print()
    
    server_process = None
    
    try:
        # Step 1: Start the streaming server
        print("🔧 Starting ML streaming server...")
        cmd = [sys.executable, "src/streaming/real_time_predictor.py"]
        server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server startup
        print("⏳ Waiting for server initialization...")
        await asyncio.sleep(4)
        
        if server_process.poll() is not None:
            stderr = server_process.stderr.read().decode()
            print(f"❌ Server failed to start: {stderr}")
            return False
        
        print("✅ Server started successfully!")
        print()
        
        # Step 2: Connect demo client
        print("🔌 Connecting to streaming server...")
        server_url = "ws://localhost:8765"
        
        websocket = await websockets.connect(
            server_url,
            ping_interval=20,
            ping_timeout=10
        )
        
        print(f"✅ Connected to {server_url}")
        print()
        
        # Step 3: Get server status
        print("📊 Requesting server status...")
        status_request = {"command": "get_status"}
        await websocket.send(json.dumps(status_request))
        
        status_response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
        status_data = json.loads(status_response)
        
        print(f"   🔄 Streaming: {'✅' if status_data.get('streaming') else '❌'}")
        print(f"   📈 Symbols: {', '.join(status_data.get('symbols', []))}")
        print(f"   ⏱️  Update Interval: {status_data.get('update_interval', 0)}s")
        print(f"   👥 Connected Clients: {status_data.get('client_count', 0)}")
        print()
        
        # Step 4: Listen for live predictions
        print("📡 Listening for live predictions (2 minutes)...")
        print("   Press Ctrl+C to stop early")
        print("-" * 60)
        
        start_time = time.time()
        prediction_count = 0
        
        while time.time() - start_time < 120:  # 2 minutes
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=35.0)
                data = json.loads(message)
                
                if data.get('type') == 'live_predictions':
                    predictions = data.get('predictions', [])
                    timestamp = data.get('timestamp', '')
                    client_count = data.get('client_count', 0)
                    
                    # Parse timestamp
                    try:
                        ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        time_str = ts.strftime('%H:%M:%S')
                    except:
                        time_str = datetime.now().strftime('%H:%M:%S')
                    
                    print(f"\n⏰ [{time_str}] Live Predictions Update (👥 {client_count} clients)")
                    print("-" * 60)
                    
                    for pred in predictions:
                        symbol = pred.get('symbol', 'N/A')
                        current_price = pred.get('current_price', 0.0)
                        predicted_price = pred.get('predicted_price', 0.0)
                        change_pct = pred.get('predicted_change_pct', 0.0)
                        confidence = pred.get('confidence_score', 0.0)
                        trend = pred.get('trend_direction', 'UNKNOWN')
                        processing_time = pred.get('processing_time_ms', 0.0)
                        
                        # Color code the change percentage
                        change_color = "📈" if change_pct > 0 else "📉" if change_pct < 0 else "➡️"
                        change_sign = "+" if change_pct > 0 else ""
                        
                        print(f"   {symbol:>6}: ${current_price:7.2f} → ${predicted_price:7.2f} "
                              f"{change_color} {change_sign}{change_pct:5.2f}% "
                              f"({confidence:4.0%} conf, {trend}, {processing_time:.0f}ms)")
                    
                    prediction_count += len(predictions)
                    
                elif data.get('type') == 'initial_data':
                    initial_preds = data.get('predictions', [])
                    print(f"📦 Received initial data: {len(initial_preds)} predictions")
                    
            except asyncio.TimeoutError:
                print("⚠️  No new predictions received (waiting...)")
            except Exception as e:
                print(f"❌ Error receiving predictions: {e}")
                break
        
        await websocket.close()
        print()
        print("🏁 Demo completed!")
        print(f"📊 Total predictions received: {prediction_count}")
        
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
        return True
    except Exception as e:
        print(f"❌ Demo error: {e}")
        return False
    finally:
        # Cleanup
        if server_process:
            print("🧹 Shutting down server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
                print("✅ Server shutdown complete")
            except:
                server_process.kill()
                print("🔨 Server forcefully terminated")

async def quick_test():
    """Quick connectivity test"""
    print("🔍 Quick connectivity test...")
    
    try:
        websocket = await websockets.connect("ws://localhost:8765", ping_timeout=5)
        print("✅ Server is running and accepting connections")
        await websocket.close()
        return True
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("💡 Make sure to run the server first:")
        print("   python3 src/streaming/real_time_predictor.py")
        return False

def print_usage():
    """Print usage information"""
    print("🎮 Real-Time Streaming ML Demo")
    print()
    print("Usage:")
    print("   python3 demo_streaming.py [command]")
    print()
    print("Commands:")
    print("   demo     - Full demonstration (default)")
    print("   test     - Quick connectivity test")
    print("   client   - Launch interactive client")
    print("   help     - Show this help")
    print()
    print("Examples:")
    print("   python3 demo_streaming.py demo")
    print("   python3 demo_streaming.py test")
    print("   python3 demo_streaming.py client")

async def launch_interactive_client():
    """Launch the interactive streaming client"""
    print("🎮 Launching interactive client...")
    try:
        # Import and run the streaming client
        from src.streaming.streaming_client import main as client_main
        await client_main()
    except ImportError:
        print("❌ Cannot import streaming client")
        print("💡 Make sure src/streaming/streaming_client.py exists")
    except Exception as e:
        print(f"❌ Client error: {e}")

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "demo"
    
    if command == "help":
        print_usage()
    elif command == "test":
        asyncio.run(quick_test())
    elif command == "client":
        asyncio.run(launch_interactive_client())
    elif command == "demo":
        success = asyncio.run(demo_streaming_system())
        sys.exit(0 if success else 1)
    else:
        print(f"❌ Unknown command: {command}")
        print_usage()
        sys.exit(1)