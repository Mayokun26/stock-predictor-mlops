#!/bin/bash
# Real-Time Streaming ML System Commands
# Quick commands for managing the streaming ML system

echo "🚀 Real-Time Streaming ML System Commands"
echo "========================================"

case "$1" in
    "start"|"server")
        echo "🔧 Starting ML streaming server..."
        python3 src/streaming/real_time_predictor.py
        ;;
    "client")
        echo "🎮 Launching interactive client..."
        python3 src/streaming/streaming_client.py
        ;;
    "demo")
        echo "📊 Running full demo..."
        python3 demo_streaming.py demo
        ;;
    "test")
        echo "🧪 Running system tests..."
        python3 test_streaming_system.py
        ;;
    "quick-test")
        echo "⚡ Quick connectivity test..."
        python3 demo_streaming.py test
        ;;
    "status")
        echo "📊 Checking server status..."
        python3 -c "
import asyncio
import json
import websockets

async def check_status():
    try:
        ws = await websockets.connect('ws://localhost:8765')
        await ws.send(json.dumps({'command': 'get_status'}))
        response = await ws.recv()
        data = json.loads(response)
        print(f\"✅ Server Status:\")
        print(f\"   Streaming: {'✅' if data.get('streaming') else '❌'}\")
        print(f\"   Symbols: {', '.join(data.get('symbols', []))}\")
        print(f\"   Update Interval: {data.get('update_interval', 0)}s\")
        print(f\"   Connected Clients: {data.get('client_count', 0)}\")
        await ws.close()
    except Exception as e:
        print(f\"❌ Cannot connect: {e}\")
        print(f\"💡 Start server with: ./streaming_commands.sh start\")

asyncio.run(check_status())
"
        ;;
    "kill"|"stop")
        echo "🛑 Stopping streaming server..."
        pkill -f "real_time_predictor.py" && echo "✅ Server stopped" || echo "❌ No server running"
        ;;
    "help"|"")
        echo ""
        echo "Usage: ./streaming_commands.sh [command]"
        echo ""
        echo "Commands:"
        echo "  start        - Start the streaming ML server"
        echo "  client       - Launch interactive WebSocket client"  
        echo "  demo         - Run full 2-minute demonstration"
        echo "  test         - Run comprehensive test suite"
        echo "  quick-test   - Quick connectivity test"
        echo "  status       - Check server status"
        echo "  stop         - Stop the streaming server"
        echo "  help         - Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./streaming_commands.sh start     # Start server"
        echo "  ./streaming_commands.sh client    # Connect client"
        echo "  ./streaming_commands.sh demo      # Full demo"
        echo "  ./streaming_commands.sh test      # Run tests"
        echo ""
        echo "WebSocket URL: ws://localhost:8765"
        echo "Update Interval: 30 seconds"
        echo "Symbols: AAPL, MSFT, TSLA, GOOGL, AMZN"
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo "💡 Use './streaming_commands.sh help' for available commands"
        exit 1
        ;;
esac