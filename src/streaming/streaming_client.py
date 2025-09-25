#!/usr/bin/env python3
"""
Real-Time ML Streaming Client

Interactive WebSocket client for receiving live stock predictions.
Demonstrates real-time ML capabilities with formatted terminal output.
"""

import asyncio
import json
import websockets
import sys
from datetime import datetime
from typing import Dict, Any
import argparse

class StreamingClient:
    """
    WebSocket client for real-time ML predictions
    
    Features:
    - Real-time prediction display with color formatting
    - Interactive command interface
    - Connection resilience with auto-reconnect
    - Performance tracking and statistics
    """
    
    def __init__(self, server_url: str = "ws://localhost:8765"):
        self.server_url = server_url
        self.websocket = None
        self.connected = False
        self.prediction_count = 0
        self.start_time = datetime.now()
        
        # Color codes for terminal output
        self.colors = {
            'green': '\033[92m',
            'red': '\033[91m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'purple': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'bold': '\033[1m',
            'end': '\033[0m'
        }
    
    def print_colored(self, text: str, color: str = 'white'):
        """Print colored text to terminal"""
        print(f"{self.colors.get(color, '')}{text}{self.colors['end']}")
    
    def format_currency(self, value: float) -> str:
        """Format currency with proper precision"""
        return f"${value:.2f}"
    
    def format_percentage(self, value: float) -> str:
        """Format percentage with color coding"""
        color = 'green' if value > 0 else 'red' if value < 0 else 'white'
        return f"{self.colors[color]}{value:+.2f}%{self.colors['end']}"
    
    def format_trend(self, trend: str) -> str:
        """Format trend direction with colors"""
        colors = {
            'BULLISH': 'green',
            'BEARISH': 'red', 
            'OVERBOUGHT': 'yellow',
            'OVERSOLD': 'cyan',
            'UNKNOWN': 'white'
        }
        color = colors.get(trend, 'white')
        return f"{self.colors[color]}{trend}{self.colors['end']}"
    
    def display_prediction(self, prediction: Dict[str, Any]):
        """Display formatted prediction data"""
        symbol = prediction.get('symbol', 'UNKNOWN')
        current_price = prediction.get('current_price', 0.0)
        predicted_price = prediction.get('predicted_price', 0.0)
        change_pct = prediction.get('predicted_change_pct', 0.0)
        confidence = prediction.get('confidence_score', 0.0)
        trend = prediction.get('trend_direction', 'UNKNOWN')
        volatility = prediction.get('volatility_score', 0.0)
        processing_time = prediction.get('processing_time_ms', 0.0)
        timestamp = prediction.get('timestamp', '')
        
        # Parse timestamp for display
        try:
            ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            time_str = ts.strftime('%H:%M:%S')
        except:
            time_str = timestamp
        
        # Display formatted prediction
        self.print_colored(f"\n{'='*60}", 'blue')
        self.print_colored(f"📊 {symbol} PREDICTION UPDATE", 'bold')
        self.print_colored(f"{'='*60}", 'blue')
        self.print_colored(f"⏰ Time: {time_str}", 'cyan')
        self.print_colored(f"💰 Current:  {self.format_currency(current_price)}", 'white')
        self.print_colored(f"🎯 Predicted: {self.format_currency(predicted_price)}", 'white')
        self.print_colored(f"📈 Change:    {self.format_percentage(change_pct)}", 'white')
        self.print_colored(f"🎪 Trend:     {self.format_trend(trend)}", 'white')
        self.print_colored(f"🎲 Confidence: {confidence:.1%}", 'purple')
        self.print_colored(f"⚡ Volatility: {volatility:.3f}", 'yellow')
        self.print_colored(f"⚙️  Processing: {processing_time:.1f}ms", 'cyan')
        
        self.prediction_count += 1
    
    def display_summary(self, predictions: list):
        """Display summary of all predictions"""
        if not predictions:
            return
            
        self.print_colored(f"\n{'🌟' * 20}", 'yellow')
        self.print_colored(f"📊 LIVE PREDICTIONS SUMMARY", 'bold')
        self.print_colored(f"{'🌟' * 20}", 'yellow')
        
        for pred in predictions:
            symbol = pred.get('symbol', 'UNKNOWN')
            change_pct = pred.get('predicted_change_pct', 0.0)
            confidence = pred.get('confidence_score', 0.0)
            trend = pred.get('trend_direction', 'UNKNOWN')
            
            self.print_colored(
                f"{symbol:>6}: {self.format_percentage(change_pct)} "
                f"({confidence:.0%} confidence, {self.format_trend(trend)})", 
                'white'
            )
    
    def display_stats(self):
        """Display client connection statistics"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        uptime_str = f"{uptime:.0f}s"
        
        self.print_colored(f"\n📊 Client Stats:", 'blue')
        self.print_colored(f"   Connected: {uptime_str}", 'cyan')
        self.print_colored(f"   Predictions: {self.prediction_count}", 'cyan')
        self.print_colored(f"   Server: {self.server_url}", 'cyan')
    
    async def handle_message(self, message: str):
        """Process incoming WebSocket messages"""
        try:
            data = json.loads(message)
            msg_type = data.get('type', 'unknown')
            
            if msg_type == 'live_predictions':
                predictions = data.get('predictions', [])
                client_count = data.get('client_count', 0)
                timestamp = data.get('timestamp', '')
                
                # Display each prediction
                for prediction in predictions:
                    self.display_prediction(prediction)
                
                # Display summary
                self.display_summary(predictions)
                
                # Show client info
                self.print_colored(f"\n👥 Connected clients: {client_count}", 'blue')
                self.display_stats()
                
            elif msg_type == 'initial_data':
                predictions = data.get('predictions', [])
                self.print_colored("🚀 Received initial data from server", 'green')
                self.display_summary(predictions)
                
            elif msg_type == 'status':
                streaming = data.get('streaming', False)
                symbols = data.get('symbols', [])
                interval = data.get('update_interval', 0)
                clients = data.get('client_count', 0)
                
                self.print_colored(f"\n📊 Server Status:", 'blue')
                self.print_colored(f"   Streaming: {'✅' if streaming else '❌'}", 'cyan')
                self.print_colored(f"   Symbols: {', '.join(symbols)}", 'cyan')
                self.print_colored(f"   Update Interval: {interval}s", 'cyan')
                self.print_colored(f"   Connected Clients: {clients}", 'cyan')
                
        except json.JSONDecodeError as e:
            self.print_colored(f"❌ JSON decode error: {e}", 'red')
        except Exception as e:
            self.print_colored(f"❌ Message handling error: {e}", 'red')
    
    async def send_command(self, command: str):
        """Send command to server"""
        if self.websocket and self.connected:
            try:
                message = {"command": command}
                await self.websocket.send(json.dumps(message))
                self.print_colored(f"📤 Sent command: {command}", 'yellow')
            except Exception as e:
                self.print_colored(f"❌ Send error: {e}", 'red')
    
    async def interactive_mode(self):
        """Run interactive command loop"""
        self.print_colored("\n🎮 Interactive Mode Commands:", 'blue')
        self.print_colored("   'status' - Get server status", 'cyan')
        self.print_colored("   'stats' - Show client statistics", 'cyan')
        self.print_colored("   'quit' - Disconnect and exit", 'cyan')
        self.print_colored("   Press Enter for commands...\n", 'cyan')
        
        while self.connected:
            try:
                # Use asyncio timeout for non-blocking input
                command = await asyncio.wait_for(
                    asyncio.to_thread(input, "Command> "), 
                    timeout=1.0
                )
                
                if command.lower() == 'quit':
                    break
                elif command.lower() == 'status':
                    await self.send_command('get_status')
                elif command.lower() == 'stats':
                    self.display_stats()
                elif command.strip():
                    await self.send_command(command)
                    
            except asyncio.TimeoutError:
                continue
            except EOFError:
                break
            except Exception as e:
                self.print_colored(f"❌ Input error: {e}", 'red')
    
    async def connect(self):
        """Connect to WebSocket server with auto-reconnect"""
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                self.print_colored(f"🔌 Connecting to {self.server_url}...", 'yellow')
                
                self.websocket = await websockets.connect(
                    self.server_url,
                    ping_interval=20,
                    ping_timeout=10
                )
                
                self.connected = True
                self.print_colored("✅ Connected to ML streaming server!", 'green')
                
                # Start listening for messages and handle interactive commands
                listen_task = asyncio.create_task(self.listen_for_messages())
                interactive_task = asyncio.create_task(self.interactive_mode())
                
                # Wait for either task to complete
                done, pending = await asyncio.wait(
                    [listen_task, interactive_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel remaining tasks
                for task in pending:
                    task.cancel()
                    
                break
                
            except websockets.exceptions.ConnectionClosed:
                self.print_colored("❌ Connection closed by server", 'red')
                self.connected = False
                break
            except Exception as e:
                retry_count += 1
                self.print_colored(f"❌ Connection failed (attempt {retry_count}): {e}", 'red')
                if retry_count < max_retries:
                    await asyncio.sleep(2)
                else:
                    self.print_colored("❌ Max retries exceeded. Giving up.", 'red')
    
    async def listen_for_messages(self):
        """Listen for incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                await self.handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            self.print_colored("❌ Connection lost", 'red')
            self.connected = False
        except Exception as e:
            self.print_colored(f"❌ Listen error: {e}", 'red')
            self.connected = False
    
    async def disconnect(self):
        """Clean disconnect from server"""
        if self.websocket:
            self.connected = False
            await self.websocket.close()
            self.print_colored("👋 Disconnected from server", 'yellow')

async def main():
    """Main entry point for streaming client"""
    parser = argparse.ArgumentParser(description='Real-Time ML Streaming Client')
    parser.add_argument('--url', default='ws://localhost:8765', 
                       help='WebSocket server URL (default: ws://localhost:8765)')
    parser.add_argument('--quiet', action='store_true', 
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    client = StreamingClient(server_url=args.url)
    
    # Display banner
    if not args.quiet:
        client.print_colored("🚀" * 20, 'blue')
        client.print_colored("📊 REAL-TIME ML STREAMING CLIENT", 'bold')
        client.print_colored("🚀" * 20, 'blue')
        client.print_colored(f"Server: {args.url}", 'cyan')
        client.print_colored("Press Ctrl+C to exit\n", 'yellow')
    
    try:
        await client.connect()
    except KeyboardInterrupt:
        client.print_colored("\n🛑 Interrupted by user", 'yellow')
    finally:
        await client.disconnect()
        client.print_colored("✅ Client shutdown complete", 'green')

if __name__ == "__main__":
    asyncio.run(main())