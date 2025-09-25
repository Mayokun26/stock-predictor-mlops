#!/usr/bin/env python3
"""
Real-Time Streaming ML Prediction Engine

Provides WebSocket-based live predictions that update every 30 seconds.
Demonstrates staff-level real-time ML capabilities for high-frequency trading.
"""

import asyncio
import json
import logging
import websockets
from datetime import datetime
from typing import Dict, List, Set, Any
import yfinance as yf
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StreamingPrediction:
    """Real-time prediction data structure"""
    symbol: str
    timestamp: str
    predicted_price: float
    current_price: float
    predicted_change_pct: float
    confidence_score: float
    trend_direction: str
    volatility_score: float
    features_used: int
    model_version: str
    processing_time_ms: float

class RealTimeMLPredictor:
    """
    Real-time ML prediction engine with WebSocket streaming
    
    Features:
    - Live market data fetching every 30 seconds
    - Real-time feature engineering and prediction
    - WebSocket broadcasting to multiple clients
    - High-performance async processing
    """
    
    def __init__(self, symbols: List[str] = None, update_interval: int = 30):
        self.symbols = symbols or ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN"]
        self.update_interval = update_interval
        self.clients: Set[Any] = set()
        self.latest_predictions: Dict[str, StreamingPrediction] = {}
        self.is_streaming = False
        
        logger.info(f"🚀 Real-time ML predictor initialized for {len(self.symbols)} symbols")
        logger.info(f"📡 Update interval: {update_interval} seconds")
    
    async def register_client(self, websocket: Any):
        """Register new WebSocket client"""
        self.clients.add(websocket)
        client_count = len(self.clients)
        logger.info(f"📱 New client connected. Total clients: {client_count}")
        
        # Send latest predictions to new client immediately
        if self.latest_predictions:
            await self.send_to_client(websocket, {
                "type": "initial_data",
                "predictions": [asdict(pred) for pred in self.latest_predictions.values()],
                "timestamp": datetime.now().isoformat()
            })
    
    async def unregister_client(self, websocket: Any):
        """Unregister WebSocket client"""
        self.clients.discard(websocket)
        client_count = len(self.clients)
        logger.info(f"📱 Client disconnected. Total clients: {client_count}")
    
    async def send_to_client(self, websocket: Any, data: Dict):
        """Send data to specific client with error handling"""
        try:
            await websocket.send(json.dumps(data))
        except websockets.exceptions.ConnectionClosed:
            await self.unregister_client(websocket)
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
            await self.unregister_client(websocket)
    
    async def broadcast_predictions(self, predictions: List[StreamingPrediction]):
        """Broadcast predictions to all connected clients"""
        if not self.clients:
            return
        
        message = {
            "type": "live_predictions",
            "predictions": [asdict(pred) for pred in predictions],
            "timestamp": datetime.now().isoformat(),
            "client_count": len(self.clients)
        }
        
        # Send to all clients concurrently
        disconnected_clients = []
        for client in self.clients:
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.append(client)
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
                disconnected_clients.append(client)
        
        # Clean up disconnected clients
        for client in disconnected_clients:
            await self.unregister_client(client)
        
        logger.info(f"📡 Broadcast sent to {len(self.clients)} clients")
    
    def calculate_advanced_features(self, df: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Real-time feature engineering optimized for streaming"""
        try:
            features = {}
            
            # Basic features
            features['current_price'] = float(df['Close'].iloc[-1])
            features['volume'] = float(df['Volume'].iloc[-1])
            
            # Technical indicators (optimized for real-time)
            if len(df) >= 14:
                # RSI
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                features['rsi'] = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0
            else:
                features['rsi'] = 50.0
            
            # Moving averages
            for window in [5, 10, 20]:
                if len(df) >= window:
                    sma = df['Close'].rolling(window).mean()
                    features[f'sma_{window}'] = float(sma.iloc[-1])
                    features[f'sma_{window}_ratio'] = features['current_price'] / features[f'sma_{window}']
                else:
                    features[f'sma_{window}'] = features['current_price']
                    features[f'sma_{window}_ratio'] = 1.0
            
            # Volatility
            if len(df) >= 20:
                returns = df['Close'].pct_change()
                volatility = returns.rolling(20).std() * np.sqrt(252)
                features['volatility_20d'] = float(volatility.iloc[-1]) if not np.isnan(volatility.iloc[-1]) else 0.2
            else:
                features['volatility_20d'] = 0.2
                
            # Momentum
            if len(df) >= 2:
                features['return_1d'] = (df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1)
            else:
                features['return_1d'] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Feature calculation error for {symbol}: {e}")
            return {"current_price": 100.0, "volume": 1000000.0, "rsi": 50.0}
    
    def generate_mock_prediction(self, symbol: str, features: Dict[str, float]) -> StreamingPrediction:
        """Generate realistic prediction (replace with actual model later)"""
        current_price = features.get('current_price', 100.0)
        
        # Simulate realistic price movement (-2% to +2%)
        change_pct = np.random.normal(0, 0.01)  # Mean 0, std 1%
        predicted_price = current_price * (1 + change_pct)
        
        # Confidence based on volatility and volume
        volatility = features.get('volatility_20d', 0.2)
        confidence = max(0.3, min(0.95, 0.8 - volatility * 2))
        
        # Trend direction
        rsi = features.get('rsi', 50)
        if rsi > 70:
            trend = "OVERBOUGHT"
        elif rsi < 30:
            trend = "OVERSOLD"
        elif change_pct > 0:
            trend = "BULLISH"
        else:
            trend = "BEARISH"
        
        return StreamingPrediction(
            symbol=symbol,
            timestamp=datetime.now().isoformat(),
            predicted_price=round(predicted_price, 2),
            current_price=round(current_price, 2),
            predicted_change_pct=round(change_pct * 100, 2),
            confidence_score=round(confidence, 3),
            trend_direction=trend,
            volatility_score=round(volatility, 3),
            features_used=len(features),
            model_version="streaming_v1.0",
            processing_time_ms=0.0  # Will be calculated in main loop
        )
    
    async def fetch_and_predict(self, symbol: str) -> StreamingPrediction:
        """Fetch live data and generate prediction for single symbol"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Fetch recent data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="30d")  # 30 days for good indicators
            
            if hist.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Calculate features
            features = self.calculate_advanced_features(hist, symbol)
            
            # Generate prediction
            prediction = self.generate_mock_prediction(symbol, features)
            
            # Calculate processing time
            end_time = asyncio.get_event_loop().time()
            prediction.processing_time_ms = round((end_time - start_time) * 1000, 2)
            
            logger.info(f"💹 {symbol}: ${prediction.predicted_price} ({prediction.predicted_change_pct:+.2f}%) "
                       f"Confidence: {prediction.confidence_score:.1%} [{prediction.processing_time_ms:.1f}ms]")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            # Return fallback prediction
            return StreamingPrediction(
                symbol=symbol,
                timestamp=datetime.now().isoformat(),
                predicted_price=100.0,
                current_price=100.0,
                predicted_change_pct=0.0,
                confidence_score=0.5,
                trend_direction="UNKNOWN",
                volatility_score=0.2,
                features_used=0,
                model_version="fallback_v1.0",
                processing_time_ms=0.0
            )
    
    async def prediction_loop(self):
        """Main streaming loop - generates predictions every interval"""
        logger.info(f"🔄 Starting prediction loop (every {self.update_interval}s)")
        self.is_streaming = True
        
        while self.is_streaming:
            try:
                # Generate predictions for all symbols concurrently
                tasks = [self.fetch_and_predict(symbol) for symbol in self.symbols]
                predictions = await asyncio.gather(*tasks)
                
                # Update latest predictions
                for prediction in predictions:
                    self.latest_predictions[prediction.symbol] = prediction
                
                # Broadcast to all clients
                if self.clients:
                    await self.broadcast_predictions(predictions)
                
                # Wait for next interval
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Prediction loop error: {e}")
                await asyncio.sleep(5)  # Short retry delay
    
    async def handle_websocket_client(self, websocket, path: str = None):
        """Handle individual WebSocket client connections"""
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    logger.info(f"📨 Received: {data}")
                    
                    # Handle client commands
                    if data.get("command") == "get_status":
                        status = {
                            "type": "status",
                            "streaming": self.is_streaming,
                            "symbols": self.symbols,
                            "update_interval": self.update_interval,
                            "client_count": len(self.clients),
                            "latest_update": datetime.now().isoformat()
                        }
                        await self.send_to_client(websocket, status)
                        
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from client: {message}")
                except Exception as e:
                    logger.error(f"Client message handling error: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)
    
    async def start_server(self, host: str = "localhost", port: int = 8765):
        """Start WebSocket server"""
        logger.info(f"🌐 Starting WebSocket server on ws://{host}:{port}")
        
        # Start prediction loop
        prediction_task = asyncio.create_task(self.prediction_loop())
        
        # Start WebSocket server
        server = await websockets.serve(
            self.handle_websocket_client,
            host,
            port,
            ping_interval=20,
            ping_timeout=10
        )
        
        logger.info(f"✅ Real-time ML streaming server ready!")
        logger.info(f"📊 Streaming {len(self.symbols)} symbols every {self.update_interval} seconds")
        logger.info(f"🔗 Connect to: ws://{host}:{port}")
        
        try:
            await server.wait_closed()
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down streaming server...")
            self.is_streaming = False
            prediction_task.cancel()
            server.close()
            await server.wait_closed()

async def main():
    """Main entry point for real-time streaming"""
    predictor = RealTimeMLPredictor(
        symbols=["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN"],
        update_interval=30  # 30 seconds
    )
    
    await predictor.start_server(host="0.0.0.0", port=8765)

if __name__ == "__main__":
    asyncio.run(main())