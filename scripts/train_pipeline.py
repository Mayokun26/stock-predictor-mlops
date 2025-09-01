#!/usr/bin/env python3
"""
Automated Model Training Pipeline for CI/CD
Integrates with MLflow for model versioning and promotion
"""

import os
import sys
import asyncio
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from train_model_with_mlflow import train_and_register_model
from comprehensive_model_evaluation import ComprehensiveModelEvaluator, EvaluationConfig
import mlflow

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLflowTrainingPipeline:
    """Automated training pipeline with MLflow integration"""
    
    def __init__(self, mlflow_uri: str = "http://localhost:5001"):
        self.mlflow_uri = mlflow_uri
        mlflow.set_tracking_uri(mlflow_uri)
        self.client = mlflow.MlflowClient()
        
    async def run_training_pipeline(self, symbols: list = None, auto_promote: bool = True):
        """Run complete training pipeline"""
        if symbols is None:
            symbols = ["AAPL", "MSFT", "TSLA"]
            
        logger.info(f"🚀 Starting automated training pipeline for {len(symbols)} symbols")
        
        results = {}
        
        # Step 1: Train all models
        logger.info("📚 Step 1: Training models")
        for symbol in symbols:
            try:
                logger.info(f"Training {symbol} model...")
                result = await train_and_register_model(symbol)
                results[symbol] = result
                logger.info(f"✅ {symbol} training completed - Test R²: {result['test_score']:.4f}")
            except Exception as e:
                logger.error(f"❌ Failed to train {symbol}: {e}")
                results[symbol] = {"error": str(e)}
        
        # Step 2: Run comprehensive evaluation
        logger.info("📊 Step 2: Running comprehensive evaluation")
        try:
            config = EvaluationConfig(
                symbols=symbols,
                evaluation_period_days=20,
                min_prediction_samples=5
            )
            evaluator = ComprehensiveModelEvaluator(config)
            evaluation_results = await evaluator.evaluate_all_models()
            
            logger.info("✅ Evaluation completed successfully")
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            evaluation_results = {}
        
        # Step 3: Model promotion based on evaluation
        if auto_promote:
            logger.info("🎯 Step 3: Automated model promotion")
            await self.promote_models_based_on_performance(evaluation_results, results)
        
        # Step 4: Generate pipeline report
        await self.generate_pipeline_report(results, evaluation_results)
        
        return results, evaluation_results
    
    async def promote_models_based_on_performance(self, evaluation_results: dict, training_results: dict):
        """Automatically promote models based on performance criteria"""
        
        promotion_criteria = {
            "min_sharpe_ratio": 1.0,
            "min_directional_accuracy": 55.0,
            "max_drawdown_threshold": 15.0,
            "min_test_r2": 0.05
        }
        
        for symbol, performances in evaluation_results.items():
            if not performances:
                logger.warning(f"No evaluation results for {symbol}, skipping promotion")
                continue
                
            # Get the latest model performance
            latest_performance = performances[0] if performances else None
            if not latest_performance:
                continue
            
            model_name = f"{symbol}_predictor"
            
            try:
                # Get current model version
                versions = self.client.search_model_versions(f"name='{model_name}'")
                if not versions:
                    logger.warning(f"No model versions found for {model_name}")
                    continue
                
                latest_version = sorted(versions, key=lambda v: int(v.version))[-1]
                
                # Check promotion criteria
                should_promote = (
                    latest_performance.sharpe_ratio >= promotion_criteria["min_sharpe_ratio"] and
                    latest_performance.directional_accuracy >= promotion_criteria["min_directional_accuracy"] and
                    latest_performance.max_drawdown <= promotion_criteria["max_drawdown_threshold"] and
                    training_results.get(symbol, {}).get('test_score', 0) >= promotion_criteria["min_test_r2"]
                )
                
                if should_promote:
                    # Promote to Production
                    if latest_version.current_stage != "Production":
                        self.client.transition_model_version_stage(
                            name=model_name,
                            version=latest_version.version,
                            stage="Production",
                            archive_existing_versions=True
                        )
                        
                        # Also promote scaler
                        try:
                            scaler_versions = self.client.search_model_versions(f"name='{model_name}_scaler'")
                            if scaler_versions:
                                scaler_version = sorted(scaler_versions, key=lambda v: int(v.version))[-1]
                                self.client.transition_model_version_stage(
                                    name=f"{model_name}_scaler",
                                    version=scaler_version.version,
                                    stage="Production",
                                    archive_existing_versions=True
                                )
                        except Exception as e:
                            logger.warning(f"Could not promote scaler for {symbol}: {e}")
                        
                        logger.info(f"🎉 Promoted {model_name} v{latest_version.version} to Production")
                        logger.info(f"   Criteria: Sharpe={latest_performance.sharpe_ratio:.2f}, "
                                  f"Accuracy={latest_performance.directional_accuracy:.1f}%, "
                                  f"Drawdown={latest_performance.max_drawdown:.1f}%")
                    else:
                        logger.info(f"✅ {model_name} v{latest_version.version} already in Production")
                else:
                    logger.warning(f"⚠️  {model_name} v{latest_version.version} does not meet promotion criteria")
                    logger.info(f"   Current: Sharpe={latest_performance.sharpe_ratio:.2f}, "
                              f"Accuracy={latest_performance.directional_accuracy:.1f}%, "
                              f"Drawdown={latest_performance.max_drawdown:.1f}%")
                    
            except Exception as e:
                logger.error(f"❌ Failed to promote {model_name}: {e}")
    
    async def generate_pipeline_report(self, training_results: dict, evaluation_results: dict):
        """Generate comprehensive pipeline execution report"""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("AUTOMATED MLOPS PIPELINE EXECUTION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Pipeline Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Training Summary
        report_lines.append("📚 TRAINING RESULTS")
        report_lines.append("-" * 40)
        
        successful_training = 0
        for symbol, result in training_results.items():
            if 'error' not in result:
                successful_training += 1
                report_lines.append(f"✅ {symbol}: R² = {result.get('test_score', 0):.4f}, "
                                  f"MAE = {result.get('test_mae', 0):.6f}")
            else:
                report_lines.append(f"❌ {symbol}: {result['error']}")
        
        report_lines.append(f"\nTraining Success Rate: {successful_training}/{len(training_results)} models")
        report_lines.append("")
        
        # Evaluation Summary
        report_lines.append("📊 EVALUATION RESULTS")
        report_lines.append("-" * 40)
        
        for symbol, performances in evaluation_results.items():
            if performances:
                perf = performances[0]
                report_lines.append(f"📈 {symbol}:")
                report_lines.append(f"   Return: {perf.total_return:.1f}%, "
                                  f"Sharpe: {perf.sharpe_ratio:.2f}, "
                                  f"Accuracy: {perf.directional_accuracy:.1f}%")
            else:
                report_lines.append(f"⚠️  {symbol}: No evaluation data")
        
        report_lines.append("")
        
        # Production Readiness
        report_lines.append("🎯 PRODUCTION READINESS")
        report_lines.append("-" * 40)
        
        try:
            production_models = 0
            for symbol in training_results.keys():
                model_name = f"{symbol}_predictor"
                try:
                    versions = self.client.search_model_versions(f"name='{model_name}'")
                    production_versions = [v for v in versions if v.current_stage == "Production"]
                    if production_versions:
                        production_models += 1
                        report_lines.append(f"🚀 {symbol}: Production v{production_versions[0].version}")
                    else:
                        staging_versions = [v for v in versions if v.current_stage == "Staging"]
                        if staging_versions:
                            report_lines.append(f"🔄 {symbol}: Staging v{staging_versions[0].version}")
                        else:
                            report_lines.append(f"⚠️  {symbol}: No promoted models")
                except Exception:
                    report_lines.append(f"❓ {symbol}: Status unknown")
            
            report_lines.append(f"\nProduction Models: {production_models}/{len(training_results)}")
            
        except Exception as e:
            report_lines.append(f"❌ Could not assess production readiness: {e}")
        
        # Save and display report
        report_content = "\n".join(report_lines)
        
        with open("pipeline_execution_report.txt", "w") as f:
            f.write(report_content)
            
        logger.info("Generated pipeline execution report: pipeline_execution_report.txt")
        print("\n" + report_content)

async def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description="Run MLOps training pipeline")
    parser.add_argument("--symbols", nargs="*", default=["AAPL", "MSFT", "TSLA"],
                       help="Stock symbols to train models for")
    parser.add_argument("--no-promote", action="store_true",
                       help="Skip automatic model promotion")
    parser.add_argument("--mlflow-uri", default="http://localhost:5001",
                       help="MLflow tracking server URI")
    
    args = parser.parse_args()
    
    pipeline = MLflowTrainingPipeline(args.mlflow_uri)
    
    try:
        training_results, evaluation_results = await pipeline.run_training_pipeline(
            symbols=args.symbols,
            auto_promote=not args.no_promote
        )
        
        logger.info("🎉 Pipeline execution completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)