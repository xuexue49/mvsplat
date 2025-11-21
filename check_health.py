#!/usr/bin/env python3
"""
Health check script to verify system integrity after refactoring.

This script tests that all core components can be instantiated without errors,
ensuring the codebase is runnable after removing Hydra and WandB dependencies.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("🔍 Starting System Integrity Check...")
    print()
    
    # Test 1: Import core modules
    print("✓ Test 1: Importing core modules...")
    try:
        from src.main import ExperimentConfig, build_configs
        from src.dataset.data_module import DataModule
        from src.model.model_wrapper import ModelWrapper
        from src.factory import get_encoder, get_decoder, get_losses
        from src.misc.step_tracker import StepTracker
        print("  ✅ All imports successful")
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False
    
    # Test 2: Instantiate ExperimentConfig
    print("\n✓ Test 2: Creating ExperimentConfig...")
    try:
        cfg = ExperimentConfig()
        # Override with dummy paths to avoid file system dependencies
        cfg.dataset_path = "/tmp/dummy_dataset"
        cfg.output_dir = "/tmp/dummy_output"
        cfg.checkpoints_dir = "/tmp/dummy_checkpoints"
        print(f"  ✅ ExperimentConfig created with experiment_name='{cfg.experiment_name}'")
    except Exception as e:
        print(f"  ❌ ExperimentConfig creation failed: {e}")
        return False
    
    # Test 3: Build configuration objects
    print("\n✓ Test 3: Building configuration objects...")
    try:
        configs = build_configs(cfg)
        print(f"  ✅ Built {len(configs)} configuration objects:")
        for key in configs.keys():
            print(f"     - {key}")
    except Exception as e:
        print(f"  ❌ Configuration building failed: {e}")
        return False
    
    # Test 4: Instantiate DataModule
    print("\n✓ Test 4: Creating DataModule...")
    try:
        step_tracker = StepTracker()
        data_module = DataModule(
            configs['dataset_cfg'],
            configs['data_loader_cfg'],
            step_tracker,
            global_rank=0,
        )
        print("  ✅ DataModule created successfully")
    except Exception as e:
        print(f"  ❌ DataModule creation failed: {e}")
        return False
    
    # Test 5: Instantiate encoder and decoder (lightweight check)
    print("\n✓ Test 5: Creating encoder and decoder...")
    try:
        encoder, encoder_visualizer = get_encoder(configs['encoder_cfg'])
        print(f"  ✅ Encoder created: {type(encoder).__name__}")
        
        decoder = get_decoder(configs['decoder_cfg'], configs['dataset_cfg'])
        print(f"  ✅ Decoder created: {type(decoder).__name__}")
    except Exception as e:
        print(f"  ❌ Encoder/Decoder creation failed: {e}")
        return False
    
    # Test 6: Instantiate losses
    print("\n✓ Test 6: Creating loss functions...")
    try:
        losses = get_losses(configs['loss_cfgs'])
        print(f"  ✅ Created {len(losses)} loss function(s)")
    except Exception as e:
        print(f"  ❌ Loss creation failed: {e}")
        return False
    
    # Test 7: Instantiate ModelWrapper (without loading weights)
    print("\n✓ Test 7: Creating ModelWrapper...")
    try:
        model_wrapper = ModelWrapper(
            optimizer_cfg=configs['optimizer_cfg'],
            test_cfg=configs['test_cfg'],
            train_cfg=configs['train_cfg'],
            encoder=encoder,
            encoder_visualizer=encoder_visualizer,
            decoder=decoder,
            losses=losses,
            step_tracker=step_tracker,
        )
        print("  ✅ ModelWrapper created successfully")
    except Exception as e:
        print(f"  ❌ ModelWrapper creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # All tests passed
    print("\n" + "="*60)
    print("✅ System Integrity Check Passed")
    print("="*60)
    print("\nAll core components initialized successfully!")
    print("The codebase is ready for training/testing.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
