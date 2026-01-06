import os
import sys
import yaml
import argparse
import torch
from pathlib import Path

from utils import Trainer
from utils.checkpoint import load_checkpoint

from Transformer import build_crit_transformer


def load_model_config(model_name, dataset_config_path=None):
    model_config_path = Path(f'config/model_configs/{model_name}_config.yaml')
    if not model_config_path.exists():
        raise FileNotFoundError(f"Config not found: {model_config_path}")
    
    with open(model_config_path, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)
    
    if dataset_config_path is None:
        dataset_config_path = model_config.get('dataset_config', 'config/dataset_config.yaml')
    dataset_config_path = Path(dataset_config_path)
    
    if not dataset_config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")
    
    with open(dataset_config_path, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    
    data_info = dataset_config.get('data_info', {})
    shared_training = dataset_config.get('training', {})
    
    if 'model' not in model_config:
        model_config['model'] = {}
    
    model_config['model']['in_channels'] = data_info.get('hls_channels', 42)
    model_config['model']['num_classes'] = data_info.get('num_classes', 6)
    model_config['model']['img_size'] = data_info.get('image_size', 128)
    
    if 'training' not in model_config:
        model_config['training'] = {}
    
    model_config['training'].update(shared_training)
    
    model_config['class_names'] = data_info.get('class_names', None)
    
    model_config['dataset_config'] = str(dataset_config_path)
    
    return model_config


def create_model(model_name, config):
    if model_name == 'transformer':
        model_config = config['model']
        model = build_crit_transformer(model_config)
        return model
    
    elif model_name == 'bilstm':
        from BiLSTM import build_bilstm
        model_config = config['model']
        model = build_bilstm(model_config)
        return model
    
    elif model_name == 'bayesian_mlp':
        from BayesianMLP import build_bayesian_cnn
        model_config = config['model']
        model = build_bayesian_cnn(model_config)
        return model
    
    elif model_name == 'random_forest':
        from RandomForest import build_random_forest
        model_config = config['model']
        model = build_random_forest(model_config)
        return model
    
    elif model_name == 'lstm':
        raise NotImplementedError("LSTM model not implemented yet")
    
    elif model_name == 'cnn':
        raise NotImplementedError("CNN model not implemented yet")
    
    elif model_name == 'resnet':
        raise NotImplementedError("ResNet model not implemented yet")
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train(args):
    print("\n" + "="*80)
    print(f"Training {args.model.upper()} Model")
    print("="*80)
    
    dataset_config_path = getattr(args, 'config_file', None)
    config = load_model_config(args.model, dataset_config_path=dataset_config_path)
    
    train_config = {
        **config.get('training', {}),
        'dataset_config': config.get('dataset_config', 'config/dataset_config.yaml'),
        'checkpoint_dir': config.get('checkpoint_dir', f'checkpoints/{args.model}'),
        'log_dir': config.get('log_dir', f'logs/{args.model}'),
        'class_names': config.get('class_names', None),
        'num_classes': config['model'].get('num_classes', 6)
    }
    
    if 'training_params' in config:
        train_config.update(config['training_params'])
    
    if args.batch_size:
        train_config['batch_size'] = args.batch_size
    if args.epochs:
        train_config['epochs'] = args.epochs
    if args.lr:
        train_config['lr'] = args.lr
    
    if args.train_file or args.val_file or args.test_file:
        train_config['file_list_override'] = {
            'train': args.train_file,
            'val': args.val_file,
            'test': args.test_file
        }
        if args.train_file:
            print(f"  Using custom train file: {args.train_file}")
        if args.val_file:
            print(f"  Using custom val file: {args.val_file}")
        if args.test_file:
            print(f"  Using custom test file: {args.test_file}")
    
    print(f"\nBuilding {args.model} model...")
    model = create_model(args.model, config)
    
    if args.model == 'random_forest':
        from utils.rf_trainer import RandomForestTrainer
        
        print(f"\nRandom Forest does not use GPU or gradient-based optimization.")
        
        trainer = RandomForestTrainer(
            model=model,
            config=train_config,
            model_name=args.model,
            device='cpu',
            save_name=args.save_name
        )
        
        if args.resume:
            print("Warning: --resume is not supported for Random Forest. Ignoring.")
        
        trainer.train()
        
        print("\n" + "="*80)
        print("Training completed successfully!")
        print("="*80)
        
        return
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    save_last_only = getattr(args, 'save_last_only', False)
    trainer = Trainer(
        model=model,
        config=train_config,
        model_name=args.model,
        device=device,
        save_name=args.save_name,
        save_last_only=save_last_only
    )
    
    if args.resume:
        print(f"\n{'='*80}")
        print(f"Loading pretrained weights from {args.resume}")
        print(f"{'='*80}")
        checkpoint = load_checkpoint(args.resume, model)
        print(f"Pretrained weights loaded successfully!")
        print(f"Starting training from epoch 0 with current learning rate: {train_config.get('lr', 0.0002)}")
        print(f"{'='*80}\n")
    
    trainer.train()
    
    print("\nTraining completed!")


def test(args):
    print("\n" + "="*80)
    print(f"Testing {args.model.upper()} Model")
    print("="*80)
    
    if not args.checkpoint:
        raise ValueError("Please provide checkpoint path with --checkpoint")
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    dataset_config_path = getattr(args, 'config_file', None)
    config = load_model_config(args.model, dataset_config_path=dataset_config_path)
    
    if args.save_name:
        results_dir = f'results/{args.model}/{args.save_name}'
        print(f"  Results will be saved to: {results_dir}")
    else:
        results_dir = config.get('results_dir', f'results/{args.model}')
    
    test_config = {
        'dataset_config': config.get('dataset_config', 'config/dataset_config.yaml'),
        'results_dir': results_dir,
        'batch_size': config.get('training', {}).get('batch_size', 24),
        'num_workers': config.get('training', {}).get('num_workers', 4),
        'class_names': config.get('class_names', None),
        'num_classes': config['model'].get('num_classes', 6)
    }
    
    if args.batch_size:
        test_config['batch_size'] = args.batch_size
    
    if args.test_file:
        test_config['file_list_override'] = {
            'train': None,
            'val': None,
            'test': args.test_file
        }
        print(f"  Using custom test file: {args.test_file}")
    
    print(f"\nBuilding {args.model} model...")
    model = create_model(args.model, config)
    
    if args.model == 'random_forest':
        from utils.rf_trainer import RandomForestTrainer
        
        print(f"\nRandom Forest model")
        
        print(f"\nLoading model from {args.checkpoint}...")
        model.load(args.checkpoint)
        
        trainer = RandomForestTrainer(
            model=model,
            config=test_config,
            model_name=args.model,
            device='cpu'
        )
        
        metrics = trainer.test(test_file=args.test_file, save_name=args.save_name)
        
        print("\n" + "="*80)
        print("Testing completed successfully!")
        print("="*80)
        
        return
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    load_checkpoint(args.checkpoint, model)
    
    trainer = Trainer(
        model=model,
        config=test_config,
        model_name=args.model,
        device=device
    )
    
    metrics = trainer.test()
    
    print("\nTesting completed!")
    print(f"Final Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Final mIoU: {metrics['miou']*100:.2f}%")


def compare(args):
    print("\n" + "="*80)
    print("Model Comparison")
    print("="*80)
    
    print("\nModel comparison not implemented yet!")
    print("Run each model with 'test' command first, then implement comparison.")


def main():
    parser = argparse.ArgumentParser(
        description='Train and test comparison models for crop type mapping',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--model', type=str, required=True,
                             choices=['transformer', 'bilstm', 'bayesian_mlp', 'random_forest', 'lstm', 'cnn', 'resnet'],
                             help='Model to train')
    train_parser.add_argument('--batch-size', type=int, default=None,
                             help='Batch size (overrides config)')
    train_parser.add_argument('--epochs', type=int, default=None,
                             help='Number of epochs (overrides config)')
    train_parser.add_argument('--lr', type=float, default=None,
                             help='Learning rate (overrides config)')
    train_parser.add_argument('--resume', type=str, default=None,
                             help='Resume from checkpoint (path to .pth file)')
    train_parser.add_argument('--train-file', type=str, default=None,
                             help='Training file list (overrides config)')
    train_parser.add_argument('--val-file', type=str, default=None,
                             help='Validation file list (overrides config)')
    train_parser.add_argument('--test-file', type=str, default=None,
                             help='Test file list (overrides config)')
    train_parser.add_argument('--save-name', type=str, default=None,
                             help='Custom name for saved model (e.g., "historical", "finetuned")')
    train_parser.add_argument('--save-last-only', action='store_true',
                             help='Only save the last checkpoint (not best model)')
    train_parser.add_argument('--device', type=str, default='cuda',
                             choices=['cuda', 'cpu'],
                             help='Device to use')
    train_parser.add_argument('--config-file', type=str, default=None,
                             help='Path to dataset config file (for batch scripts to use independent configs)')
    
    test_parser = subparsers.add_parser('test', help='Test a model')
    test_parser.add_argument('--model', type=str, required=True,
                            choices=['transformer', 'bilstm', 'bayesian_mlp', 'random_forest', 'lstm', 'cnn', 'resnet'],
                            help='Model to test')
    test_parser.add_argument('--checkpoint', type=str, required=True,
                            help='Path to model checkpoint')
    test_parser.add_argument('--batch-size', type=int, default=None,
                            help='Batch size (overrides config)')
    test_parser.add_argument('--test-file', type=str, default=None,
                            help='Test file list (overrides config)')
    test_parser.add_argument('--save-name', type=str, default=None,
                            help='Custom name for test results folder (e.g., "historical", "finetuned")')
    test_parser.add_argument('--device', type=str, default='cuda',
                            choices=['cuda', 'cpu'],
                            help='Device to use')
    test_parser.add_argument('--config-file', type=str, default=None,
                            help='Path to dataset config file (for batch scripts to use independent configs)')
    
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument('--models', type=str, nargs='+',
                               default=['transformer'],
                               help='Models to compare')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'test':
        test(args)
    elif args.command == 'compare':
        compare(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
