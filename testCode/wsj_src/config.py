import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for AE_MLP Training")

    # General settings
    parser.add_argument('--TEST', action='store_true', default=False, help="Enable test mode")
    parser.add_argument('--FULLTRAIN', action='store_true', default=False, help="Train all data mode")
    parser.add_argument('--usegpu', type=bool, default=True, help="Use GPU for training")
    parser.add_argument('--purgedCV', type=bool, default=True, help="Enable Purged Cross Validation")
    parser.add_argument('--gpuid', type=int, default=0, help="GPU ID to use")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")

    # Model settings
    parser.add_argument('--model', type=str, default='AE_MLP', help="Model type")
    parser.add_argument('--use_wandb', action='store_true', default=False, help="Use Weights & Biases for logging")
    parser.add_argument('--use_tb', type=bool, default=True, help="Use TensorBoard for logging")
    parser.add_argument('--project', type=str, default='AE_MLP-purgedcv-with-lags', help="Project name")

    # Paths
    parser.add_argument('--save_model_root', type=str, default="/root/autodl-tmp/JaneStreeReal2024/xdzy/models/", help="Directory to save models")
    parser.add_argument('--dname', type=str, default="./input_df/", help="Data directory")
    parser.add_argument('--tbroot', type=str, default="/root/tf-logs", help="TensorBoard log directory")

    # DataLoader settings
    parser.add_argument('--loader_workers', type=int, default=6, help="Number of workers for data loading")

    # Optimization settings
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="Weight decay for optimizer")
    parser.add_argument('--dropouts', type=float, nargs='+', default=[0.03527936123679956, 0.038424974585075086, 0.42409238408801436, 0.10431484318345882, 0.49230389137187497, 0.32024444956111164, 0.2716856145683449, 0.4379233941604448], help="Dropout rates")

    # Training/Test settings
    parser.add_argument('--bs', type=int, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--n_hidden', type=int, nargs='+', default=[96, 96, 896, 448, 448, 256], help="Hidden layer sizes")
    parser.add_argument('--patience', type=int, help="Early stopping patience")
    parser.add_argument('--max_epochs', type=int, default=2000, help="Maximum number of epochs")
    parser.add_argument('--N_fold', type=int, default=5, help="Number of folds for cross-validation")

    # Input/output paths
    parser.add_argument('--input_path', type=str, default='../', help="Input path")
 
    # Feature and label settings
    parser.add_argument('--feature_names', type=str, nargs='+', default=[f"feature_{i:02d}" for i in range(79)] + [f"responder_{idx}_lag_1" for idx in range(9)], help="Feature names")
    parser.add_argument('--label_name', type=str, default='responder_6', help="Label name")
    parser.add_argument('--weight_name', type=str, default='weight', help="Weight column name")

    # Time and data split settings
    parser.add_argument('--test_train_ratio', type=int, default=5, help="Test-train ratio")
    parser.add_argument('--time_col', type=str, default="date_id", help="Time column name")
    parser.add_argument('--group_gap', type=int, default=31, help="Group gap for data splitting")

    args = parser.parse_args()

    args.train_path = os.path.join(args.input_path, "training_data.parquet")
    args.valid_path = os.path.join(args.input_path, "validation_data.parquet")
    
    
    # Set defaults for TEST mode
    if args.TEST:
        args.bs = 8192
        args.lr = 1e-3
        args.patience = 5
        args.max_epochs = 2
        args.N_fold=2
        args.project += "-TEST"
    else:
        args.bs = 4 * 8192
        args.patience = 15

    if args.FULLTRAIN:
        args.project += "-FULLTRAIN"

    # Create directories if needed
    os.makedirs( os.path.join(args.save_model_root, args.project), exist_ok=True)
    args.save_model_root = os.path.join(args.save_model_root, args.project)
    return args

if __name__ == "__main__":
    config = parse_args()
    print(vars(config))