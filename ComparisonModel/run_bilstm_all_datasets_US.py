import os
import sys
import yaml
import shutil
import subprocess
import time
import pandas as pd
from datetime import datetime
from pathlib import Path


class BiLSTMBatchRunner:
    def __init__(self):
        self.datasets = {
            6: 11,
            12: 17,
            18: 23,
            24: 29,
            30: 35,
            36: 41,
            42: 47,
            48: 53,
            54: 59,
            60: 65,
            66: 71,
            72: 77
        }
        
        self.original_config_file = Path('config/dataset_config.yaml')
        self.config_file = Path('config/dataset_config_bilstm.yaml')
        self.backup_file = Path('config/dataset_config_bilstm.yaml.backup')
        self.python_exe = sys.executable
        
        self.log_file = f'bilstm_batch_run_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        self._init_config_file()
        
        self.epochs = 3
        
        self.historical_train = 'Dataset/train.txt'
        self.historical_val = 'Dataset/val.txt'
        self.historical_test = 'Dataset/test.txt'
        
        self.finetuned_train = 'Dataset/202201_train.txt'
        self.finetuned_val = 'Dataset/202201_val.txt'
        self.finetuned_test = 'Dataset/202201_test.txt'
        
        self.results_summary = []
        
        print(f"\n{'='*100}")
        print(f"BiLSTM Batch Runner (Two-Stage Training, Pixel-Level Classification)")
        print(f"{'='*100}")
        print(f"Python: {self.python_exe}")
        print(f"Datasets: {len(self.datasets)}")
        print(f"Epochs: {self.epochs} epochs (per stage)")
        print(f"Log file: {self.log_file}")
        print(f"{'='*100}\n")
    
    def _init_config_file(self):
        if not self.config_file.exists():
            if self.original_config_file.exists():
                shutil.copy2(self.original_config_file, self.config_file)
                self.log(f"Created independent config file: {self.config_file} (copied from original)")
            else:
                raise FileNotFoundError(f"Original config file not found: {self.original_config_file}")
    
    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def backup_config(self):
        try:
            shutil.copy2(self.config_file, self.backup_file)
            self.log(f"Config file backed up: {self.backup_file}")
            return True
        except Exception as e:
            self.log(f"Failed to backup config file: {e}")
            return False
    
    def restore_config(self):
        try:
            if self.backup_file.exists():
                shutil.copy2(self.backup_file, self.config_file)
                self.log(f"Config file restored: {self.config_file}")
                self.backup_file.unlink()
                return True
        except Exception as e:
            self.log(f"Failed to restore config file: {e}")
            return False
    
    def modify_config(self, folder, channels):
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if 'dataset' not in config:
                config['dataset'] = {}
            if 'data_info' not in config:
                config['data_info'] = {}
            
            config['dataset']['image_root'] = f'Dataset/HLS/{folder}'
            config['data_info']['hls_channels'] = channels
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            
            self.log(f"Config updated: image_root=Dataset/HLS/{folder}, hls_channels={channels}")
            return True
            
        except Exception as e:
            self.log(f"Failed to modify config file: {e}")
            import traceback
            self.log(f"Detailed error: {traceback.format_exc()}")
            return False
    
    def run_command(self, command, description):
        self.log(f"\n{'='*100}")
        self.log(f"Starting: {description}")
        self.log(f"Command: {command}")
        self.log(f"{'='*100}\n")
        
        start_time = time.time()
        
        try:
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
            except NameError:
                script_dir = os.getcwd()
                if not os.path.exists(os.path.join(script_dir, 'main.py')):
                    if 'ComparisonModel' in script_dir:
                        script_dir = script_dir
                    else:
                        script_dir = os.path.join(script_dir, 'ComparisonModel') if os.path.exists(os.path.join(script_dir, 'ComparisonModel', 'main.py')) else script_dir
            
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=False,
                cwd=script_dir
            )
            
            elapsed_time = time.time() - start_time
            self.log(f"\n{'='*100}")
            self.log(f"Success: {description} (Time: {elapsed_time:.2f}s / {elapsed_time/60:.2f}min)")
            self.log(f"{'='*100}\n")
            
            return True, elapsed_time
            
        except subprocess.CalledProcessError as e:
            elapsed_time = time.time() - start_time
            self.log(f"\n{'='*100}")
            self.log(f"Failed: {description} (Time: {elapsed_time:.2f}s)")
            self.log(f"Error code: {e.returncode}")
            self.log(f"{'='*100}\n")
            
            return False, elapsed_time
    
    def extract_metrics(self, folder, stage='finetuned'):
        metrics_file = Path(f'results/bilstm/{stage}_{folder}/metrics.txt')
        
        if not metrics_file.exists():
            self.log(f"Metrics file not found: {metrics_file}")
            return None
        
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            metrics = {'Dataset': folder}
            
            in_overall_section = False
            for line in lines:
                line = line.strip()
                
                if 'Overall Metrics:' in line:
                    in_overall_section = True
                    continue
                elif 'Per-Class Metrics:' in line:
                    in_overall_section = False
                    continue
                
                if in_overall_section and ':' in line:
                    if 'Accuracy:' in line:
                        value = line.split(':')[1].strip().replace('%', '')
                        metrics['Overall OA'] = float(value)
                    elif 'Kappa:' in line:
                        value = line.split(':')[1].strip()
                        metrics['Overall Kappa'] = float(value)
                    elif 'F1 Macro:' in line:
                        value = line.split(':')[1].strip().replace('%', '')
                        metrics['Overall F1 Macro'] = float(value)
            
            class_mapping = {
                'other': 'Other',
                'corn': 'Corn',
                'cotton': 'Cotton',
                'soybeans': 'Soybeans',
                'spring wheat': 'Spring Wheat',
                'winter wheat': 'Winter Wheat'
            }
            
            in_perclass_section = False
            for line in lines:
                line_stripped = line.strip()
                
                if 'Per-Class Metrics:' in line_stripped:
                    in_perclass_section = True
                    continue
                
                if in_perclass_section and ('---' in line_stripped or 
                                            'Class' in line_stripped and 'IoU' in line_stripped):
                    continue
                
                if not line_stripped or '===' in line_stripped:
                    continue
                
                if in_perclass_section:
                    parts = line_stripped.split()
                    
                    if len(parts) >= 6:
                        if parts[0] == 'spring' and len(parts) >= 7:
                            class_name = 'spring wheat'
                            values = parts[2:]
                        elif parts[0] == 'winter' and len(parts) >= 7:
                            class_name = 'winter wheat'
                            values = parts[2:]
                        else:
                            class_name = parts[0]
                            values = parts[1:]
                        
                        if class_name in class_mapping:
                            display_name = class_mapping[class_name]
                            
                            if len(values) >= 5:
                                prec = float(values[2].replace('%', ''))
                                recall = float(values[3].replace('%', ''))
                                f1 = float(values[4].replace('%', ''))
                                
                                metrics[f'{display_name} Precision'] = prec
                                metrics[f'{display_name} Recall'] = recall
                                metrics[f'{display_name} F1'] = f1
            
            self.log(f"Successfully extracted metrics: Dataset {folder} ({stage})")
            self.log(f"   Extracted {len(metrics)} metric values")
            return metrics
            
        except Exception as e:
            self.log(f"Failed to extract metrics: {e}")
            import traceback
            self.log(f"   Detailed error: {traceback.format_exc()}")
            return None
    
    def run_single_dataset(self, folder, channels):
        self.log(f"\n{'#'*100}")
        self.log(f"Processing Dataset: {folder} (channels: {channels})")
        self.log(f"Processing Dataset: {folder} channels (HLS channels: {channels})")
        self.log(f"{'#'*100}\n")
        
        dataset_start = time.time()
        
        historical_name = f'historical_{folder}'
        finetuned_name = f'finetuned_{folder}'
        
        if not self.modify_config(folder, channels):
            return False
        
        self.log(f"\n{'='*100}")
        self.log(f"Stage 1/3: Train historical classifier (2020-2021 data)")
        self.log(f"Stage 1/3: Train historical classifier (2020-2021 data)")
        self.log(f"{'='*100}\n")
        
        historical_train_cmd = (
            f"{self.python_exe} main.py train "
            f"--model bilstm "
            f"--epochs {self.epochs} "
            f"--save-name {historical_name} "
            f"--save-last-only "
            f"--train-file {self.historical_train} "
            f"--val-file {self.historical_val} "
            f"--test-file {self.historical_test} "
            f"--config-file {self.config_file}"
        )
        
        success, historical_train_time = self.run_command(
            historical_train_cmd,
            f"Train historical classifier - Dataset {folder}"
        )
        
        if not success:
            self.log(f"Stage 1 training failed, skipping subsequent steps")
            return False
        
        self.log(f"\n{'='*100}")
        self.log(f"Stage 2/3: Fine-tune model (2022 data)")
        self.log(f"Stage 2/3: Fine-tune model (2022 data)")
        self.log(f"{'='*100}\n")
        
        historical_checkpoint = f"checkpoints/bilstm/model_last_{historical_name}.pth"
        
        finetuned_train_cmd = (
            f"{self.python_exe} main.py train "
            f"--model bilstm "
            f"--epochs {self.epochs} "
            f"--save-name {finetuned_name} "
            f"--save-last-only "
            f"--resume {historical_checkpoint} "
            f"--train-file {self.finetuned_train} "
            f"--val-file {self.finetuned_val} "
            f"--test-file {self.finetuned_test} "
            f"--config-file {self.config_file}"
        )
        
        success, finetuned_train_time = self.run_command(
            finetuned_train_cmd,
            f"Fine-tune model - Dataset {folder}"
        )
        
        if not success:
            self.log(f"Stage 2 fine-tuning failed, skipping test")
            return False
        
        self.log(f"\n{'='*100}")
        self.log(f"Stage 3/3: Test fine-tuned model")
        self.log(f"Stage 3/3: Test fine-tuned model")
        self.log(f"{'='*100}\n")
        
        finetuned_checkpoint = f"checkpoints/bilstm/model_last_{finetuned_name}.pth"
        
        test_cmd = (
            f"{self.python_exe} main.py test "
            f"--model bilstm "
            f"--checkpoint {finetuned_checkpoint} "
            f"--test-file {self.finetuned_test} "
            f"--save-name {finetuned_name} "
            f"--config-file {self.config_file}"
        )
        
        success, test_time = self.run_command(
            test_cmd,
            f"Test fine-tuned model - Dataset {folder}"
        )
        
        if not success:
            self.log(f"Test failed")
            return False
        
        metrics = self.extract_metrics(folder, stage='finetuned')
        if metrics:
            metrics['Historical Train Time (min)'] = historical_train_time / 60
            metrics['Finetune Train Time (min)'] = finetuned_train_time / 60
            metrics['Test Time (min)'] = test_time / 60
            metrics['Total Time (min)'] = (historical_train_time + finetuned_train_time + test_time) / 60
            self.results_summary.append(metrics)
        
        dataset_elapsed = time.time() - dataset_start
        self.log(f"\n{'#'*100}")
        self.log(f"Dataset {folder} completed (Total time: {dataset_elapsed:.2f}s / {dataset_elapsed/60:.2f}min)")
        self.log(f"{'#'*100}\n")
        
        return True
    
    def generate_excel_summary(self):
        if not self.results_summary:
            self.log("No results to generate Excel")
            return False
        
        try:
            df = pd.DataFrame(self.results_summary)
            
            columns_order = ['Dataset', 'Overall OA', 'Overall Kappa', 'Overall F1 Macro']
            
            class_names = ['Corn', 'Cotton', 'Soybeans', 'Spring Wheat', 'Winter Wheat', 'Other']
            for cls in class_names:
                columns_order.extend([f'{cls} F1', f'{cls} Precision', f'{cls} Recall'])
            
            columns_order.extend(['Historical Train Time (min)', 'Finetune Train Time (min)', 
                                'Test Time (min)', 'Total Time (min)'])
            
            existing_columns = [col for col in columns_order if col in df.columns]
            df = df[existing_columns]
            
            output_file = Path('results/bilstm/all_experiments_summary.xlsx')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_excel(output_file, index=False, engine='openpyxl')
            
            self.log(f"\n{'='*100}")
            self.log(f"Excel summary generated: {output_file}")
            self.log(f"{'='*100}\n")
            
            return True
            
        except Exception as e:
            self.log(f"Failed to generate Excel: {e}")
            return False
    
    def run_all(self):
        self.log(f"\n{'#'*100}")
        self.log(f"Starting batch run for BiLSTM model (Two-Stage Training, Pixel-Level)")
        self.log(f"Starting batch run for BiLSTM model (Two-Stage Training, Pixel-Level)")
        self.log(f"Datasets: {list(self.datasets.keys())}")
        self.log(f"{'#'*100}\n")
        
        overall_start = time.time()
        
        if not self.backup_config():
            self.log("Failed to backup config, terminating")
            return
        
        success_count = 0
        failed_datasets = []
        
        for folder, channels in self.datasets.items():
            success = self.run_single_dataset(folder, channels)
            if success:
                success_count += 1
            else:
                failed_datasets.append(folder)
        
        self.restore_config()
        
        self.generate_excel_summary()
        
        overall_elapsed = time.time() - overall_start
        
        self.log(f"\n{'#'*100}")
        self.log(f"Batch run completed!")
        self.log(f"Batch run completed!")
        self.log(f"{'#'*100}")
        self.log(f"Total time: {overall_elapsed:.2f}s ({overall_elapsed/60:.2f}min / {overall_elapsed/3600:.2f}hours)")
        self.log(f"Success: {success_count}/{len(self.datasets)}")
        
        if failed_datasets:
            self.log(f"Failed datasets: {failed_datasets}")
        
        self.log(f"\nResults locations:")
        self.log(f"Results locations:")
        for folder in self.datasets.keys():
            self.log(f"  - results/bilstm/finetuned_{folder}/")
        self.log(f"  - results/bilstm/all_experiments_summary.xlsx")
        
        self.log(f"\nCheckpoint locations:")
        self.log(f"Checkpoint locations:")
        for folder in self.datasets.keys():
            self.log(f"  - checkpoints/bilstm/model_last_historical_{folder}.pth")
            self.log(f"  - checkpoints/bilstm/model_last_finetuned_{folder}.pth")
        
        self.log(f"{'#'*100}\n")


def main():
    try:
        runner = BiLSTMBatchRunner()
        runner.run_all()
        
    except KeyboardInterrupt:
        print("\n\nUser interrupted execution")
        print("User interrupted execution")
        
        backup_file = Path('config/dataset_config.yaml.backup')
        config_file = Path('config/dataset_config.yaml')
        if backup_file.exists():
            shutil.copy2(backup_file, config_file)
            backup_file.unlink()
            print("Config file restored")
        
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        print(f"Error occurred: {e}")
        
        backup_file = Path('config/dataset_config.yaml.backup')
        config_file = Path('config/dataset_config.yaml')
        if backup_file.exists():
            shutil.copy2(backup_file, config_file)
            backup_file.unlink()
            print("Config file restored")
        
        sys.exit(1)


if __name__ == '__main__':
    main()
