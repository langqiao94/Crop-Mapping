import os
import sys
import yaml
import shutil
import subprocess
import time
import pandas as pd
from datetime import datetime
from pathlib import Path


class RandomForestBatchRunner:
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
        
        self.config_file = Path('config/dataset_config.yaml')
        self.backup_file = Path('config/dataset_config.yaml.backup')
        self.python_exe = sys.executable
        
        self.results_summary = []
        
        self.log_file = f'rf_batch_run_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        print(f"\n{'='*100}")
        print(f"Random Forest Batch Runner")
        print(f"{'='*100}")
        print(f"Python: {self.python_exe}")
        print(f"Datasets: {len(self.datasets)}")
        print(f"Log file: {self.log_file}")
        print(f"{'='*100}\n")
    
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
            
            config['dataset']['image_root'] = f'Dataset/HLS/{folder}'
            config['data_info']['hls_channels'] = channels
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
            
            self.log(f"Config updated: image_root=Dataset/HLS/{folder}, hls_channels={channels}")
            return True
            
        except Exception as e:
            self.log(f"Failed to modify config file: {e}")
            return False
    
    def run_command(self, command, description):
        self.log(f"\n{'='*100}")
        self.log(f"Starting: {description}")
        self.log(f"Command: {command}")
        self.log(f"{'='*100}\n")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=False
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
    
    def extract_metrics(self, folder):
        metrics_file = Path(f'results/random_forest/combined_{folder}/metrics.txt')
        
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
            
            self.log(f"Successfully extracted metrics: Dataset {folder}")
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
        save_name = f'combined_{folder}'
        
        if not self.modify_config(folder, channels):
            return False
        
        train_cmd = (
            f"{self.python_exe} main.py train "
            f"--model random_forest "
            f"--save-name {save_name} "
            f"--train-file Dataset/202201_train.txt "
            f"--val-file Dataset/202201_val.txt "
            f"--test-file Dataset/202201_test.txt"
        )
        
        success, train_time = self.run_command(
            train_cmd,
            f"Train Random Forest - Dataset {folder}"
        )
        
        if not success:
            self.log(f"Training failed, skipping test")
            return False
        
        test_cmd = (
            f"{self.python_exe} main.py test "
            f"--model random_forest "
            f"--checkpoint checkpoints/random_forest/model_best_{save_name}.pkl "
            f"--test-file Dataset/2022011_test.txt "
            f"--save-name {save_name}"
        )
        
        success, test_time = self.run_command(
            test_cmd,
            f"Test Random Forest - Dataset {folder}"
        )
        
        if not success:
            self.log(f"Test failed")
            return False
        
        metrics = self.extract_metrics(folder)
        if metrics:
            metrics['Train Time (min)'] = train_time / 60
            metrics['Test Time (min)'] = test_time / 60
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
            
            columns_order.extend(['Train Time (min)', 'Test Time (min)'])
            
            existing_columns = [col for col in columns_order if col in df.columns]
            df = df[existing_columns]
            
            output_file = Path('results/random_forest/all_experiments_summary.xlsx')
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
        self.log(f"Starting batch run for Random Forest model")
        self.log(f"Starting batch run for Random Forest model")
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
            self.log(f"  - results/random_forest/combined_{folder}/")
        self.log(f"  - results/random_forest/all_experiments_summary.xlsx")
        
        self.log(f"\nCheckpoint locations:")
        self.log(f"Checkpoint locations:")
        for folder in self.datasets.keys():
            self.log(f"  - checkpoints/random_forest/model_best_combined_{folder}.pkl")
        
        self.log(f"{'#'*100}\n")


def main():
    try:
        runner = RandomForestBatchRunner()
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
