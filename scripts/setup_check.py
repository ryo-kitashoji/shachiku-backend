#!/usr/bin/env python3
"""
ShachikuAI セットアップ検証スクリプト

このスクリプトは環境が正しくセットアップされているかチェックします。
"""

import os
import sys
import platform
import subprocess
import importlib
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_status(message, status="info"):
    colors = {
        "ok": Colors.GREEN + "✓ ",
        "error": Colors.RED + "✗ ",
        "warning": Colors.YELLOW + "⚠ ",
        "info": Colors.BLUE + "ℹ "
    }
    print(f"{colors.get(status, '')}{message}{Colors.END}")

def check_python_version():
    """Python バージョンをチェック"""
    print(f"\n{Colors.BOLD}Python バージョンチェック{Colors.END}")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} (OK)", "ok")
        return True
    else:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} (Python 3.8+が必要)", "error")
        return False

def check_system_info():
    """システム情報を表示"""
    print(f"\n{Colors.BOLD}システム情報{Colors.END}")
    print_status(f"OS: {platform.system()} {platform.release()}", "info")
    print_status(f"アーキテクチャ: {platform.machine()}", "info")
    
    # メモリ情報（Linuxの場合）
    if platform.system() == "Linux":
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal'):
                        mem_total = int(line.split()[1]) // 1024  # MB
                        print_status(f"総メモリ: {mem_total} MB", "info")
                        break
        except:
            print_status("メモリ情報の取得に失敗", "warning")

def check_dependencies():
    """主要な依存関係をチェック"""
    print(f"\n{Colors.BOLD}依存関係チェック{Colors.END}")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("fastapi", "FastAPI"), 
        ("uvicorn", "Uvicorn"),
        ("pydantic", "Pydantic"),
        ("numpy", "NumPy"),
        ("requests", "Requests")
    ]
    
    all_ok = True
    for package, name in required_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print_status(f"{name}: {version}", "ok")
        except ImportError:
            print_status(f"{name}: インストールされていません", "error")
            all_ok = False
    
    return all_ok

def check_cuda():
    """CUDA環境をチェック"""
    print(f"\n{Colors.BOLD}CUDA環境チェック{Colors.END}")
    
    try:
        import torch
        if torch.cuda.is_available():
            print_status(f"CUDA利用可能", "ok")
            print_status(f"CUDA バージョン: {torch.version.cuda}", "info")
            print_status(f"GPU デバイス数: {torch.cuda.device_count()}", "info")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print_status(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)", "info")
        else:
            print_status("CUDA利用不可（CPUモードで動作）", "warning")
    except ImportError:
        print_status("PyTorchがインストールされていません", "error")

def check_environment_file():
    """環境設定ファイルをチェック"""
    print(f"\n{Colors.BOLD}環境設定ファイルチェック{Colors.END}")
    
    env_file = Path(".env")
    if env_file.exists():
        print_status(".envファイルが存在します", "ok")
        
        # 重要な設定項目をチェック
        with open(env_file, 'r') as f:
            content = f.read()
            
        required_vars = [
            "MODEL_NAME",
            "MODEL_PATH", 
            "API_HOST",
            "API_PORT"
        ]
        
        for var in required_vars:
            if var in content:
                print_status(f"{var}: 設定済み", "ok")
            else:
                print_status(f"{var}: 未設定", "warning")
    else:
        print_status(".envファイルが存在しません", "error")
        return False
    
    return True

def check_directories():
    """必要なディレクトリをチェック"""
    print(f"\n{Colors.BOLD}ディレクトリ構造チェック{Colors.END}")
    
    required_dirs = [
        "data",
        "data/models",
        "data/training",
        "logs"
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print_status(f"{dir_path}: 存在", "ok")
        else:
            print_status(f"{dir_path}: 作成します", "warning")
            try:
                path.mkdir(parents=True, exist_ok=True)
                print_status(f"{dir_path}: 作成完了", "ok")
            except Exception as e:
                print_status(f"{dir_path}: 作成失敗 - {e}", "error")
                all_ok = False
    
    return all_ok

def check_network():
    """ネットワーク接続をチェック"""
    print(f"\n{Colors.BOLD}ネットワーク接続チェック{Colors.END}")
    
    try:
        import urllib.request
        urllib.request.urlopen('https://huggingface.co', timeout=10)
        print_status("Hugging Face接続: OK", "ok")
        return True
    except Exception as e:
        print_status(f"Hugging Face接続: 失敗 - {e}", "error")
        return False

def check_docker():
    """Docker環境をチェック"""
    print(f"\n{Colors.BOLD}Docker環境チェック{Colors.END}")
    
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print_status(f"Docker: {result.stdout.strip()}", "ok")
        else:
            print_status("Docker: インストールされていません", "warning")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_status("Docker: インストールされていません", "warning")
    
    try:
        result = subprocess.run(['docker-compose', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print_status(f"Docker Compose: {result.stdout.strip()}", "ok")
        else:
            print_status("Docker Compose: インストールされていません", "warning")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_status("Docker Compose: インストールされていません", "warning")

def main():
    """メイン関数"""
    print(f"{Colors.BOLD}ShachikuAI セットアップ検証{Colors.END}")
    print("=" * 50)
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_environment_file(),
        check_directories(),
        check_network()
    ]
    
    # システム情報とオプショナルチェック
    check_system_info()
    check_cuda()
    check_docker()
    
    # 結果サマリー
    print(f"\n{Colors.BOLD}検証結果サマリー{Colors.END}")
    print("=" * 30)
    
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print_status(f"すべてのチェックが完了しました ({passed}/{total})", "ok")
        print_status("ShachikuAIを起動する準備ができています！", "ok")
        print(f"\n{Colors.BLUE}起動コマンド:{Colors.END}")
        print("  python main.py")
        print(f"  または")
        print("  docker-compose up --build")
    else:
        print_status(f"一部のチェックで問題があります ({passed}/{total})", "warning")
        print_status("SETUP.mdを参照して問題を解決してください", "info")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)