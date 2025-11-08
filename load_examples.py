#!/usr/bin/env python3
"""
Script to load example documents into the RAG system.

This script uploads all example files from the data/ directory
to the RAG API for indexing and vector storage.

Usage:
    python load_examples.py
"""

import requests
import os
from pathlib import Path
import sys


def load_example_files(api_url: str = "http://localhost:8000"):
    """
    Load all example files from data/ directory into the RAG system.

    Args:
        api_url: Base URL of the RAG API (default: http://localhost:8000)
    """
    # Check if API is running
    try:
        health_response = requests.get(f"{api_url}/health", timeout=5)
        if health_response.status_code != 200:
            print(f"[ERROR] API is not healthy. Status: {health_response.status_code}")
            return False
        print(f"[OK] API is running at {api_url}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Cannot connect to API at {api_url}")
        print(f"   Error: {e}")
        print(f"\n   Please start the API server first:")
        print(f"   uvicorn app.main:app --reload")
        return False

    # Get example files
    data_dir = Path("data")
    example_files = [
        "example_sales.xlsx",
        "example_products.xlsx",
        "example_report.docx",
        "example_instructions.docx"
    ]

    print(f"\nLoading example files from {data_dir}/")
    print("=" * 80)

    success_count = 0
    total_chunks = 0

    for filename in example_files:
        filepath = data_dir / filename

        if not filepath.exists():
            print(f"[WARNING] {filename} - File not found, skipping")
            continue

        print(f"\n[UPLOAD] {filename}...")

        try:
            with open(filepath, 'rb') as f:
                files = {'file': (filename, f)}
                response = requests.post(
                    f"{api_url}/api/v1/ingest",
                    files=files,
                    timeout=120  # 2 minutes timeout for large files
                )

            if response.status_code == 201:
                result = response.json()
                chunks = result.get('chunks_stored', 0)
                file_type = result.get('file_type', 'unknown')

                print(f"   [OK] Successfully loaded ({file_type})")
                print(f"   [INFO] {chunks} chunks created and stored")

                success_count += 1
                total_chunks += chunks
            else:
                print(f"   [ERROR] Failed to upload: {response.status_code}")
                print(f"   Response: {response.text[:200]}")

        except requests.exceptions.Timeout:
            print(f"   [ERROR] Upload timeout (file too large or server busy)")
        except Exception as e:
            print(f"   [ERROR] {e}")

    print("\n" + "=" * 80)
    print(f"[SUCCESS] Loaded {success_count}/{len(example_files)} files")
    print(f"[INFO] Total chunks indexed: {total_chunks}")

    # Verify with health check
    try:
        health_response = requests.get(f"{api_url}/api/v1/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            points = health_data.get('services', {}).get('vector_store_points', '0')
            print(f"[INFO] Vector store now contains: {points} points")
    except:
        pass

    print("\n[SUCCESS] Example files loaded successfully!")
    print(f"\nYou can now test queries:")
    print(f'  curl -X POST "{api_url}/api/v1/query" \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"query": "What are the sales figures?", "top_k": 5}}\'')

    return success_count == len(example_files)


if __name__ == "__main__":
    # Get API URL from command line or use default
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    print("=" * 80)
    print("RAG System - Example Data Loader")
    print("=" * 80)

    success = load_example_files(api_url)
    sys.exit(0 if success else 1)
