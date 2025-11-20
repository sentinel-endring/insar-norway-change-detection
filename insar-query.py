#!/usr/bin/env python3
"""
Author: Torgeir Ferdinand Klingenberg, Torgeir.Ferdinand.Klingenberg@kartverket.no
Script is modified version of IDBRunQuery.py by Daniel Stodle to handle multiple dataset queries using the InSAR Norway API.
Modified to support flexible dataset time periods.
"""
import os
import sys
import traceback
import requests
import json
import argparse
import time
import pathlib
from typing import List, Dict

class InsarQuery:
    def __init__(self, host: str = "https://insar.ngu.no", cert: str = ""):
        self.session = requests.Session()
        if cert:
            self.session.verify = cert
        self.host = host if not host.endswith("/") else host[:-1]
        if not self.host.startswith("http"):
            self.host = "https://" + self.host

    def get_datasets(self) -> Dict:
        """Retrieve and organize available datasets."""
        with self.session.get(f"{self.host}/insar-api/list-datasets") as r:
            if r.status_code != 200:
                raise Exception("Failed to retrieve datasets")
            datasets = json.loads(r.content)
            groups = {}
            for ds in datasets:
                if ds["group"] in groups:
                    groups[ds["group"]].append(ds)
                else:
                    groups[ds["group"]] = [ds]
            return groups

    def get_sentinel_datasets(self, period: str = "2019-2023") -> List[str]:
        """
        Get list of Sentinel-1 datasets for a specific period.
        
        Args:
            period: Time period to target. Options include:
                   - "2019-2023" (default)
                   - "2020-2024" 
                   - "2018-2022" (latest release)
                   - "latest" (uses 2018-2022-release datasets)
                   - "all" (gets all available Sentinel-1 datasets)
        """
        groups = self.get_datasets()
        sentinel_datasets = []
        
        if period == "latest":
            # Use the "latest" release datasets (2018-2022)
            target_groups = [
                "Sentinel1 deformation (latest).Descending 1",
                "Sentinel1 deformation (latest).Descending 2", 
                "Sentinel1 deformation (latest).Ascending 1",
                "Sentinel1 deformation (latest).Ascending 2"
            ]
        elif period == "all":
            # Get all Sentinel-1 groups
            target_groups = [group for group in groups.keys() 
                           if "Sentinel" in group and "Radarsat" not in group]
        else:
            # Use specific period (2019-2023, 2020-2024, etc.)
            target_groups = [
                f"Sentinel-1 ({period}).Descending 1",
                f"Sentinel-1 ({period}).Descending 2",
                f"Sentinel-1 ({period}).Ascending 1", 
                f"Sentinel-1 ({period}).Ascending 2"
            ]
        
        print(f"Looking for dataset groups matching period: {period}")
        print(f"Target groups: {target_groups}")
        
        found_groups = []
        for group_name in target_groups:
            if group_name in groups:
                found_groups.append(group_name)
                for ds in groups[group_name]:
                    sentinel_datasets.append(ds["name"])
            else:
                print(f"Warning: Group '{group_name}' not found in available datasets")
        
        print(f"Found {len(found_groups)} matching groups with {len(sentinel_datasets)} datasets")
        
        if not sentinel_datasets:
            print("No datasets found for the specified period. Available groups:")
            for group_name in sorted(groups.keys()):
                if "Sentinel" in group_name:
                    print(f"  - {group_name}")
        
        return sentinel_datasets
    
    def get_radarsat_datasets(self) -> List[str]:
        """Get list of Radarsat-2 datasets."""
        groups = self.get_datasets()
        radarsat_datasets = []
        
        # Look for Radarsat datasets
        for group_name, datasets in groups.items():
            if "Radarsat" in group_name:
                for ds in datasets:
                    radarsat_datasets.append(ds["name"])
        
        return radarsat_datasets

    def start_query(self, dataset: str, bbox: str) -> str:
        """Start a query for a specific dataset and bbox."""
        url = f"{self.host}/insar-api/{dataset}/query?bbox={bbox}"
        with self.session.post(url) as r:
            if r.status_code != 200:
                print(f"Failed to start query for dataset {dataset}")
                return None
            result = json.loads(r.content)
            return result["id"]

    def poll_query(self, qid: str, output_path: str) -> dict:
        """Poll query status until completion."""
        state = None
        print(f"Query with id {qid} is executing...")
        url = f"{self.host}/insar-api/query-state?id={qid}"
        errors = 0
        time.sleep(1)
        
        while errors < 10:
            try:
                with self.session.get(url) as r:
                    if r.status_code != 200:
                        errors += 1
                        print(f"HTTP error occurred: {r.status_code}. Retry {errors}/10")
                    else:
                        self._write_state(output_path, r.content)
                        state = json.loads(r.content)
            except Exception as e:
                errors += 1
                print(f"Exception occurred: {e}. Retry {errors}/10")
                traceback.print_exc()
                
            if state:
                print(f"State: {state['state']} {state['progress']*100:.1f}% complete [{state['messages'][-1]}] "
                      f"Time: {state['duration']:.1f} seconds")
                if state["state"] == "error":
                    return None
                if state["complete"]:
                    break
            time.sleep(2)
            
        if errors >= 10:
            print("Too many errors, giving up on this query")
            return None
        return state

    def download_results(self, state: dict, qid: str, output_path: str) -> bool:
        """Download query results."""
        success = True
        for csv in state["csv"]:
            url = f"{self.host}/insar-api/query-download?id={qid}&csv={csv}"
            try:
                print(f"Fetching {csv}")
                filename = csv[:-3] if csv.endswith(".gz") else csv
                filepath = os.path.join(output_path, filename)
                
                with open(filepath, "wb") as fd, self.session.get(url, stream=True) as r:
                    for data in r.iter_content(chunk_size=1024*1024):
                        fd.write(data)
                print(f"Successfully downloaded {csv}")
            except Exception as e:
                print(f"Failed to download {csv}: {e}")
                print(f"Manual download URL: {url}")
                success = False
        return success

    def _write_state(self, path: str, state: bytes):
        """Write query state to file."""
        with open(os.path.join(path, "state.json"), "wb") as fd:
            fd.write(state)

def main():
    parser = argparse.ArgumentParser(description="InSAR Norway Multi-Dataset Query Tool")
    parser.add_argument("--path", required=True, help="Directory to store query results")
    parser.add_argument("--bbox", help="Bounding box as lon,lat,lon,lat")
    parser.add_argument("--host", default="https://insar.ngu.no", help="Host to query")
    parser.add_argument("--cert", default="", help="Root certificate for verification")
    parser.add_argument("--list-datasets", action="store_true", help="List available datasets")
    parser.add_argument("--period", default="2019-2023", 
                       help="Time period for Sentinel-1 datasets (2019-2023, 2020-2024, 2018-2022, latest, all)")
    parser.add_argument("--include-radarsat", action="store_true", 
                       help="Also include Radarsat-2 datasets")
    parser.add_argument("--datasets-only", action="store_true",
                       help="Only query Sentinel datasets, skip other dataset types")
    
    options = parser.parse_args()

    # Create output directory
    pathlib.Path(options.path).mkdir(parents=True, exist_ok=True)
    
    # Initialize query handler
    query = InsarQuery(options.host, options.cert)
    
    if options.list_datasets:
        groups = query.get_datasets()
        for group_name, datasets in groups.items():
            print(f"{group_name}:")
            for ds in datasets:
                print(f"  * {ds['displayName']}: {ds['name']}")
        return

    if not options.bbox:
        print("Error: --bbox argument is required")
        return

    # Collect datasets to process
    datasets = []
    
    # Get Sentinel datasets for specified period
    sentinel_datasets = query.get_sentinel_datasets(options.period)
    if sentinel_datasets:
        datasets.extend(sentinel_datasets)
        print(f"Found {len(sentinel_datasets)} Sentinel-1 datasets for period {options.period}")
    else:
        print(f"No Sentinel-1 datasets found for period {options.period}")
        return
    
    # Optionally include Radarsat datasets
    if options.include_radarsat:
        radarsat_datasets = query.get_radarsat_datasets()
        if radarsat_datasets:
            datasets.extend(radarsat_datasets)
            print(f"Added {len(radarsat_datasets)} Radarsat-2 datasets")
    
    if not datasets:
        print("No datasets to process")
        return
    
    print(f"\nProcessing {len(datasets)} total datasets")
    
    # Process each dataset
    for i, dataset in enumerate(datasets, 1):
        print(f"\n[{i}/{len(datasets)}] Processing dataset: {dataset}")
        dataset_path = os.path.join(options.path, dataset)
        pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)
        
        # Start query
        qid = query.start_query(dataset, options.bbox)
        if not qid:
            print(f"Skipping dataset {dataset} due to query start failure")
            continue
            
        # Poll for completion
        state = query.poll_query(qid, dataset_path)
        if not state:
            print(f"Skipping dataset {dataset} due to polling failure")
            continue
            
        # Download results
        if not query.download_results(state, qid, dataset_path):
            print(f"Some downloads failed for dataset {dataset}")
    
    print(f"\nCompleted processing of {len(datasets)} datasets")

if __name__ == "__main__":
    main()
