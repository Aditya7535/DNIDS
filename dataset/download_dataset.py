"""
=============================================================
 KDD Cup 99 Dataset Downloader
=============================================================
 This script downloads the KDD Cup 1999 dataset (10% version)
 which is used for network intrusion detection research.

 Dataset Info:
   - Source: UCI Machine Learning Repository
   - Size:  ~2MB compressed, ~75MB uncompressed
   - Records: ~494,021 network connections
   - Features: 41 network traffic features + 1 label
   - Labels: 'normal.' and 22 attack types

 Attack Categories:
   - DOS   (Denial of Service): smurf, neptune, back, teardrop, pod, land
   - R2L   (Remote to Local): warezclient, guess_passwd, warezmaster, imap, ftp_write, multihop, phf, spy
   - U2R   (User to Root): buffer_overflow, rootkit, loadmodule, perl
   - PROBE (Surveillance): satan, ipsweep, portsweep, nmap
=============================================================
"""

import os
import sys
import gzip
import shutil
import urllib.request

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

# URL for the 10% KDD Cup 99 dataset
DATASET_URL = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"

# Alternative URL (in case primary is down)
DATASET_URL_ALT = "https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/kddcup.data_10_percent.gz"

# Output paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GZ_FILE = os.path.join(SCRIPT_DIR, "kddcup.data_10_percent.gz")
CSV_FILE = os.path.join(SCRIPT_DIR, "kddcup.data_10_percent.csv")

# The 41 feature names for the KDD Cup 99 dataset
# (the dataset CSV does NOT include headers, so we define them here)
FEATURE_NAMES = [
    'duration',             # 1.  Length of connection (seconds)
    'protocol_type',        # 2.  Protocol: tcp, udp, icmp
    'service',              # 3.  Network service: http, ftp, telnet, etc.
    'flag',                 # 4.  Connection status: SF, S0, REJ, etc.
    'src_bytes',            # 5.  Bytes from source to destination
    'dst_bytes',            # 6.  Bytes from destination to source
    'land',                 # 7.  1 if same host/port; 0 otherwise
    'wrong_fragment',       # 8.  Number of wrong fragments
    'urgent',               # 9.  Number of urgent packets
    'hot',                  # 10. Number of "hot" indicators
    'num_failed_logins',    # 11. Number of failed login attempts
    'logged_in',            # 12. 1 if successfully logged in; 0 otherwise
    'num_compromised',      # 13. Number of compromised conditions
    'root_shell',           # 14. 1 if root shell obtained; 0 otherwise
    'su_attempted',         # 15. 1 if "su root" attempted; 0 otherwise
    'num_root',             # 16. Number of root accesses
    'num_file_creations',   # 17. Number of file creation operations
    'num_shells',           # 18. Number of shell prompts
    'num_access_files',     # 19. Number of access control file operations
    'num_outbound_cmds',    # 20. Number of outbound commands (FTP)
    'is_host_login',        # 21. 1 if login is to host; 0 otherwise
    'is_guest_login',       # 22. 1 if login is guest; 0 otherwise
    'count',                # 23. Connections to same host (last 2 sec)
    'srv_count',            # 24. Connections to same service (last 2 sec)
    'serror_rate',          # 25. % connections with SYN errors
    'srv_serror_rate',      # 26. % same-service connections with SYN errors
    'rerror_rate',          # 27. % connections with REJ errors
    'srv_rerror_rate',      # 28. % same-service connections with REJ errors
    'same_srv_rate',        # 29. % connections to same service
    'diff_srv_rate',        # 30. % connections to different services
    'srv_diff_host_rate',   # 31. % different hosts (same service)
    'dst_host_count',       # 32. Connections to same dest host
    'dst_host_srv_count',   # 33. Same dest host & same service
    'dst_host_same_srv_rate',       # 34. % same service (dest host)
    'dst_host_diff_srv_rate',       # 35. % different services (dest host)
    'dst_host_same_src_port_rate',  # 36. % same source port (dest host)
    'dst_host_srv_diff_host_rate',  # 37. % different hosts (dest host + service)
    'dst_host_serror_rate',         # 38. % SYN errors (dest host)
    'dst_host_srv_serror_rate',     # 39. % SYN errors (dest host + service)
    'dst_host_rerror_rate',         # 40. % REJ errors (dest host)
    'dst_host_srv_rerror_rate',     # 41. % REJ errors (dest host + service)
]

# Label column name
LABEL_NAME = 'label'

# All column names (41 features + 1 label = 42 columns)
ALL_COLUMNS = FEATURE_NAMES + [LABEL_NAME]

# Categorical features that need label encoding
CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']


def download_progress(block_num, block_size, total_size):
    """Display download progress bar."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        bar_length = 40
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        sys.stdout.write(f'\r  [{bar}] {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB)')
        sys.stdout.flush()


def download_dataset():
    """Download and extract the KDD Cup 99 dataset."""
    
    print("=" * 60)
    print("  KDD Cup 99 Dataset Downloader")
    print("=" * 60)
    
    # Check if CSV already exists
    if os.path.exists(CSV_FILE):
        file_size = os.path.getsize(CSV_FILE)
        print(f"\n✅ Dataset already exists: {CSV_FILE}")
        print(f"   File size: {file_size / 1024 / 1024:.1f} MB")
        
        # Count lines
        with open(CSV_FILE, 'r') as f:
            line_count = sum(1 for _ in f)
        print(f"   Records: {line_count:,}")
        return CSV_FILE
    
    # ── Method 1: Use scikit-learn's built-in fetcher (most reliable) ──
    print(f"\n📥 Downloading KDD Cup 99 dataset (10% version)...")
    print(f"   Using scikit-learn's fetch_kddcup99 (most reliable)...")
    
    try:
        from sklearn.datasets import fetch_kddcup99
        import pandas as pd
        
        # fetch_kddcup99 downloads and caches the dataset automatically
        data = fetch_kddcup99(subset=None, percent10=True, download_if_missing=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(data.data, columns=FEATURE_NAMES)
        
        # Decode byte strings to regular strings for categorical columns
        for col in ['protocol_type', 'service', 'flag']:
            df[col] = df[col].apply(
                lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x)
            )
        
        # Add the label column
        labels = [
            l.decode('utf-8') if isinstance(l, bytes) else str(l)
            for l in data.target
        ]
        df['label'] = labels
        
        # Save as CSV (without header to match original KDD format)
        df.to_csv(CSV_FILE, index=False, header=False)
        
        print(f"   ✅ Downloaded and saved via scikit-learn!")
        
    except Exception as e:
        print(f"\n⚠️  scikit-learn fetch failed: {e}")
        print(f"   Trying direct URL download...")
        
        # ── Method 2: Direct URL download ──
        try:
            urllib.request.urlretrieve(DATASET_URL, GZ_FILE, download_progress)
            print()
        except Exception as e1:
            print(f"\n⚠️  Primary URL failed: {e1}")
            print(f"   Trying alternative URL...")
            try:
                urllib.request.urlretrieve(DATASET_URL_ALT, GZ_FILE, download_progress)
                print()
            except Exception as e2:
                print(f"\n❌ All download methods failed: {e2}")
                print("\n💡 Manual download instructions:")
                print(f"   1. Go to: {DATASET_URL}")
                print(f"   2. Download kddcup.data_10_percent.gz")
                print(f"   3. Extract to: {CSV_FILE}")
                sys.exit(1)
        
        # Extract the .gz file
        print(f"📦 Extracting dataset...")
        try:
            with gzip.open(GZ_FILE, 'rb') as f_in:
                with open(CSV_FILE, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"   ✅ Extracted to: {CSV_FILE}")
        except Exception as e:
            print(f"   ❌ Extraction failed: {e}")
            sys.exit(1)
        
        # Clean up .gz file
        if os.path.exists(GZ_FILE):
            os.remove(GZ_FILE)
            print(f"   🗑️  Removed compressed file: {GZ_FILE}")
    
    # Verify the dataset
    file_size = os.path.getsize(CSV_FILE)
    with open(CSV_FILE, 'r') as f:
        line_count = sum(1 for _ in f)
    
    print(f"\n{'─' * 60}")
    print(f"  Dataset Summary:")
    print(f"  ├── File: {CSV_FILE}")
    print(f"  ├── Size: {file_size / 1024 / 1024:.1f} MB")
    print(f"  ├── Records: {line_count:,}")
    print(f"  ├── Features: {len(FEATURE_NAMES)}")
    print(f"  └── Columns: {len(ALL_COLUMNS)} (41 features + 1 label)")
    print(f"{'─' * 60}")
    
    # Show sample data
    print(f"\n📋 Sample records (first 3 lines):")
    with open(CSV_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            # Truncate long lines for display
            display_line = line.strip()
            if len(display_line) > 100:
                display_line = display_line[:100] + "..."
            print(f"   [{i+1}] {display_line}")
    
    print(f"\n✅ Dataset ready for use!")
    return CSV_FILE


def get_dataset_path():
    """Return the path to the dataset CSV file."""
    return CSV_FILE


def get_feature_names():
    """Return the list of 41 feature names."""
    return FEATURE_NAMES


def get_all_columns():
    """Return all column names (41 features + label)."""
    return ALL_COLUMNS


# ─────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────
if __name__ == "__main__":
    download_dataset()
