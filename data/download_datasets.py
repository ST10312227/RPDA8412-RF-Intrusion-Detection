"""Download the public UNSW-NB15 and CICIDS2017 benchmark CSV files into data/.

In accordance with ethical clearance CSREC/ST10312227/104 and GitHub's 100 MB file-size
limit, raw secondary datasets are fetched directly from their public repositories rather
than redistributed inside the git repository.
"""
import os
import shutil
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
CIC_DIR = os.path.join(HERE, "cicids")
os.makedirs(CIC_DIR, exist_ok=True)

def main():
    print("Downloading UNSW-NB15 official train/test partitions...")
    for remote, local in [("train.csv", "UNSW_NB15_training-set.csv"),
                          ("test.csv", "UNSW_NB15_testing-set.csv")]:
        p = hf_hub_download(repo_id="Mireu-Lab/UNSW-NB15", filename=remote, repo_type="dataset")
        shutil.copy2(p, os.path.join(HERE, local))
        print(" -> saved", local)

    cicids_files = [
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    ]
    print("Downloading CICIDS2017 MachineLearningCSV files...")
    for fname in cicids_files:
        p = hf_hub_download(repo_id="Mireu-Lab/CIC-IDS", filename=f"CSV/{fname}", repo_type="dataset")
        shutil.copy2(p, os.path.join(CIC_DIR, fname))
        print(" -> saved cicids/" + fname)

if __name__ == "__main__":
    main()
