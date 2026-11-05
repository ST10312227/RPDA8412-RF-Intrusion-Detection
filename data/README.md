# Benchmark Datasets (`UNSW-NB15` & `CICIDS2017`)

In accordance with institutional ethical clearance (`CSREC/ST10312227/104`) and GitHub's 100 MB per-file limit, the raw benchmark CSV files are not redistributed inside this repository. Instead, they can be obtained directly from their originating publishers or fetched in one command:

```bash
pip install -r ../requirements.txt
python3 download_datasets.py
```

## Originating Sources
1. **UNSW-NB15** (175,341 training + 82,332 testing flows): Moustafa, N. and Slay, J. (2015), UNSW Canberra Cyber Range Lab — <https://research.unsw.edu.au/projects/unsw-nb15-dataset>
2. **CICIDS2017** (`MachineLearningCSV` flow files): Sharafaldin, I., Lashkari, A. H. and Ghorbani, A. A. (2018), Canadian Institute for Cybersecurity — <https://www.unb.ca/cic/datasets/ids-2017.html>
