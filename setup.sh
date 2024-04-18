mkdir data
mkdir data/datasets
mkdir data/datasets/asvspoof


echo "Download dataset index"
gdown "https://drive.google.com/u/0/uc?id=1bFEm0X8BF-X6l96Y__zV1Nxpc5sA2-_y" -O data/datasets/asvspoof/dev_index.json
gdown "https://drive.google.com/u/0/uc?id=14Z0K8BqYXJzUBf2eK99t1apJKfYwp93Z" -O data/datasets/asvspoof/eval_index.json
gdown "https://drive.google.com/u/0/uc?id=1IbHJs5Jj0cIioqGRVXCLwLzNrlvba1W-" -O data/datasets/asvspoof/train_index.json