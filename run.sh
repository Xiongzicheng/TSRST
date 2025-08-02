# 1.rescale
python rescale.py "data/MB/" --image

# 2.data preprocess
python preprocess.py "data/MB/" --image

# 3.extract image features
python extract_features.py "data/MB/" --device="cuda"

# 4.impute super resolution gene expression
python impute.py "data/MB/" --epochs=400 --device="cuda" --n-states=1

# 5.plot imputed gene expression
python plot_imputed.py "data/MB/"

# 6.plot spots gene expression
python plot_spots.py "data/MB/"

# 7.annotation
python cluster.py --filter-size=8 --min-cluster-size=20 --n-clusters=10 "data/MB/embeddings-gene.pickle" "data/MB/clusters-gene/"