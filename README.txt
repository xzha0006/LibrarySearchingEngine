1. Credit level
python3 my_search_engine_Credit.py index collection_dir index_dir stopwords.txt
example: python3 my_search_engine_Credit.py index ../smallFiles/collection/ . stopwords.txt

python3 my_search_engine_Credit.py search index_dir num_docs keyword_list
example: python3 my_search_engine_Credit.py search . 10 dog cat

2. HD level
python3 my_search_engine_HD.py
There is a simple manual inside the program.

Parallel indexing
pip install mpi4py
mpiexec -n 3 python3 my_search_engine.py
