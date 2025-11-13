## X-CoT
# MSRVTT
python xcot.py --exp_name=MSRVTT --dataset_name=MSRVTT --topk=20 --topk_retrieval_method=clip
python xcot.py --exp_name=MSRVTT --dataset_name=MSRVTT --topk=20 --topk_retrieval_method=vlm2vec
python xcot.py --exp_name=MSRVTT --dataset_name=MSRVTT --topk=20 --topk_retrieval_method=xpool

# DiDeMo
python xcot.py --exp_name=DiDeMo --dataset_name=DiDeMo --topk=20 --topk_retrieval_method=clip
python xcot.py --exp_name=DiDeMo --dataset_name=DiDeMo --topk=20 --topk_retrieval_method=vlm2vec
python xcot.py --exp_name=DiDeMo --dataset_name=DiDeMo --topk=20 --topk_retrieval_method=xpool

# LSMDC
python xcot.py --exp_name=LSMDC --dataset_name=LSMDC --topk=20 --topk_retrieval_method=clip
python xcot.py --exp_name=LSMDC --dataset_name=LSMDC --topk=20 --topk_retrieval_method=vlm2vec
python xcot.py --exp_name=LSMDC --dataset_name=LSMDC --topk=20 --topk_retrieval_method=xpool

# MSVD
python xcot.py --exp_name=MSVD --dataset_name=MSVD --topk=20 --topk_retrieval_method=clip
python xcot.py --exp_name=MSVD --dataset_name=MSVD --topk=20 --topk_retrieval_method=vlm2vec
python xcot.py --exp_name=MSVD --dataset_name=MSVD --topk=20 --topk_retrieval_method=xpool
