### Inital Retrievers
## MSRVTT
# Xpool Topk Ranking
python xpool_ranking.py --test_mode=benchmark --exp_name=MSRVTT --videos_dir=data/MSRVTT/videos/all --batch_size=32 --huggingface --load_epoch=-1 --dataset_name=MSRVTT --retrieve_topk --topk=20 --topk_retrieval_method=xpool
# CLIP 
python xpool_ranking.py --test_mode=benchmark --exp_name=MSRVTT --videos_dir=data/MSRVTT/videos/all --batch_size=32 --huggingface --dataset_name=MSRVTT --retrieve_topk --topk=20 --pooling_type=avg --topk_retrieval_method=clip
# VLM2Vec Topk Ranking
python vlm2vec_ranking.py --test_mode=benchmark --arch=qwen2vl_vlm2vec --exp_name=MSRVTT --videos_dir=data/MSRVTT/videos/all  --batch_size=32  --dataset_name=MSRVTT --gpu=0 --pooling_type=avg --data_no_norm --retrieve_topk --topk=20 --topk_retrieval_method=vlm2vec 

## DiDeMo
# Xpool Topk Ranking
python xpool_ranking.py --test_mode=benchmark --exp_name=DiDeMo  --videos_dir=data/Didemo/test_videos/ --batch_size=32 --huggingface --load_epoch=-1 --dataset_name=DiDeMo --retrieve_topk --topk=20 --topk_retrieval_method=xpool
# CLIP 
python xpool_ranking.py --test_mode=benchmark --exp_name=DiDeMo  --videos_dir=data/Didemo/test_videos/ --batch_size=32 --huggingface --dataset_name=DiDeMo --retrieve_topk --topk=20 --pooling_type=avg --topk_retrieval_method=clip
# VLM2Vec Topk Ranking
python vlm2vec_ranking.py --test_mode=benchmark --arch=qwen2vl_vlm2vec --exp_name=DiDeMo --videos_dir=data/Didemo/test_videos/  --batch_size=32  --dataset_name=DiDeMo --gpu=0 --pooling_type=avg --data_no_norm --retrieve_topk --topk=20 --topk_retrieval_method=vlm2vec 

## LSMDC
# Xpool Topk Ranking
python xpool_ranking.py --test_mode=benchmark --exp_name=LSMDC --videos_dir=data/LSMDC --batch_size=32 --huggingface --load_epoch=-1 --dataset_name=LSMDC --retrieve_topk --topk=20 --topk_retrieval_method=xpool
# CLIP 
python xpool_ranking.py --test_mode=benchmark --exp_name=LSMDC --videos_dir=data/LSMDC --batch_size=32 --huggingface --dataset_name=LSMDC --retrieve_topk --topk=20 --pooling_type=avg --topk_retrieval_method=clip
# VLM2Vec Topk Ranking
python vlm2vec_ranking.py --test_mode=benchmark --arch=qwen2vl_vlm2vec --exp_name=LSMDC --videos_dir=data/LSMDC  --batch_size=32  --dataset_name=LSMDC --gpu=0 --pooling_type=avg --data_no_norm --retrieve_topk --topk=20 --topk_retrieval_method=vlm2vec 

## MSVD
python xpool_ranking.py --test_mode=benchmark --exp_name=MSVD  --videos_dir=data/MSVD/YouTubeClips --batch_size=32 --huggingface --load_epoch=-1 --dataset_name=MSVD --retrieve_topk --topk=20 --topk_retrieval_method=xpool
# CLIP 
python xpool_ranking.py --test_mode=benchmark --exp_name=MSVD  --videos_dir=data/MSVD/YouTubeClips --batch_size=32 --huggingface  --dataset_name=MSVD --retrieve_topk --topk=20 --pooling_type=avg --topk_retrieval_method=clip
# VLM2Vec Topk Ranking
python vlm2vec_ranking.py --test_mode=benchmark --arch=qwen2vl_vlm2vec --exp_name=MSVD --videos_dir=data/MSVD/YouTubeClips  --batch_size=32  --dataset_name=MSVD --gpu=0 --pooling_type=avg --data_no_norm --retrieve_topk --topk=20 --topk_retrieval_method=vlm2vec 
