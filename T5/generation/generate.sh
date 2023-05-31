#!/bin/bash
python generate.py --method chunking --direction backward --index_rank top3
python generate.py --method chunking --direction forward --index_rank top3
python generate.py --method rotation --direction backward --index_rank top3

# # python generate.py --method chunking --direction backward --index_rank top1
# # python generate.py --method chunking --direction forward --index_rank top1
# python generate.py --method rotation --direction forward --index_rank top1
# python generate.py --method rotation --direction forward --index_rank top3