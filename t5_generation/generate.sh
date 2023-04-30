#!/bin/bash
# accelerate launch --config_file ./configs/generation_config.yaml generate.py --method summarization --direction forward
python generate.py --method chunking --direction backward --use_metadata full
python generate.py --method rotation --direction backward --use_metadata full