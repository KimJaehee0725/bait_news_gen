from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5ForConditionalGeneration, AutoTokenizer, T5TokenizerFast
import json
from glob import glob
import os
from tqdm import tqdm
from datasets import load_dataset, Dataset, load_metric
import nltk
import numpy as np
import logging
import argparse

logger = logging.getLogger(__name__)


# todo : eval metric 추가 [3.29]
# todo : checkpoint 이어 학습 코드 추가 [3.29]
def main(args):
    model_path = args.model_path
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def preprocess(data, max_source_length = 512, max_target_length = 32, padding = 'max_length'):
    
        inputs, targets = [], []
        for i in range(len(data['content'])):
            if data['content'][i] and data['title'][i]:
                inputs.append(data['content'][i])
                targets.append(data['title'][i])
        
        prefix = 'summarize: '
            
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
        
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)
        
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs
    
    
    full_dataset = load_dataset('json', data_files={
    'train': './hf_dataset/hf_train.json',
    'valid': './hf_dataset/hf_valid.json',
    'test': './hf_dataset/hf_test.json'})
    
    # train_dataset = full_dataset['train'].select(list(range(3000)))
    train_dataset = full_dataset['train'].map(
        preprocess,
        batched=True,
        fn_kwargs = {'max_source_length': 512, 
                    'max_target_length': 32, 
                    'padding': 'max_length'},
        # num_proc=data_args.preprocessing_num_workers,
        # remove_columns=column_names,
        # load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on train dataset",
    )
    
    if args.validation:
        valid_dataset = full_dataset['valid'].select(list(range(10000))).map(
            preprocess,
            batched=True,
            fn_kwargs = {'max_source_length': 512, 
                        'max_target_length': 32, 
                        'padding': 'max_length'},
            # num_proc=data_training_args.preprocessing_num_workers,
            # remove_columns=column_names,
            # load_from_cache_file=not data_training_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )
    
    if args.test:
        test_dataset = full_dataset['test'].select(list(range(10000))).map(
            preprocess,
            batched=True,
            fn_kwargs = {'max_source_length': 512, 
                        'max_target_length': 32, 
                        'padding': 'max_length'},
            # num_proc=data_training_training_args.preprocessing_num_workers,
            # remove_columns=column_names,
            # load_from_cache_file=not data_training_training_args.overwrite_cache,
            desc="Running tokenizer on test dataset",
        )    
    
    
    # def postprocess_text(preds, labels):
    #     preds = [pred.strip() for pred in preds]
    #     labels = [label.strip() for label in labels]

    #     # rougeLSum expects newline after each sentence
    #     preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    #     labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    #     return preds, labels
    
    metric = load_metric("rouge")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                        for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) 
                        for label in decoded_labels]
        
        # Compute ROUGE scores
        result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                                use_stemmer=True)

        # Extract ROUGE f1 scores
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
        # Add mean generated length to metrics
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                        for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}
    
    
    batch_size = args.batch_size
    model_name = model_path.split('/')[-1] + '_{}'.format(batch_size)
    model_dir = f"Models/{model_name}_{args.exp_name}"

    # ! lr 바꾸기 [3.28]
    training_args = Seq2SeqTrainingArguments(
        model_dir,
        evaluation_strategy="steps",
        eval_steps=15000, # 3000
        logging_strategy="steps",
        logging_steps=15000, # 3000
        save_strategy="steps",
        save_steps=15000,
        learning_rate=4e-5, # 4e-5
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=False, # True
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        report_to="wandb",
        run_name=model_dir,
        resume_from_checkpoint = args.model_path if args.resume else None
    )
        
    # label_pad_token_id = -100 if True else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # train
    logger.info("*** Train ***")
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    trainer.save_model(model_dir)
    
    # Evaluation
    if args.validation:
        max_length = 32
        num_beams = 4
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    # Prediction
    if args.test:
        logger.info("*** Predict ***")
        predict_results = trainer.predict(
            test_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            predictions = tokenizer.batch_decode(
                predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            predictions = [pred.strip() for pred in predictions]
            output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
            with open(output_prediction_file, "w") as writer:
                writer.write("\n".join(predictions))
    
    
if __name__== "__main__":    
    parser = argparse.ArgumentParser()
    
    # # data path
    # parser.add_argument("--data_dir", type=str, default='../data/Part1') 
    # parser.add_argument("--save_dir", type=str, default='../data-direct')
    # parser.add_argument("--output_dir", type=str, default='./models')
    
    # parser.add_argument("--category", type=str, default='EC')
    
    # # hyperparam
    # parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    # parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    # parser.add_argument("--weight_decay", type=float, default=0.01)
    # parser.add_argument("--learning_rate", type=float, default=4e-6)
    
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    # parser.add_argument("--num_train_epochs", type=int, default=2)
    # parser.add_argument("--warmup_steps", type=int, default=0)
    
    # parser.add_argument("--with_tracking", type=bool, default=True)
    parser.add_argument("--model_path", type=str, default="./Models/checkpoint-100000_4_epoch3_1")
    parser.add_argument("--exp_name", type=str, default="3")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--validation", type=bool, default=True)
    parser.add_argument("--test", type=bool, default=True)
    parser.add_argument("--resume", type=bool, default=True)
    
    args = parser.parse_args()
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main(args)