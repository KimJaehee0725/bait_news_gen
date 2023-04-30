from datasets import load_dataset, Dataset
from functools import partial
from itertools import chain
import torch
from tqdm.auto import tqdm
from accelerate import Accelerator
                
def generation(source_title_list: list,
               source_content_list: list,
               target_title_list: list,
               target_content_list: list,
               prefix: str,
               model,
               tokenizer,
               accelerator,
               sum_model=None,
               sum_tokenizer=None,
               batch_size=8,
               use_metadata='content',
               method='chunking',
               direction='forward',
               max_input_length=512):
    
    prefix_length = len(tokenizer.encode(prefix))-1
    
    # ^ single-gpu
    def _generate_title(batch):
        inputs = tokenizer(batch["inputs"], max_length=max_input_length, padding=True ,truncation=True, return_tensors="pt")
        outputs = model.generate(**inputs.to(accelerator.device), num_beams=8, do_sample=True, min_length=10, max_length=32)
        summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        batch["decoded_output"] = summaries
        return batch
    
    # ^ for ddp
    # def _generate_title(sentences):
    #     inputs = tokenizer(sentences['inputs'], max_length=max_input_length, padding=True ,truncation=True, return_tensors="pt")
    #     input_dataset = generate_dataset(inputs)
    #     input_loader = torch.utils.data.DataLoader(input_dataset, batch_size=batch_size)
    #     input_loader = accelerator.prepare(input_loader)
        
    #     output_list = []
    #     model.eval()
    #     for _, batch in tqdm(enumerate(input_loader)):
    #         with torch.no_grad():
    #             outputs = accelerator.unwrap_model(model).generate(**batch, num_beams=8, do_sample=True, min_length=10, max_length=32)
    #         output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    #         output_list.extend(output)

    #     return output_list

    def _generate_chunking(batch, name):
        inputs = tokenizer(batch['{}_content'.format(name)], max_length=chunking_index, truncation=True)['input_ids']
        chunked_content = tokenizer.batch_decode(inputs, skip_special_tokens=True)
        batch['{}_chunked_content'.format(name)] = chunked_content
        return batch

    def _direction_chunking(batch, direction):
        if direction=='forward':
            batch['inputs'] = list(map(" ".join, zip([prefix]*batch_size, 
                                                    batch['source_chunked_content'], 
                                                    batch['target_chunked_content'])))
        if direction=='backward':
            batch['inputs'] = list(map(" ".join, zip([prefix]*batch_size, 
                                                    batch['target_chunked_content'], 
                                                    batch['source_chunked_content'])))
        return batch
    
    # ^ single-gpu
    def _generate_summarization(batch, name):
        inputs = sum_tokenizer(batch["{}_inputs".format(name)], max_length=512, padding=True ,truncation=True, return_tensors="pt")
        outputs = sum_model.generate(**inputs.to(accelerator.device), num_beams=8, do_sample=True, min_length=10, max_length=max_input_length/2)
        summaries = sum_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        batch["{}_summarization_content".format(name)] = summaries
        return batch
    
    # ^ for ddp
    # def _generate_summarization(sentences, name):
    #     inputs = sum_tokenizer(sentences['{}_inputs'.format(name)], max_length=max_input_length, padding=True ,truncation=True, return_tensors="pt")
    #     input_dataset = generate_dataset(inputs)
    #     input_loader = torch.utils.data.DataLoader(input_dataset, batch_size=batch_size)
    #     input_loader = accelerator.prepare(input_loader)
        
    #     output_list = []
    #     sum_model.eval()
    #     for _, batch in tqdm(enumerate(input_loader)):
    #         with torch.no_grad():
    #             outputs = accelerator.unwrap_model(sum_model).generate(**batch, num_beams=8, do_sample=True, min_length=10, max_length=max_input_length/2)
    #         output = sum_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    #         output_list.extend(output)

    #     return output_list

    def _direction_summarization(batch, direction):
        if direction=='forward':
            batch['inputs'] = list(map(" ".join, zip([prefix]*batch_size, 
                                                    batch['source_summarization_content'], 
                                                    batch['target_summarization_content'])))
        if direction=='backward':
            batch['inputs'] = list(map(" ".join, zip([prefix]*batch_size, 
                                                    batch['target_summarization_content'], 
                                                    batch['source_summarization_content'])))
        return batch
    
    def _generate_rotation(batch, name):
        rotation_content = list(map(lambda x: x.split('\n'), batch['{}_content'.format(name)]))
        batch['{}_rotation_content'.format(name)] = rotation_content
        return batch
    
    def _direction_rotation(batch, direction):
        source_content_split = batch['source_rotation_content']
        target_content_split = batch['target_rotation_content']
        
        content_list = []
        for i in range(len(source_content_split)):
            min_len = min(len(source_content_split[i]), len(target_content_split[i]))
            
            if min_len == len(source_content_split[i]) and min_len == len(target_content_split[i]):
                if direction=='forward':
                    rotation_list = list(chain(*zip(source_content_split[i][:min_len], target_content_split[i][:min_len])))
                if direction=='backward':
                    rotation_list = list(chain(*zip(target_content_split[i][:min_len], source_content_split[i][:min_len])))
                content_list.append(prefix + ' '.join(rotation_list))
                
            if min_len == len(source_content_split[i]) and min_len != len(target_content_split[i]):
                if direction=='forward':
                    rotation_list = list(chain(*zip(source_content_split[i][:min_len], target_content_split[i][:min_len])))
                if direction=='backward':
                    rotation_list = list(chain(*zip(target_content_split[i][:min_len], source_content_split[i][:min_len])))
                rotation_list.extend(target_content_split[i][min_len:])
                content_list.append(prefix + ' '.join(rotation_list))
                
            if min_len == len(target_content_split[i]) and min_len != len(source_content_split[i]):
                if direction=='forward':
                    rotation_list = list(chain(*zip(source_content_split[i][:min_len], target_content_split[i][:min_len])))
                if direction=='backward':
                    rotation_list = list(chain(*zip(target_content_split[i][:min_len], source_content_split[i][:min_len])))
                rotation_list.extend(source_content_split[i][min_len:])
                content_list.append(prefix + ' '.join(rotation_list))
            assert len(content_list) == i+1  

        batch['inputs'] = content_list
        return batch
        
    if use_metadata=='content':
        if method=="chunking":
            chunking_index = int((max_input_length-prefix_length)/2)
            
            sentences = Dataset.from_dict({'source_content': source_content_list, 
                                           'target_content': target_content_list})
            
            # 전처리
            sentences = sentences.map(partial(_generate_chunking, name='source'), 
                                      batched=True, batch_size=batch_size)
            sentences = sentences.map(partial(_generate_chunking, name='target'), 
                                      batched=True, batch_size=batch_size)
            sentences = sentences.map(partial(_direction_chunking, direction=direction),
                                      batched=True, batch_size=batch_size)
            # # 생성
            # ^ single gpu
            decoded_output = sentences.map(_generate_title, 
                                           batched=True, batch_size=batch_size)

            # 생성
            # ^ for ddp
            # decoded_output_list = _generate_title(sentences)
                
        if method=="summarization":
            sentences = Dataset.from_dict({'source_content': source_content_list, 
                                        'target_content': target_content_list})
            sentences = sentences.map(lambda x: {
                'source_inputs': prefix + x['source_content']
            })
            sentences = sentences.map(lambda x: {
                'target_inputs': prefix + x['target_content']
            })
            # ^ single gpu
            sentences = sentences.map(partial(_generate_summarization, name='source'), 
                                        batched=True, batch_size=batch_size)
            sentences = sentences.map(partial(_generate_summarization, name='target'), 
                                        batched=True, batch_size=batch_size)
            # ^ for ddp
            # sentences = sentences.add_column(name="source_summarization_content", 
            #                                 column=_generate_summarization(sentences, 'source'))
            # sentences = sentences.add_column(name="target_summarization_content", 
            #                                 column=_generate_summarization(sentences, 'target'))
            
            
            sentences = sentences.map(partial(_direction_summarization, direction=direction),
                                        batched=True, batch_size=batch_size)            
            # 생성
            # ^ single gpu
            decoded_output = sentences.map(_generate_title, 
                                        batched=True, batch_size=batch_size)
            
            # 생성
            # ^ for ddp
            # decoded_output_list = _generate_title(sentences)
        
        if method=="rotation":
            sentences = Dataset.from_dict({'source_content': source_content_list, 
                                            'target_content': target_content_list})
            # 전처리
            sentences = sentences.map(partial(_generate_rotation, name='source'),
                                        batched=True, batch_size=batch_size)
            sentences = sentences.map(partial(_generate_rotation, name='target'),
                                        batched=True, batch_size=batch_size)

            sentences = sentences.map(partial(_direction_rotation, direction=direction),
                                        batched=True, batch_size=batch_size)
            # # 생성
            # ^ single gpu
            decoded_output = sentences.map(_generate_title, 
                                            batched=True, batch_size=batch_size)
            
            # 생성
            # ^ for ddp
            # decoded_output_list = _generate_title(sentences)


    if use_metadata=='full':
        source_title_content = list(chain(*zip(source_title_list, source_content_list)))
        source_content_list = []
        for i in range(0, len(source_title_content), 2):
            source_content_list.append(' '.join(source_title_content[i:i+2]))
        
        target_title_content = list(chain(*zip(target_title_list, target_content_list)))
        target_content_list = []
        for i in range(0, len(target_title_content), 2):
            target_content_list.append(' '.join(target_title_content[i:i+2]))
            
        if method=="chunking":
            chunking_index = int((max_input_length-prefix_length)/2)
            
            sentences = Dataset.from_dict({'source_content': source_content_list, 
                                           'target_content': target_content_list})
            
            # 전처리
            sentences = sentences.map(partial(_generate_chunking, name='source'), 
                                      batched=True, batch_size=batch_size)
            sentences = sentences.map(partial(_generate_chunking, name='target'), 
                                      batched=True, batch_size=batch_size)
            sentences = sentences.map(partial(_direction_chunking, direction=direction),
                                      batched=True, batch_size=batch_size)
            # # 생성
            # ^ single gpu
            decoded_output = sentences.map(_generate_title, 
                                           batched=True, batch_size=batch_size)

            # 생성
            # ^ for ddp
            # decoded_output_list = _generate_title(sentences)
                
        if method=="summarization":
            sentences = Dataset.from_dict({'source_content': source_content_list, 
                                        'target_content': target_content_list})
            sentences = sentences.map(lambda x: {
                'source_inputs': prefix + x['source_content']
            })
            sentences = sentences.map(lambda x: {
                'target_inputs': prefix + x['target_content']
            })
            # ^ single gpu
            sentences = sentences.map(partial(_generate_summarization, name='source'), 
                                        batched=True, batch_size=batch_size)
            sentences = sentences.map(partial(_generate_summarization, name='target'), 
                                        batched=True, batch_size=batch_size)
            # ^ for ddp
            # sentences = sentences.add_column(name="source_summarization_content", 
            #                                 column=_generate_summarization(sentences, 'source'))
            # sentences = sentences.add_column(name="target_summarization_content", 
            #                                 column=_generate_summarization(sentences, 'target'))
            
            
            sentences = sentences.map(partial(_direction_summarization, direction=direction),
                                        batched=True, batch_size=batch_size)            
            # 생성
            # ^ single gpu
            decoded_output = sentences.map(_generate_title, 
                                        batched=True, batch_size=batch_size)
            
            # 생성
            # ^ for ddp
            # decoded_output_list = _generate_title(sentences)
        
        if method=="rotation":
            sentences = Dataset.from_dict({'source_content': source_content_list, 
                                            'target_content': target_content_list})
            # 전처리
            sentences = sentences.map(partial(_generate_rotation, name='source'),
                                        batched=True, batch_size=batch_size)
            sentences = sentences.map(partial(_generate_rotation, name='target'),
                                        batched=True, batch_size=batch_size)

            sentences = sentences.map(partial(_direction_rotation, direction=direction),
                                        batched=True, batch_size=batch_size)
            # # 생성
            # ^ single gpu
            decoded_output = sentences.map(_generate_title, 
                                            batched=True, batch_size=batch_size)
            
            # 생성
            # ^ for ddp
            # decoded_output_list = _generate_title(sentences)
        

        
    
    # ^ single gpu
    return decoded_output['decoded_output']
    # ^ for ddp
    # return decoded_output_list
            
            
class generate_dataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.inputs = inputs
        
    def __len__(self):
        return self.inputs['input_ids'].shape[0]
    
    def __getitem__(self, idx):
        batch_input = {}
        batch_input['input_ids'] = self.inputs['input_ids'][idx]
        batch_input['attention_mask'] = self.inputs['attention_mask'][idx]
        return batch_input
