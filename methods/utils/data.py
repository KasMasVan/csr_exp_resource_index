import json
import xml.etree.ElementTree as ET

import torch


# write my own data loader, or using HF dataloader?

def preprocess_function_seq2seq(examples, **kwargs):
    ending_names, header_name, tokenizer = kwargs['ending_names'], kwargs['header_name'], kwargs['tokenizer']
    num_choice = len(ending_names)
    question_headers = examples[header_name]
    # the tokenizer handles multiple spaces.
    first_sentences = [[context] * len(ending_names) for context in examples[header_name]]
    # second_sentences = [
    #     [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_header)
    # ]
    second_sentences = [
        [f"{examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
    ]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    tokenized_headers = tokenizer(first_sentences, padding=True, truncation=True)
    tokenized_endings = tokenizer(second_sentences, padding=True, truncation=True)
    header_dict = {f"header_{k}": [v[i : i + num_choice] for i in range(0, len(v), num_choice)] for k, v in tokenized_headers.items()}
    ending_dict = {f"ending_{k}": [v[i : i + num_choice] for i in range(0, len(v), num_choice)] for k, v in tokenized_endings.items()}
    return {**header_dict, **ending_dict}

def preprocess_function_causal(examples, **kwargs):
    ending_names, header_name, tokenizer = kwargs['ending_names'], kwargs['header_name'], kwargs['tokenizer']
    num_choice = len(ending_names)
    question_headers = examples[header_name]
    # the tokenizer handles multiple spaces.
    first_sentences = [[context] * len(ending_names) for context in examples[header_name]]
    # second_sentences = [
    #     [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_header)
    # ]
    second_sentences = [
        [f"{examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
    ]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    tokenized_headers = tokenizer(first_sentences, truncation=True)
    tokenized_endings = tokenizer(second_sentences, truncation=True)

    # reference: https://github.com/peterwestuw/surface-form-competition/blob/main/utils.py#L177
    max_len = max(len(header + ending) for header, ending in zip(tokenized_headers['input_ids'], tokenized_endings['input_ids']))
    input_ids = torch.full((len(tokenized_headers['input_ids']), max_len), tokenizer.pad_token_id, dtype=torch.long)
    labels = tokenizer.pad_token_id * torch.ones((len(tokenized_headers['input_ids']), max_len), dtype=torch.long)
    ending_attention_mask = torch.zeros((len(tokenized_headers['input_ids']), max_len), dtype=torch.long)
    for i, (header, ending) in enumerate(zip(tokenized_headers['input_ids'], tokenized_endings['input_ids'])):
        input_ids[i, :len(header)] = torch.tensor(header)
        input_ids[i, len(header):len(header)+len(ending)] = torch.tensor(ending)
        ending_attention_mask[i, len(header):len(header)+len(ending)] = torch.tensor(1)
        labels[i, len(header):len(header)+len(ending)] = torch.tensor(ending)

    flatten_dict = {"input_ids": input_ids, "labels": labels, "ending_attention_mask": ending_attention_mask}
    return_dict = {f"{k}": [v[i : i + num_choice] for i in range(0, len(v), num_choice)] for k, v in flatten_dict.items()}
    return return_dict

def copa_loader(path, args):
    
    root = ET.parse(path).getroot()
    examples_copa = []
    for type_tag in root.findall('item'):
        # xml stuff
        value = type_tag.get('most-plausible-alternative')
        asks_for = type_tag.get('asks-for')
        children = list(type_tag)
        # get the texts
        p = children[0].text
        a1 = children[1].text[:1].lower() +  children[1].text[1:]
        a2 = children[2].text[:1].lower() +  children[2].text[1:]
        if asks_for =='effect':
            bridge = ' so'
        elif asks_for =='cause':
            bridge = ' because'
        else: 
            assert(False)
        # examples_copa  += [{'options': [{'premise': ' ' + p[:-1] + bridge,
        #                                  'hypothesis': ' ' + a1,
        #                                  'uncond_premise': bridge,
        #                                  'uncond_hypothesis': ' ' + a1},
        #                                {'premise': ' ' + p[:-1] + bridge,
        #                                  'hypothesis': ' ' + a2,
        #                                  'uncond_premise': bridge,
        #                                  'uncond_hypothesis': ' ' + a2}], 
        #           'label':int(value)-1}]
        premise = ' ' + p[:-1] + bridge
        if getattr(args, 'multiple_choice_prompt', None) is not None:
                # Question: The pond froze over for the winter so
                # A. People skated on the pond.
                # B. People brought boats to the pond.
                # Answer:
                hypotheses = ["A", "B"]
                premise = f"{args.multiple_choice_prompt} {premise}\nA. {a1}\nB. {a2}\nAnswer:"
        else:
            hypotheses = [' ' + a1, ' ' + a2]
        examples_copa += [{
            'label': int(value)-1,
            'premise': premise,
            'uncond_premise': bridge,
            'hypothesis0': hypotheses[0],
            'hypothesis1': hypotheses[1],
        }]
    return examples_copa

def winogrande_loader():
    print(f"winogrande loader")

def cqa_loader(path, args):
    examples_cqa = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            label = ['A','B','C','D','E'].index(d['answerKey'])
            premise = ' ' + d['question']['stem']
            premise = premise[:-1] + '?'  
            
            # to ensure identical perforamnce to the PMI paper.
            # options = [option['text'].lower() for option in d['question']['choices']]
            options = [f" \"{option['text'].lower()}\"" for option in d['question']['choices']]
            
            # examples += [{'options': [{'premise':premise + '? the answer is:' ,
                        #                 'hypothesis': ' "{}"'.format(c['text'].lower()),
                        #                 'uncond_premise': ' the answer is:',
                        #                 'uncond_hypothesis': ' "{}"'.format(c['text'].lower())} for c in d['question']['choices']], 
                        # 'label':label}]
            # if args.multiple_choice_prompt is not None:
            if getattr(args, 'multiple_choice_prompt', None) is not None:
                hypotheses = ["A", "B", "C", "D", "E"]
                # Question: How does a bishop move from one place to another?
                # A. chess game
                # B. church
                # C. in a car
                # D. queen
                # E. cathedral
                # Answer:
                premise = f"{args.multiple_choice_prompt} {premise}\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\nE. {options[4]}\nAnswer:"
            else:
                hypotheses = options
                premise = premise + ' the answer is:'
            examples_cqa += [{
                'label': label,
                'premise': premise,
                'uncond_premise': ' the answer is:',
                'hypothesis0': hypotheses[0],
                'hypothesis1': hypotheses[1],
                'hypothesis2': hypotheses[2],
                'hypothesis3': hypotheses[3],
                'hypothesis4': hypotheses[4],
            }]
    return examples_cqa