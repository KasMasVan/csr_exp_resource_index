import json
import os
import xml.etree.ElementTree as ET

import torch
from huggingface_hub import upload_folder

# write my own data loader, or using HF dataloader?
# steps for data loader: label, premise, options, hypothesis.
# uncond_premise = " the answer is:"

def upload_to_huggingface_hub(dataset, args):
    suffix = f"{args.dataset}_{args.seed}_{args.sample}_{args.checkpoint.split('/')[-1]}_{args.batch_size}"
    temp_data_path = os.path.join(f"../temp_data/{args.method}", suffix)
    dataset.save_to_disk(temp_data_path)
    _ = upload_folder(
        folder_path=temp_data_path, 
        path_in_repo=f"temp_data/{args.method}/{suffix}",
        repo_id="Vanmas/PoE_data",
        repo_type="dataset",)
    # remove the temp data folder
    os.system(f"rm -rf {temp_data_path}")

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

def preprocess_function_seq2seq_channel(examples, **kwargs):
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

    # swap first_sentences and second_sentences
    first_sentences, second_sentences = second_sentences, first_sentences

    # tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    tokenized_headers = tokenizer(first_sentences, padding=True, truncation=True)
    tokenized_endings = tokenizer(second_sentences, padding=True, truncation=True)
    header_dict = {f"header_{k}": [v[i : i + num_choice] for i in range(0, len(v), num_choice)] for k, v in tokenized_headers.items()}
    ending_dict = {f"ending_{k}": [v[i : i + num_choice] for i in range(0, len(v), num_choice)] for k, v in tokenized_endings.items()}
    return {**header_dict, **ending_dict}

def preprocess_function_causal_channel(examples, **kwargs):
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

    # swap first_sentences and second_sentences
    first_sentences, second_sentences = second_sentences, first_sentences

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

def cqa_loader(path, args):
    examples_cqa = []
    if args.calibration_prompt is not None:
        uncond_premise = args.calibration_prompt
    else:
        uncond_premise = " the answer is:"
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
                premise = premise + uncond_premise
            examples_cqa += [{
                'label': label,
                'premise': premise,
                'uncond_premise': uncond_premise,
                'hypothesis0': hypotheses[0],
                'hypothesis1': hypotheses[1],
                'hypothesis2': hypotheses[2],
                'hypothesis3': hypotheses[3],
                'hypothesis4': hypotheses[4],
            }]
    return examples_cqa

def obqa_loader(path, args):
    if args.calibration_prompt is not None:
        uncond_premise = args.calibration_prompt
    else:
        uncond_premise = " the answer is:"
    with open(path) as lines:
        abc2idx = { 'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3 }

        examples_obqa = []
        for line in lines:
            j = json.loads(line)

            label = abc2idx[j['answerKey']]
            premise = j['question']['stem']
            options_text = [f" {option['text']}" for option in j['question']['choices']]
            options_sym = [option['label'] for option in j['question']['choices']]
            if getattr(args, 'multiple_choice_prompt', None) is not None:
                # Question: Greenhouses are great for plants like
                # A. Pizza
                # B. Lollipops
                # C. Candles
                # D. French beans
                # Answer:
                hypotheses = options_sym
                premise = f"{args.multiple_choice_prompt} {premise}\nA. {options_text[0]}\nB. {options_text[1]}\nC. {options_text[2]}\nD. {options_text[3]}\nAnswer:"
            else:
                hypotheses = options_text
                # premise = premise + uncond_premise
            
            examples_obqa += [{
                'label': label,
                'premise': premise,
                'uncond_premise': uncond_premise,
                'hypothesis0': hypotheses[0],
                'hypothesis1': hypotheses[1],
                'hypothesis2': hypotheses[2],
                'hypothesis3': hypotheses[3],
            }]
    return examples_obqa

def piqa_loader(path, args):
    if args.calibration_prompt is not None:
        uncond_premise = args.calibration_prompt
    else:
        uncond_premise = " the answer is:"
    examples_piqa = []
    qa_path, label_path = path[0], path[1]

    with open(qa_path) as lines, open(label_path) as labels:
        for line, label_sym in zip(lines, labels):
            label = int(label_sym[0])
            line = json.loads(line)
            premise = line['goal']
            options_text = [line['sol1'], line['sol2']]
            options_sym = ['A', 'B']

            if getattr(args, 'multiple_choice_prompt', None) is not None:
                # Question: To clear snot out of your nose,
                # A. place a tissue over your nose and blow the snot out.
                # B. place a tissue over your nose and suck the snot in.
                # Answer:
                hypotheses = options_sym
                premise = f"{args.multiple_choice_prompt} {premise}\nA. {options_text[0]}\nB. {options_text[1]}\nAnswer:"
            else:
                hypotheses = options_text
                premise = premise + uncond_premise
            
            examples_piqa += [{
                'label': label,
                'premise': premise,
                'uncond_premise': uncond_premise,
                'hypothesis0': hypotheses[0],
                'hypothesis1': hypotheses[1],
            }]
    return examples_piqa

def qasc_loader(path, args):
    if args.calibration_prompt is not None:
        uncond_premise = args.calibration_prompt
    else:
        uncond_premise = " the answer is:"
    examples_qasc = []

    with open(path) as lines:
        for line in lines:
            line = json.loads(line)
            label = ['A','B','C','D','E','F','G','H'].index(line['answerKey'])
            premise = f"{line['question']['stem']}"

            options_text = [option['text'] for option in line['question']['choices']]
            options_sym = [option['label'] for option in line['question']['choices']]

            if getattr(args, 'multiple_choice_prompt', None) is not None:
                # Question: Cameron returned home with a bag of candy to eat all night
                # long. What will Others want to do next?
                # A. great
                # B. buy the candy to eat
                # C. bored
                # E. ...
                # Answer:
                hypotheses = options_sym
                premise = f"{args.multiple_choice_prompt} {premise}\nA. {options_text[0]}\nB. {options_text[1]}\nC. {options_text[2]}\nD. {options_text[3]}\nE. {options_text[4]}\nF. {options_text[5]}\nG. {options_text[6]}\nH. {options_text[7]}\nAnswer:"
            else:
                hypotheses = options_text
                premise = premise + uncond_premise
            
            examples_qasc += [{
                'label': label,
                'premise': premise,
                'uncond_premise': uncond_premise,
                'hypothesis0': hypotheses[0],
                'hypothesis1': hypotheses[1],
                'hypothesis2': hypotheses[2],
                'hypothesis3': hypotheses[3],
                'hypothesis4': hypotheses[4],
                'hypothesis5': hypotheses[5],
                'hypothesis6': hypotheses[6],
                'hypothesis7': hypotheses[7],
            }]
    return examples_qasc

def siqa_loader(path, args):
    if args.calibration_prompt is not None:
        uncond_premise = args.calibration_prompt
    else:
        uncond_premise = " the answer is:"
    examples_siqa = []
    qa_path, label_path = path[0], path[1]

    with open(qa_path) as lines, open(label_path) as labels:
        for line, label_sym in zip(lines, labels):
            label = int(label_sym[0]) - 1
            line = json.loads(line)
            premise = f"{line['context']} {line['question']}"
            
            options_text = [line['answerA'], line['answerB'], line['answerC']]
            options_sym = ['A', 'B', 'C']

            if getattr(args, 'multiple_choice_prompt', None) is not None:
                # Question: Cameron returned home with a bag of candy to eat all night
                # long. What will Others want to do next?
                # A. great
                # B. buy the candy to eat
                # C. bored
                # Answer:
                hypotheses = options_sym
                premise = f"{args.multiple_choice_prompt} {premise}\nA. {options_text[0]}\nB. {options_text[1]}\nC. {options_text[2]}\nAnswer:"
            else:
                hypotheses = options_text
                premise = premise + uncond_premise
            
            examples_siqa += [{
                'label': label,
                'premise': premise,
                'uncond_premise': uncond_premise,
                'hypothesis0': hypotheses[0],
                'hypothesis1': hypotheses[1],
                'hypothesis2': hypotheses[2],
            }]
    return examples_siqa

def winogrande_loader(path, args):
    if args.calibration_prompt is not None:
        uncond_premise = args.calibration_prompt
    else:
        uncond_premise = " the answer is:"
    examples_winogrande = []
    qa_path, label_path = path[0], path[1]

    with open(qa_path) as lines, open(label_path) as labels:
        for line, label_sym in zip(lines, labels):
            label = int(label_sym[0]) - 1
            line = json.loads(line)
            premise = line['sentence']
            
            options_text = [line['option1'], line['option2']]
            options_sym = ['A', 'B']

            if getattr(args, 'multiple_choice_prompt', None) is not None:
                # Question: So _ plays video games because Leslie has a lot of free time
                # while Nelson has to work all the time.
                # A. Leslie
                # B. Nelson
                # Answer:
                hypotheses = options_sym
                premise = f"{args.multiple_choice_prompt} {premise}\nA. {options_text[0]}\nB. {options_text[1]}\nAnswer:"
            else:
                hypotheses = options_text
                premise = premise + uncond_premise
            
            examples_winogrande += [{
                'label': label,
                'premise': premise,
                'uncond_premise': uncond_premise,
                'hypothesis0': hypotheses[0],
                'hypothesis1': hypotheses[1],
            }]
    return examples_winogrande

def emoji_movie_loader(path, args):
    if args.calibration_prompt is not None:
        uncond_premise = args.calibration_prompt
    else:
        uncond_premise = " the answer is:"
    examples = []

    with open(path) as json_file:
        data = json.load(json_file)
        for instance in data['examples']:
            options_text = list(instance['target_scores'].keys())
            options_sym = [chr(ord('A') + i) for i in range(len(options_text))]
            label = options_text.index(instance['target'])
            premise = instance['input']

            if getattr(args, 'multiple_choice_prompt', None) is not None:
                # Question: What movie does this emoji describe? 👧🐟🐠🐡
                # A. finding nemo
                # B. the wolf of wall street
                # C. ...
                # D. ...
                # E. ...
                # Answer:
                hypotheses = options_sym
                premise = f"{args.multiple_choice_prompt} {premise}\nA. {options_text[0]}\nB. {options_text[1]}\nC. {options_text[2]}\nD. {options_text[3]}\nE. {options_text[4]}\nAnswer:"
            else:
                hypotheses = options_text
                premise = premise + uncond_premise

            examples += [{
                'label': label,
                'premise': premise,
                'uncond_premise': uncond_premise,
                'hypothesis0': hypotheses[0],
                'hypothesis1': hypotheses[1],
                'hypothesis2': hypotheses[2],
                'hypothesis3': hypotheses[3],
                'hypothesis4': hypotheses[4],
            }]
    return examples

def ruin_names_loader(path, args):
    if args.calibration_prompt is not None:
        uncond_premise = args.calibration_prompt
    else:
        uncond_premise = " the answer is:"
    examples = []

    with open(path) as json_file:
        data = json.load(json_file)
        for instance in data['examples']:
            options_text = list(instance['target_scores'].keys())
            num_options = len(options_text)
            options_sym = [chr(ord('A') + i) for i in range(num_options)]
            for target, score in instance['target_scores'].items():
                if score == 1:
                    raw_label = target # e.g., stare wars
            label = options_text.index(raw_label)             
            premise = instance['input']

            if getattr(args, 'multiple_choice_prompt', None) is not None:
                # Question: "Which of the following is a humorous edit of this artist or movie name: 'star wars'?"
                # A. stare wars
                # B. stariwars
                # C. ...
                # D. ...
                # Answer:
                hypotheses = options_sym
                # premise = f"{args.multiple_choice_prompt} {premise}\nA. {options_text[0]}\nB. {options_text[1]}\nC. {options_text[2]}\nD. {options_text[3]}\nE. {options_text[4]}\nAnswer:"
                premise = f"{args.multiple_choice_prompt} {premise}\n"
                for idx in range(num_options):
                    premise += f"{options_sym[idx]}. {options_text[idx]}\n" 
                premise += "Answer:"                
            else:
                hypotheses = options_text
                premise = premise + uncond_premise
            example = [{
                'label': label,
                'premise': premise,
                'uncond_premise': uncond_premise,
            }]
            for idx in range(num_options):
                example[0][f'hypothesis{idx}'] = hypotheses[idx]

            examples += example
    return examples

def conceptual_combinations_loader(path, args):
    if args.calibration_prompt is not None:
        uncond_premise = args.calibration_prompt
    else:
        uncond_premise = " the answer is:"
    examples = []

    for one_path in path:
        with open(one_path) as json_file:
            data = json.load(json_file)
            for instance in data['examples']:
                options_text = list(instance['target_scores'].keys())
                num_options = len(options_text)
                options_sym = [chr(ord('A') + i) for i in range(num_options)]
                for target, score in instance['target_scores'].items():
                    if score == 1:
                        raw_label = target # e.g., stare wars
                label = options_text.index(raw_label)             
                premise = instance['input']

                if getattr(args, 'multiple_choice_prompt', None) is not None:
                    # Question: "Which of the following is a humorous edit of this artist or movie name: 'star wars'?"
                    # A. stare wars
                    # B. stariwars
                    # C. ...
                    # D. ...
                    # Answer:
                    hypotheses = options_sym
                    # premise = f"{args.multiple_choice_prompt} {premise}\nA. {options_text[0]}\nB. {options_text[1]}\nC. {options_text[2]}\nD. {options_text[3]}\nE. {options_text[4]}\nAnswer:"
                    premise = f"{args.multiple_choice_prompt} {premise}\n"
                    for idx in range(num_options):
                        premise += f"{options_sym[idx]}. {options_text[idx]}\n" 
                    premise += "Answer:"                
                else:
                    hypotheses = options_text
                    premise = premise + uncond_premise
                example = [{
                    'label': label,
                    'premise': premise,
                    'uncond_premise': uncond_premise,
                }]
                for idx in range(num_options):
                    example[0][f'hypothesis{idx}'] = hypotheses[idx]

                examples += example
    return examples

def anli_loader(path, args):
    if args.calibration_prompt is not None:
        uncond_premise = args.calibration_prompt
    else:
        uncond_premise = " the answer is:"
    examples = []
    options_text = ["entailment", "neutral", "contradiction"]
    options_sym = ["A", "B", "C"]
    num_options = 3

    for one_path in path:
        with open(one_path) as lines:
            for line in lines:
                line = json.loads(line)
                label = ['e', 'n', 'c'].index(line['label'])       
                premise = f"{line['context']} {line['hypothesis']}"
                
                if getattr(args, 'multiple_choice_prompt', None) is not None:
                    # Question: "Which of the following is a humorous edit of this artist or movie name: 'star wars'?"
                    # A. entailment
                    # B. neutral
                    # C. contradiction
                    # Answer:
                    hypotheses = options_sym
                    # premise = f"{args.multiple_choice_prompt} {premise}\nA. {options_text[0]}\nB. {options_text[1]}\nC. {options_text[2]}\nD. {options_text[3]}\nE. {options_text[4]}\nAnswer:"
                    premise = f"{args.multiple_choice_prompt} {premise}\n"
                    for idx in range(num_options):
                        premise += f"{options_sym[idx]}. {options_text[idx]}\n" 
                    premise += "Answer:"                
                else:
                    hypotheses = options_text
                    premise = premise + uncond_premise
                example = [{
                    'label': label,
                    'premise': premise,
                    'uncond_premise': uncond_premise,
                }]
                for idx in range(num_options):
                    example[0][f'hypothesis{idx}'] = hypotheses[idx]
                examples += example

    return examples