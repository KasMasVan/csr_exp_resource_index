import json
import xml.etree.ElementTree as ET


# write my own data loader, or using HF dataloader?

def preprocess_function(examples, **kwargs):
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

def copa_loader(path, **kwargs):
    
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
        examples_copa += [{
            'label': int(value)-1,
            'premise': ' ' + p[:-1] + bridge,
            'uncond_premise': bridge,
            'hypothesis0': ' ' + a1,
            'hypothesis1': ' ' + a2,
        }]
    return examples_copa

def winogrande_loader():
    print(f"winogrande loader")

def cqa_loader(path, **kwargs):
    examples_cqa = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            label = ['A','B','C','D','E'].index(d['answerKey'])
            premise = ' ' +d['question']['stem']
            premise = premise[:-1] + '?' 
            options = [option['text'].lower() for option in d['question']['choices']]

            # examples += [{'options': [{'premise':premise + '? the answer is:' ,
                        #                 'hypothesis': ' "{}"'.format(c['text'].lower()),
                        #                 'uncond_premise': ' the answer is:',
                        #                 'uncond_hypothesis': ' "{}"'.format(c['text'].lower())} for c in d['question']['choices']], 
                        # 'label':label}]
            examples_cqa += [{
                'label': label,
                'premise': premise + ' the answer is:',
                'uncond_premise': ' the answer is:',
                'hypothesis0': options[0],
                'hypothesis1': options[1],
                'hypothesis2': options[2],
                'hypothesis3': options[3],
                'hypothesis4': options[4],
            }]
    return examples_cqa