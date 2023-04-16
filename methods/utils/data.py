import xml.etree.ElementTree as ET

# write my own data loader, or using HF dataloader?

def copa_loader(path, **kwargs):
    # cond_mcp, uncond_mcp, domain_cond = kwargs['cond_mcp'], kwargs['uncond_mcp'], kwargs['domain_cond']
    cond_mcp, uncond_mcp, domain_cond = "test", "test", "test"
    
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
        options = [a1, a2]
        if asks_for =='effect':
            bridge = ' so'
        elif asks_for =='cause':
            bridge = ' because'
        else: 
            assert(False)
            
        examples_copa  += [{'options': [{'premise':' '+ cond_mcp.replace("[answers]", str(options)) + p[:-1] + bridge + '.' + domain_cond,
                                         'hypothesis': ' '+ a1,
                                         'uncond_premise':uncond_mcp.replace("[answers]", str(options)) + bridge + '.' + domain_cond,
                                         'uncond_hypothesis':' '+a1},
                                       {'premise':' '+ cond_mcp.replace("[answers]", str(options)) + p[:-1] + bridge + '.' + domain_cond,
                                         'hypothesis': ' '+a2,
                                         'uncond_premise':uncond_mcp.replace("[answers]", str(options)) + bridge + '.' + domain_cond,
                                         'uncond_hypothesis':' '+a2}], 
                  'label':int(value)-1}]
    return examples_copa

def winogrande_loader():
    print(f"winogrande loader")
