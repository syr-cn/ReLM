import numpy as np
import random
from utils import to_char
from utils import smiles2iupac

system_message_1 = """
Please choose the correct answer in the choices according to the context, the question and the answer candidates.
Each candidate is accociated with a confidence score within a bracket, where a lower score is better.
Your answer should be one uppercase character, representing your choice.
"""

system_message_iupac_smiles = """
Please choose the correct answer in the choices according to the context, the question and the answer candidates.
Each candidate is accociated with a confidence score within a bracket, where a lower score is better.
Your answer should be one uppercase character, representing your choice.
The compounds are given in the form of NAME<SMILES>, where NAME indicates the IUPAC name of the candidate, and SMILES stands for the smiles string of the candidate.
"""

system_message_2 = """
You are an expert at correctly answering molecular reaction multiple-choice questions.
I will write a question, and then a short list of answers you can choose from, and then you must choose the correct answer and reply with that answer and only that answer.
Your answer should be one uppercase character, representing your choice.
I will give you three examples. The compounds are given in the form of IUPAC names.
"""

system_message_3 = """
You are an expert at correctly answering molecular reaction multiple-choice questions.
I will write a question, and then a short list of answers you can choose from, and then you must choose the correct answer and reply with that answer and your confidence score.
The confidence score should be between 1 and 9, the higher it is, the more confident you are in your choice.
You should provide the score according to your confidence and familiarity with the problem 
I will give you three examples with potential confidence score.
The compounds are given in the form of IUPAC names.
"""


system_message_4 = """
Please choose the correct answer in the choices according to the context, the question and the answer candidates.
Each candidate is accociated with a confidence score within a bracket, where a lower score is better.
I will write a question, and then a short list of answers you can choose from, and then you must choose the correct answer and reply with that answer and your confidence score.
The confidence score should be between 1 and 9, the higher it is, the more confident you are in your choice.
You should provide the score according to your confidence and familiarity with the problem 
The compounds are given in the form of IUPAC names.
"""

system_message_5 = """
Please choose the correct answer in the choices according to the context, the question and the answer candidates.
Each candidate is accociated with a confidence score within a bracket, where a lower score is better.
I will write a question, and then a short list of answers you can choose from, and then you must choose the correct answer and reply with that answer and your confidence score.
The confidence score should be between 1 and 9, the higher it is, the more confident you are in your choice.
You should provide the score according to your confidence and familiarity with the problem 
The compounds are given in the form of IUPAC<SMILES>, where IUPAC indicates the IUPAC name of the candidate, and SMILES stands for the smiles string of the candidate.
"""

system_message_6 = """
Please choose the correct answer in the choices according to the context, the question, and the answer candidates.
I will write a question, and then a short list of answers you can choose from, and then you must choose the correct answer and reply with that answer and your confidence score.
The confidence score should be between 1 and 9. The higher it is, the more confident you are in your choice.
You should provide the score according to your confidence and familiarity with the problem.
The compounds are given in the form of IUPAC<SMILES>, where IUPAC indicates the IUPAC name of the candidate, and SMILES stands for the smiles string of the candidate.
"""

system_message_7 = """
You are ReactionGPT, an expert at correctly answering multiple choice questions about chemical reaction.

Please choose the correct answer in the choices according to the context, the question, and the answer candidates.
I will write a question, and then a short list of answers you can choose from, and then you must choose the correct answer and reply with that answer.
The compounds are given in the form of IUPAC<SMILES>, where IUPAC indicates the IUPAC name of the candidate, and SMILES stands for the smiles string of the candidate.
"""

system_message_8 = """
You are ReactionGPT, an expert at correctly answering multiple choice questions about chemical reaction.

I will write a question, and then a short list of answers you can choose from, and then you must choose the correct answer and reply with that answer and your confidence score.
Please choose the correct answer in the choices according to the context, the question, and the answer candidates.
The confidence score should be between 1 and 9. The higher it is, the more confident you are in your choice.
You should provide the score according to your confidence and familiarity with the problem.
The compounds are given in the form of IUPAC<SMILES>, where IUPAC indicates the IUPAC name of the candidate, and SMILES stands for the smiles string of the candidate.
"""

system_message_9 = """
You are ReactionGPT, an expert at correctly answering multiple choice questions about chemical reaction.

I will write a question, and then a short list of answers you can choose from, and then you must choose the correct answer and reply with that answer and your confidence score.
Please choose the correct answer in the choices according to the context, the question, and the answer candidates.
The confidence score should be between 1 and 9. The higher it is, the more confident you are in your choice.
You should provide the score according to your confidence and familiarity with the problem.
The compounds are given in the form of IUPAC<SMILES>, where IUPAC indicates the IUPAC name of the candidate, and SMILES stands for the SMILES string of the candidate.
For those compounds without available IUPAC names, only the smiles string will be provided.

The given reaction is a multicomponent reaction that leads to the synthesis of diverse imidazopyridines.
"""

system_message_10 = """
You are ReactionGPT, an expert at correctly answering multiple choice questions about chemical reaction.

I will write a question, and then a short list of answers you can choose from, and then you must choose the correct answer and reply with that answer and your confidence score.
Please choose the correct answer in the choices according to the context, the question, and the answer candidates.
The confidence score should be between 1 and 9. The higher it is, the more confident you are in your choice.
You should provide the score according to your confidence and familiarity with the problem.
The compounds are given in the form of IUPAC<SMILES>, where IUPAC indicates the IUPAC name of the candidate, and SMILES stands for the SMILES string of the candidate.
For those compounds without available IUPAC names, only the smiles string will be provided.

The given reaction is a Nickel-catalyzed cross-couplings reaction, which forms carbon-carbon (C-C) and carbon-heteroatom (C-X, where X can be O, N, S, etc.) bonds.
"""

system_message_json = '''
You are ReactionGPT, an expert at correctly answering multiple choice questions about chemical reaction.

Please choose the correct answer in the choices according to the context, the question, and the answer candidates.
I will write a question, and then a short list of answers you can choose from, and then you must choose the correct answer, and provide your confidence score about the answer.
The confidence score should be between 1 and 9. The higher it is, the more confident you are in your choice.
You should provide the score according to your confidence and familiarity with the problem.
Besides the answer and the confidence score, you should provide your thoughts about your answer, including your thought, your reasoning, and you self-criticism.
The compounds are given in the form of IUPAC<SMILES>, where IUPAC indicates the IUPAC name of the candidate, and SMILES stands for the smiles string of the candidate.

You should only respond in JSON format as described below
Response Format: 
{
    "thoughts": {
        "text": "thought",
        "reatant": "roles and properties of reactants",
        "reaction type": "a 3 component reaction approach towards diverse imidazopyridines",
        "product: "properties of the possible products"
        "criticism": "constructive self-criticism",
    },
    "answer": {
        "choice": "one capital letter, representing your choice",
        "confidence": "one integer, representing your confidence score"
    }
}
Ensure the response can be parsed by Python json.loads
'''

# use_catalyst=0
SYSTEM_MESSAGE = system_message_8
query_info=0
SYSTEM_MESSAGE_NOCONF = system_message_7


# catalyst_name = None
Imidazo_Catalyst = '2,2,2-trifluoroacetic acid<C(=O)(C(F)(F)F)O>'
# if use_catalyst:
#     catalyst_name = Imidazo_Catalyst

in_context_json = [
    {"role": "user", "content": 
'''
Question: Which of the following is the major product of the organic reaction between 5-methylpyridin-2-amine<CC1=CN=C(C=C1)N>, isocyanocyclohexane<[C-]#[N+]C1CCCCC1> and 2,3-dihydroxybenzaldehyde<C(c1cccc(c1O)O)=O>?
A. tert-butyl 3-(4-chlorophenyl)-1,2-thiazole-4-carboxylate<CC(C)(C)OC(=O)c1csnc1-c1ccc(Cl)cc1>
B. tert-butyl-dimethyl-oct-7-ynoxysilane<C#CCCCCCCO[Si](C)(C)C(C)(C)C>
C. 5,6,7,8-tetrahydrothieno[3,2-b]thiepin-8-ol<OC1CCCSc2ccsc21>
'''
    },
    {"role": "assistant", "content": 
'''
{
"thoughts": {
"text": "The given reaction is a multicomponent reaction that leads to the synthesis of diverse imidazopyridines. The given compounds include a pyridine, an isocyanide, and an aldehyde which suggests that the reaction might involve a multi-component reaction strategy such as the Ugi reaction or the Passerini reaction.",
"reactant": "5-methylpyridin-2-amine, isocyanocyclohexane, and 2,3-dihydroxybenzaldehyde are the reactants. 5-methylpyridin-2-amine is a pyridine derivative, isocyanocyclohexane is an isocyanide, and 2,3-dihydroxybenzaldehyde is an aldehyde. ",
"reaction type": "The given reaction is a 3 component reaction approach towards diverse imidazopyridines.",
"product": "The product should have imidazopyridine as the main functional group.",
"reasoning": "There is no silicon element in the reactants, so the answer can't be B; there's also no sulfur elements in the reactants, so the answer can't be A.",
"criticism": "Since I know the reaction is about imidazopyridine synthesis, thus I'm prety confident about my answer."
},
"answer": {
"choice": "A",
"confidence": 9
}
}
'''
    }
]

def prompt_json(context, func=lambda x:x, useD=False, conf=True):
    messages = [{"role": "system", "content": system_message_json}]
    reactant, options, dist, answer, _ = context[-1]
    messages.extend(in_context_json)
    messages.append({
        "role": "user",
        "content": prompt_single(reactant, options, dist, None, func=func, test=True, useD=useD)
    })
    return messages

def prompt_json_in_context(context, func=lambda x:x, useD=False, conf=True):
    prompt = []
    prompt.append(system_message_json)
    context_prompt = [i['content'] for i in in_context_json]
    context_prompt = '\n'.join(context_prompt)
    prompt.append(context_prompt)

    prompt.append(f'Please answer the following question based on the example above.')
    reactant, options, dist, _, _ = context[-1]
    query = prompt_single(reactant, options, dist, None, func=func, test=True, useD=useD)
    prompt.append(query)
    messages = '\n\n'.join(prompt)
    return messages

def prompt_chat(context, func=lambda x:x, useD=False, conf=True, name=''):
    if '10b9' in name:
        system_message = system_message_9
        catalyst_name = Imidazo_Catalyst
    elif '9b8a' in name:
        system_message = system_message_10
        catalyst_name = None
    else:
        system_message = SYSTEM_MESSAGE
        catalyst_name = None
    
    if conf:
        messages = [{"role": "system", "content": system_message}]
    else:
        messages = [{"role": "system", "content": SYSTEM_MESSAGE_NOCONF}]
    for item in context[:-1]:
        reactant, options, dist, answer, confidence = item
        messages.append({
            "role": "user",
            "content": prompt_single(reactant, options, dist, None, func=func, test=False, useD=useD)
        })
        messages.append({
            "role": "assistant",
            "content": f"{to_char(answer)} {confidence}" if conf else f"{to_char(answer)}"
        })
    reactant, options, dist, _, _ = context[-1]
    messages.append({
        "role": "user",
        "content": prompt_single(reactant, options, dist, None, func=func, test=True, useD=useD, catalyst=catalyst_name)
    })

    return messages


def prompt_in_context(context, func=lambda x:x, useD=False, conf=True, name=''):
    if '10b9' in name:
        system_message = system_message_9
        catalyst_name = Imidazo_Catalyst
    elif '9b8a' in name:
        system_message = system_message_10
        catalyst_name = None
    else:
        system_message = SYSTEM_MESSAGE
        catalyst_name = None
    
    prompt = []
    if conf:
        prompt.append(system_message)
    else:
        prompt.append(SYSTEM_MESSAGE_NOCONF)

    for idx, item in enumerate(context[:-1]):
        prompt.append(f'Example No.{idx+1}:')
        reactant, options, dist, answer, confidence = item
        context_prompt = prompt_single(reactant, options, dist, None, func=func, test=False, useD=useD)
        context_prompt += f"{to_char(answer)} {confidence}" if conf else f"{to_char(answer)}"
        prompt.append(context_prompt)
    
    prompt.append(f'Please answer the following question based on the examples above.')
    reactant, options, dist, _, _ = context[-1]
    query_prompt = prompt_single(reactant, options, dist, None, func=func, test=True, useD=useD, catalyst=catalyst_name)
    prompt.append(query_prompt)

    prompt = '\n\n'.join(prompt)
    return prompt


def prompt_single(reactant, options, dist, answer, func=lambda x:x, test=False, useD=False, catalyst=None):
    question = ''
    if query_info and test:
        question += 'The given reaction is a Nickel-catalyzed cross-couplings reaction, which forms carbon-carbon (C-C) and carbon-heteroatom (C-X, where X can be O, N, S, etc.) bonds.\n'
    question += 'Question: Which of the following is the major product of the organic reaction '
    reactant, reactant_iupac = reactant
    if '.' in reactant:
        reactant_iupac = reactant_iupac.split('|')
        reactants = reactant.split('.')
        if len(reactant_iupac) != len(reactants) or reactant_iupac==reactants:
            reactant_iupac = [func(s) for s in reactants]
        reactants = [i if i==j else f'{i}<{j}>' for i, j in zip(reactant_iupac, reactants)]
        reactants = ' '.join([', '.join(reactants[:-1]), 'and', reactants[-1]])
        if catalyst:
            question += f'between {reactants}, with {catalyst} as catalyst?'
        else:
            question += f'between {reactants}?'
    else:
        if reactant == reactant_iupac:
            reactant_iupac = func(reactant)
        reactant = f'{reactant_iupac}<{reactant}>'
        question += f'of {func(reactant)} in water?'

    choice = ''
    for idx, (opt, dis) in enumerate(zip(options, dist)):
        smiles, iupac = opt
        if '.' in smiles:
            iupac = iupac.split('|')
            smiles = smiles.split('.')
            if len(iupac) != len(smiles) or smiles==iupac:
                iupac = smiles2iupac(smiles)
            opt = [f'{i}<{j}>' for i, j in zip(iupac, smiles)]
            opt = ' '.join([', '.join(opt[:-1]), 'and', opt[-1]])
        else:
            if smiles==iupac:
                iupac = smiles2iupac(smiles)
            opt = f'{iupac}<{smiles}>'
        choice += f'{to_char(idx)}. {opt}\n'
    if useD:
        choice += f'{to_char(len(options))}. None of above'

    if test or (not answer):
        answer = f'Answer: '
    else:
        answer = answer if answer>=0 else len(options)
        answer = f'Answer: {to_char(answer)}'

    return '\n'.join((question, choice, answer))

