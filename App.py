# Created by shuchi_liao at 6/10/
# Q&A trivial app based on Sematic search + gpt model: This is a demo app, which only utilizes a short article
# from wikipedia as reference. for more realistic model, we will need more functions/procedures including
# text embedding using gpt models and vector search and comparison, then use the most relevant documents as
# prompt fed to gpt model for answers. Detailed info can be found in OpenAI documentation.
# References:
# https://platform.openai.com/docs/api-reference
# https://github.com/openai/openai-cookbook/tree/main/examples

import os
import openai
import re
import tiktoken
import streamlit as st

st.title('Demo Q&A system')

''' Please note I already set a 10 dollar limit for all API calls!'''
openai.api_key = os.environ["OPENAI_API_KEY"]


GPT_MODEL = "gpt-3.5-turbo"
# GPT_MODEL = "gpt-4"
# More accurate result may be achieved by using gpt-4 model but costs more

'''I limited the token numbers to 2048, which saves me money.'''
MAX_TOKENS = 2048  # GPT documentation says per request no more than 4096-500 tokens.

HEADER = 'Use the provided context to answer the question as truthfully as possible and if the answer is not ' \
         'contained within the text below, say "I could not find an answer. It seems not a question about Sam Altman."'


@st.cache_data
def clean_text_file(file, skip_lines=0):
    """ Read and clean text file"""
    with open(file, 'r') as f:
        # skip first n lines which contains irrelevant text
        for _ in range(skip_lines):
            next(f)
        # read the remaining file contents
        contents = f.read()

    # remove reference number pattern of '[num]'
    clean_contents = re.sub(r'\[\d+\]', '', contents)

    # Remove empty lines
    cleaned_lines = [line for line in clean_contents.split('\n') if line.strip()]
    cleaned_text = '\n'.join(cleaned_lines)

    # other clean process like special character/stopword removal, Lemmatization, lower casing
    # may be applied but not required for GPT models

    return cleaned_text


def num_tokens(context, model= GPT_MODEL):
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(context))


def construct_prompt(query, context, header=HEADER, token_budget = MAX_TOKENS):
    """ Construct a prompt with user question and provided context"""
    message = header + f'\n\nWikipedia article section:\n"""\n{context}\n"""'
    qeustion = f"\n\nQuestion: {query}"
    prompt = message + qeustion

    # I limited the token numbers to 2048, which saves me money and means you cannot ask a very long question.
    if num_tokens(prompt) > token_budget:
        raise ValueError('Token numbers exceed maximum tokens allowed')
    return prompt


def answer(query, context, header=HEADER, token_budget=MAX_TOKENS, model=GPT_MODEL, print_message=False):
    """ Answer user query with gpt model based on the context"""
    prompt = construct_prompt(query, context, header, token_budget)
    if print_message:
        print(prompt)
    messages = [
        {"role": "system", "content": "You answer questions about Sam Altman."},
        {"role": "user", "content": prompt},
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response["choices"][0]["message"]["content"]


# This line should only run once when loading the app for the first time due to st.cache_data
contexts = clean_text_file('./Sam Altman.txt', 28)
# print(num_tokens(contexts))
# Since the total tokens for the context is less than 1600, we don't need to chunk the text into smaller pieces


st.write("Ask questions about Sam Altman")
question = st.text_input("Enter your question:")
if st.button("Get Answer"):
    ans = answer(question, contexts)
    st.write("Answer:", ans)







