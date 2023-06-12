# Created by simonliao at 6/10/
# embedding document using openAI embedding model
# References:


import os
import openai
import re
import tiktoken

# Please note I already set a 10 dollar limit for API calls!
openai.api_key = os.environ["OPENAI_API_KEY"]
GPT_MODEL = 'gpt-4'
MAX_TOKENS = 1600 # by GPT documentation

def clean_text_file(file):
    """ Read / clean text file"""
    with open(file, 'r') as f:
        contents = f.read()

    # remove reference number pattern of '[num]'
    clean_contents = re.sub(r'\[\d+\]', '', contents)

    # Remove empty lines
    cleaned_lines = [line for line in clean_contents.split('\n') if line.strip()]
    cleaned_text = '\n'.join(cleaned_lines)

    # other clean process like stopword removal, Lemmatization, lower casing
    # may be applied but not required for GPT models

    return cleaned_text

def num_tokens(text, model= GPT_MODEL):
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def halved_by_delimiter(string, delimiter="\n"):
    """Split a string in two, on a delimiter, trying to balance tokens on each side."""
    chunks = string.split(delimiter)
    if len(chunks) == 1:
        return [string, ""]  # no delimiter found
    elif len(chunks) == 2:
        return chunks  # no need to search for halfway point
    else:
        total_tokens = num_tokens(string)
        halfway = total_tokens // 2
        best_diff = halfway
        for i, chunk in enumerate(chunks):
            left = delimiter.join(chunks[: i + 1])
            left_tokens = num_tokens(left)
            diff = abs(halfway - left_tokens)
            if diff >= best_diff:
                break
            else:
                best_diff = diff
        left = delimiter.join(chunks[:i])
        right = delimiter.join(chunks[i:])
        return [left, right]


def truncated_string(
    string,
    model,
    max_tokens,
    print_warning = True):
    """Truncate a string to a maximum number of tokens."""
    encoding = tiktoken.encoding_for_model(model)
    encoded_string = encoding.encode(string)
    truncated_string = encoding.decode(encoded_string[:max_tokens])
    if print_warning and len(encoded_string) > max_tokens:
        print(f"Warning: Truncated string from {len(encoded_string)} tokens to {max_tokens} tokens.")
    return truncated_string


def chunk_text(
    string,
    max_tokens=MAX_TOKENS,
    model= GPT_MODEL,
    max_recursion = 5,
) :
    """
    Split text into a list of strings, each with no more than max_tokens.
    """
    num_tokens_in_string = num_tokens(string)
    # if length is fine, return string
    if num_tokens_in_string <= max_tokens:
        return [string]
    # if recursion hasn't found a split after X iterations, just truncate
    elif max_recursion == 0:
        return [truncated_string(string, model=model, max_tokens=max_tokens)]
    # otherwise, split in half and recurse
    else:
        text = string
        for delimiter in ["\n\n", "\n", ". "]:
            left, right = halved_by_delimiter(text, delimiter=delimiter)
            if left == "" or right == "":
                # if either half is empty, retry with a more fine-grained delimiter
                continue
            else:
                # recurse on each half
                results = []
                for half in [left, right]:
                    half_strings = chunk_text(
                        half,
                        max_tokens=max_tokens,
                        model=model,
                        max_recursion=max_recursion - 1,
                    )
                    results.extend(half_strings)
                return results
    # otherwise no split was found, so just truncate (should be very rare)
    return [truncated_string(string, model=model, max_tokens=max_tokens)]

contexts = clean_text_file('./Sam Altman.txt')
contexts = chunk_text(contexts)
# Since the total tokens for the context is less than 1600, we don't need to chunk the text


EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's best embeddings as of Apr 2023

# Calculate embeddings for the context
response = openai.Embedding.create(model=EMBEDDING_MODEL, input=contexts)
embedding = response["data"][0]["embedding"]




