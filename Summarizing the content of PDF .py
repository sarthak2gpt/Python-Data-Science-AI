#!/usr/bin/env python
# coding: utf-8

# In[3]:


import PyPDF2
from nltk.tokenize import sent_tokenize

def extract_text(pdf_path):
    pdf_file = open(pdf_path, 'rb')
    reader = PyPDF2.PdfFileReader(pdf_file)
    num_pages = reader.numPages

    text = []
    for page_num in range(num_pages):
        page = reader.getPage(page_num)
        page_text = page.extract_text()
        text.append(page_text)

    pdf_file.close()
    return text

def summarize_text(text, word_limit):
    summaries = []
    for page_num, page_text in enumerate(text):
        sentences = sent_tokenize(page_text)
        summary = ''
        word_count = 0
        for sentence in sentences:
            if word_count + len(sentence.split()) <= word_limit:
                summary += '- ' + sentence + '\n'
                word_count += len(sentence.split())
        summaries.append((page_num + 1, summary))
    return summaries

def main():
    pdf_path = 'Blockchain_recommendation.pdf'
    word_limit = int(input("Enter the word limit for each page's summary: "))
    text = extract_text(pdf_path)
    summaries = summarize_text(text, word_limit)
    if len(summaries) != len(text):
        print("Error: The number of summaries does not match the number of pages.")
        return
    for page_num, summary in summaries:
        print(f'Page {page_num}:')
        print(summary)
        print()

if __name__ == '__main__':
    main()


# In[ ]:


# Explanation of code 

# In this script, after extracting the text from each page using PyPDF2, 

# the user is prompted to enter the word limit for each page's summary. The summarize_text function then generates the summaries,

# ensuring that the word count of each summary does not exceed the specified limit. The summaries, along with the page numbers,

# are printed out individually for each page.

