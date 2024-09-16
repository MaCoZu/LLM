import os
import sys
from datetime import datetime, timedelta
import argparse
import requests
from urllib.parse import urlparse
from dotenv import load_dotenv
from gnews import GNews
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document

load_dotenv('.env')

def date_x_month_ago(months):
    '''returns (year, month, day) of the date x month from today '''
    today = datetime.now()
    date_x = today - timedelta(days=30 * months)
    date_x = (date_x.year, date_x.month, date_x.day)
    return date_x

def get_headlines(keyword, month_from_today, max_results):
    ''' returns [{title: v, description: v, published date: v, url: v, publisher:v }] from Google News API '''
    google_news = GNews(language='en',
                        country='US',
                        start_date=date_x_month_ago(month_from_today),
                        max_results=max_results,
                        end_date=None,
                        exclude_websites=[
                            'yahoo.com',
                            'foxnews.com',
                            'msn.com',
                        ])

    return google_news.get_news(keyword)

# prompting to perform news summary
prompt_template = """Generate summary for the text below, using the following steps:
                1. Summary consists of maximum 100 words
                2. If the text cannot be found or error, return: "Content empty"
                3. Use only materials from the text supplied.
                4. Do not talk about the text as if written by somebody else. Write it like the author would have done.
                5. Name causes and responsibilities from the conflicts described.
                6. If statistics are available, include them in the summary. Ensuring that all statistical data is accurately represented.
                7. Create the summary in English

                "{text}"
                SUMMARY:"""

#  save as a template
prompt = PromptTemplate.from_template(prompt_template)

# declare LLM model
llm = ChatOpenAI(temperature=0,
                 model_name="gpt-3.5-turbo-1106",
                 api_key=os.environ['OPENAI_API_KEY'])


def generate_summary(search_keyword, month_from_today, max_results=5):
    # Get news headlines
    news_by_keyword = get_headlines(search_keyword, month_from_today, max_results)

    # Define output file path
    output_folder = "summaries"
    os.makedirs(output_folder, exist_ok=True)  # Create 'summaries' folder if it doesn't exist
    output_file = os.path.join(output_folder, f"{search_keyword}_summary.txt")

    summaries = []

    # Write summaries to the file
    with open(output_file, 'w') as f:
        for item in news_by_keyword:
            try:
                # Extract news content
                loader = UnstructuredURLLoader(urls=[item['url']])
                docs = loader.load()

                if docs:
                    # Summarize using stuff for easy processing
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    summary = chain.run(docs)

                    # find the original url not the 'news.google.com/rss...' thing
                    original_url = requests.head(item['url'], allow_redirects=True)

                    # remove the publisher name from the title
                    index = item['title'].rfind(' - ')
                    only_title = item['title'][:index]

                    # Write summary to the file
                    f.write(only_title + '\n')
                    f.write(item['publisher']['title'] + ' ' + item['published date'][:-13] + '\n\n')
                    f.write(original_url.url)
                    f.write('\n\n')
                    f.write(summary)
                    f.write('\n' + '_' * 20)
                    f.write('\n\n\n\n')

                    summaries.append(summary)
                else:
                    print(f"No content found for URL: {item['url']}")
            except Exception as e:
                print(f"Error processing URL {item['url']}: {str(e)}")
                continue  # Skip this URL and move to the next one

    return summaries

def generate_final_summary(summaries):
    combined_text = " ".join(summaries)
    docs = [Document(page_content=combined_text)]

    final_summary_prompt = PromptTemplate.from_template(
        """As an expert analyst, carefully review and consolidate the following summaries into a comprehensive final summary:

        {text}

        Your task is to:
        1. Create a concise summary of approximately 300 words.
        2. Fact-check and cross-reference the information from different sources.
        3. Highlight the most important and recurring points across the summaries.
        4. Provide a balanced view of the topic, considering different perspectives if present.
        5. Ensure all statistical data is accurately represented and consistent across sources.
        6. Identify and resolve any contradictions or inconsistencies in the information.
        7. Present the information in a clear, coherent, and objective manner.

        FINAL SUMMARY:"""
    )

    chain = load_summarize_chain(llm, chain_type="stuff", prompt=final_summary_prompt)
    final_summary = chain.run(docs)

    return final_summary

def main():
    parser = argparse.ArgumentParser(description="Generate news digest")
    parser.add_argument("keyword", help="Keyword for news search")
    parser.add_argument("months", type=int, help="Number of months from today to search")
    parser.add_argument("--max_results", type=int, default=5, help="Maximum number of news results (default: 5)")

    args = parser.parse_args()

    summaries = generate_summary(args.keyword, args.months, args.max_results)
    final_summary = generate_final_summary(summaries)

    # Write final summary to file
    output_folder = "summaries"
    final_output_file = os.path.join(output_folder, f"{args.keyword}_final_summary.txt")
    with open(final_output_file, 'w') as f:
        f.write(final_summary)

    print(f"News digest generated. Individual summaries and final summary saved in the 'summaries' folder.")

if __name__ == "__main__":
    main()
