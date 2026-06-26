"""
Layer 1: Deep Archive Macro-Sentiment Scraper (Pagination Engine)
Author: Saptarshi Dutta | AI4Invest Pipeline
Objective: Autonomously navigate archive pages to build a multi-cycle dataset.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

def main():
    print("--- INITIATING DEEP ARCHIVE SCRAPER (PAGINATION MODE) ---")
    
    master_dates = []
    master_headlines = []
    
    # ---------------------------------------------------------
    # THE PAGINATION ENGINE
    # ---------------------------------------------------------
    MAX_PAGES = 2500  # We test with 5 pages first to ensure stability
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for page in range(1, MAX_PAGES + 1):
        # Dynamically inject the page number into the URL
        url = f"https://economictimes.indiatimes.com/markets/stocks/news?pageno={page}"
        print(f"\n[PAGE {page}/{MAX_PAGES}] Pinging endpoint: {url}")
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            # If the website blocks us or the page doesn't exist, stop the loop safely
            if response.status_code != 200:
                print(f"[WARNING] Server returned status code {response.status_code}. Stopping scraper.")
                break
                
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = soup.find_all('div', class_='eachStory') 
            
            page_headline_count = 0
            
            for article in articles:
                headline_tag = article.find('h3')
                if not headline_tag:
                    continue
                headline_text = headline_tag.text.strip()
                
                time_tag = article.find('time')
                if not time_tag:
                    time_tag = article.find('span', class_='date-format')
                    
                date_text = time_tag.text.strip() if time_tag else "UNKNOWN_DATE"
                
                if date_text != "UNKNOWN_DATE":
                    master_headlines.append(headline_text)
                    master_dates.append(date_text)
                    page_headline_count += 1
            
            print(f" -> Extracted {page_headline_count} headlines from Page {page}. Total so far: {len(master_headlines)}")
            
            # ANTI-BAN PROTOCOL: Wait 2 to 4 seconds before turning the page
            # If we hit the server too fast, they will permanently ban your IP address.
            sleep_time = random.uniform(2.0, 4.0)
            time.sleep(sleep_time)

        except Exception as e:
            print(f"[ERROR] Failed to scrape {url}: {e}")

    # Build and Clean the final dataset
    print("\nSynthesizing Master CSV...")
    df = pd.DataFrame({
        'Date': master_dates,
        'Headline': master_headlines
    })
    
    print("Executing strict chronological formatting...")
    df['Date'] = df['Date'].str.replace(' IST', '', regex=False)
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y, %I:%M %p', errors='coerce').dt.date
    
    df = df.dropna(subset=['Date'])
    
    output_name = "timestamped_historical_headlines.csv"
    df.to_csv(output_name, index=False)
    
    print("\n==================================================")
    print("   SCRAPING COMPLETE")
    print("==================================================")
    print(f"Total Timestamped Rows Saved : {len(df)}")
    print(f"File stored securely as      : {output_name}")

if __name__ == "__main__":
    main()