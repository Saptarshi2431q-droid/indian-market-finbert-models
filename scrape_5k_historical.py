"""
Deep-Archive Historical Scraper (5k Target Fix)
Author: Saptarshi Dutta | AI4Invest Pipeline
Objective: Scrape 5,000+ historical market headlines using verified HTML layouts.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

def scrape_et_archives_multi(target_headline_count=5000):
    print(f"Initializing Bulletproof Scraper. Target: {target_headline_count} headlines...")
    
    # VERIFIED COMPATIBLE ET CATEGORY IDs: 
    # Stocks, IPOs, Corporate Earnings, Technicals/F&O, Forex, Commodities
    category_ids = ["2146843", "14655708", "43036443", "2146844", "24707641", "24707640"]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    
    scraped_data = []
    
    for cat_id in category_ids:
        page_num = 1
        retries = 0
        base_url = f"https://economictimes.indiatimes.com/markets/stocks/news/articlelist/{cat_id}.cms?curpg="
        
        print(f"\n--- Switching to ET Category ID: {cat_id} ---")
        
        while len(scraped_data) < target_headline_count:
            url = base_url + str(page_num)
            print(f"Scraping Category {cat_id} | Page {page_num} | Current Count: {len(scraped_data)}...")
            
            try:
                response = requests.get(url, headers=headers, timeout=20)
                
                if response.status_code == 404:
                    print(f"Reached end of archive for category {cat_id} at page {page_num}. Moving to next...")
                    break
                
                if response.status_code != 200:
                    print(f"Warning: Status {response.status_code}. Retrying...")
                    time.sleep(5)
                    retries += 1
                    if retries > 3:
                        break 
                    continue
                    
                soup = BeautifulSoup(response.text, 'html.parser')
                stories = soup.find_all('div', class_='eachStory')
                
                if not stories:
                    print("No standard stories found on this page layout. Moving to next category...")
                    break
                    
                for story in stories:
                    headline_tag = story.find('h3')
                    if headline_tag and headline_tag.text:
                        scraped_data.append({
                            "Headline": headline_tag.text.strip(),
                            "Source": "Economic Times",
                            "Category_ID": cat_id
                        })
                        
                    if len(scraped_data) >= target_headline_count:
                        break
                        
                page_num += 1
                retries = 0 
                time.sleep(random.uniform(1.0, 3.0))
                
            except requests.exceptions.Timeout:
                print(f"Server timed out on page {page_num}. Retrying...")
                retries += 1
                time.sleep(5)
                if retries > 3:
                    print("Too many timeouts. Moving to next category...")
                    break
                    
            except Exception as e:
                print(f"Unexpected Error on page {page_num}: {e}. Moving to next category...")
                break
                
        if len(scraped_data) >= target_headline_count:
            break

    df = pd.DataFrame(scraped_data)
    df = df.drop_duplicates(subset=['Headline'])
    
    print(f"\n[SUCCESS] Extracted {len(df)} historical headlines.")
    return df

if __name__ == "__main__":
    dataset = scrape_et_archives_multi(target_headline_count=5000)
    dataset.to_csv("historical_5k_headlines.csv", index=False)
    print("Dataset saved to 'historical_5k_headlines.csv'")