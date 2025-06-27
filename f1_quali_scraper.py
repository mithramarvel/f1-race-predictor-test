import requests
from bs4 import BeautifulSoup
import pandas as pd
import urllib.parse
from io import StringIO

def clean_race_name(url):
    name = url.split('/')[-1].replace('_', ' ')
    name = urllib.parse.unquote(name)  # Fix % encoding
    name = name.replace("Grand Prix", "").strip()
    return name + " Grand Prix"
# ========================
# Utility Function: Convert time to seconds
# ========================
def time_to_seconds(t):
    try:
        if isinstance(t, str) and ':' in t:
            mins, secs = t.split(':')
            return round(int(mins) * 60 + float(secs), 3)
        return None
    except:
        return None

# ========================
# Function: Extract qualifying table from race page
# ========================
def find_table_after_header(header):
    # Find the next <table class="wikitable"> that appears *anywhere* after this header
    for sibling in header.find_all_next():
        if sibling.name == 'table' and 'wikitable' in sibling.get('class', []):
            return sibling
        # Optional: stop search at the next section (another header)
        if sibling.name in ['h2', 'h3', 'h4']:
            break
    return None
def extract_qualifying_table(soup):
   
    qualifying_table = None
    for header in soup.find_all(['h2', 'h3', 'h4']):
        header_text = header.get_text().lower()
        #print(f"🔎 Checking header: {header_text}")

        if ("qualifying classification" in header_text or "qualifying results" in header_text) and "sprint" not in header_text:
            qualifying_table=find_table_after_header(header)
            if qualifying_table:
                break
    else:
        #print("❌ No qualifying table found under this header.")
        return None
    def match_col(columns, keyword):
        """Returns the first column that contains the keyword (case-insensitive)"""
        for col in columns:
            if keyword.lower() in col.lower():
                return col
        return None

    table_text = qualifying_table.text
    if not all(q in table_text for q in ['Q1', 'Q2', 'Q3']):
        return None  
    try:  
            df = pd.read_html(StringIO(str(qualifying_table)))[0]

            # Flatten multi-index if needed
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [' '.join(col).strip() for col in df.columns.values]

            columns = df.columns.tolist()

            # Try matching both normal and sprint headers
            driver_col = match_col(columns, 'Driver')
            team_col = match_col(columns, 'Constructor')
            q1_col = match_col(columns, 'Q1') 
            q2_col = match_col(columns, 'Q2') 
            q3_col = match_col(columns, 'Q3')
            grid_col = match_col(columns, 'Final grid') or match_col(columns,'Grid')

            if not all([driver_col, team_col, q1_col, q2_col, q3_col, grid_col]):
                print("⚠️ Table found but required columns missing. Columns were:", columns)
                return None  # Try next table

            # Clean and process
            df_clean = df[[driver_col, team_col, q1_col, q2_col, q3_col, grid_col]].copy()
            df_clean.columns = ['Driver', 'Team', 'Q1', 'Q2', 'Q3', 'Grid_Pos']

            df_clean = df_clean[df_clean['Driver'].notna()]
            df_clean = df_clean[~df_clean['Driver'].astype(str).str.contains('107%|Driver', na=False)]
            df_clean = df_clean[~df_clean['Grid_Pos'].astype(str).str.strip().eq('—')]

            df_clean['Q1_sec'] = df_clean['Q1'].apply(time_to_seconds)
            df_clean['Q2_sec'] = df_clean['Q2'].apply(time_to_seconds)
            df_clean['Q3_sec'] = df_clean['Q3'].apply(time_to_seconds)

            return df_clean

    except Exception as e:
        print(f"⚠️ Error parsing qualifying table: {e}")
        return None
def extract_race_date(soup):
    # Try to find date in infobox table rows
    infobox = soup.find("table", class_="infobox")
    if infobox:
        for row in infobox.find_all("tr"):
            header = row.find("th")
            data = row.find("td")
            if header and "Date" in header.text and data:
                return data.text.strip()
    return "Unknown"
# ========================
# Main Scraper Logic
# ========================
BASE_URL = 'https://en.wikipedia.org/wiki/2025_Formula_One_World_Championship'
response = requests.get(BASE_URL)
soup = BeautifulSoup(response.content, 'lxml')

# Step 1: Find all 2025 race links
race_links = []
for a in soup.find_all('a', href=True):
    href = a['href']
    if '2025_' in href and '_Grand_Prix' in href and 'redlink=1' not in href and 'speedway' not in href.lower():
        full_link = 'https://en.wikipedia.org' + href
        if full_link not in race_links:
            race_links.append(full_link)

print(f"Found {len(race_links)} race pages.")

# Step 2: Scrape each race
all_race_data = []

for race_url in race_links:
    try:
        race_res = requests.get(race_url)
        race_soup = BeautifulSoup(race_res.content, 'lxml')

        # Race name & date
        race_name = clean_race_name(race_url)
        race_date = extract_race_date(race_soup)
        print(f"Scraping: {race_name}...")

        # Extract Qualifying data
        qualifying_df = extract_qualifying_table(race_soup)
        if qualifying_df is not None:
            qualifying_df['Race'] = race_name
            qualifying_df['Date'] = race_date
            all_race_data.append(qualifying_df)
        else:
            print(f"⚠️  No qualifying data found for {race_name}")
    
    except Exception as e:
        print(f"❌ Error scraping {race_url}: {e}")

# Step 3: Save to CSV
if all_race_data:
    final_df = pd.concat(all_race_data, ignore_index=True)
    final_df.to_csv("f1_2025_quali_data.csv", index=False)
    print("✅ Saved to f1_2025_quali_data.csv")
else:
    print("⚠️ No data was scraped.")
 