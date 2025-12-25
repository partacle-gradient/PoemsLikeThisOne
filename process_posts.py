import os
import glob
import json
import cv2
import pytesseract
import pandas as pd
import spacy
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import logging

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# --- CONFIGURATION ---
INPUT_DIR = './scraped_raw_sample'
OUTPUT_DIR = './scraped_processed_sample'

# Thresholds for classifying "Art" vs "Poem"
MIN_WORD_COUNT = 15       # Minimum valid words to be considered a poem
MIN_CONFIDENCE = 50       # Minimum OCR confidence for a word to count

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("poem_processor.log", mode='w'), # Overwrite log on each run
        logging.StreamHandler()
    ]
)

# Load NLP model
try:
    NLP = spacy.load("en_core_web_sm")
except OSError:
    logging.error("Spacy model not found. Please run: python -m spacy download en_core_web_sm")
    exit()

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- HELPER FUNCTIONS ---

def is_poem_image(df, filename="unknown"):
    """
    Determines if an image is a poem based on text density and confidence.
    Expects the Tesseract DataFrame as input.
    """
    if df.empty: 
        logging.debug(f"[{filename}] DataFrame empty.")
        return False
    
    # Clean data: Coerce confidence to numeric, handling errors
    df['conf'] = pd.to_numeric(df['conf'], errors='coerce').fillna(-1)
    
    # Filter for valid words:
    valid_words = df[
        (df['conf'] > MIN_CONFIDENCE) & 
        (df['text'].str.strip().str.len() > 2)
    ]
    
    word_count = len(valid_words)
    avg_conf = valid_words['conf'].mean() if word_count > 0 else 0
    
    is_poem = word_count >= MIN_WORD_COUNT
    
    # Log the decision stats
    log_msg = f"[{filename}] Stats: Words={word_count}, AvgConf={avg_conf:.1f}% -> {'POEM' if is_poem else 'IMAGE/ART'}"
    if is_poem:
        logging.info(log_msg)
    else:
        logging.info(log_msg) # Changed to INFO so you can see why things are skipped
        
    return is_poem

def get_image_layout(image_path):
    """
    Reads image, preprocesses it, and runs OCR to get text + coordinates.
    Returns: DataFrame, Image Height
    """
    img = cv2.imread(image_path)
    if img is None: 
        logging.warning(f"Could not read image: {image_path}")
        return None, None
    
    height, width, _ = img.shape
    
    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    try:
        data = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT, config='--psm 6')
        df = pd.DataFrame(data)
        
        # Initial cleanup
        df = df[df['conf'] != '-1']
        df['text'] = df['text'].astype(str).str.strip()
        df = df[df['text'] != '']
        return df, height
    except Exception as e:
        logging.error(f"OCR Error on {image_path}: {e}")
        return None, None

def extract_content_from_page(df, page_height, is_first_page=False, is_last_page=False):
    """
    Splits text into Header, Body, and Footer based on vertical position.
    """
    if df.empty: return {"header": [], "body": [], "footer": []}

    # Define vertical zones
    header_thresh = page_height * 0.20
    footer_thresh = page_height * 0.80

    header_lines = []
    body_lines = []
    footer_lines = []

    # Group words into lines
    df['line_group'] = (df['top'] // 20) * 20
    
    for _, group in df.groupby('line_group'):
        text_line = " ".join(group.sort_values('left')['text'])
        avg_top = group['top'].mean()
        
        # Zone Logic
        if is_first_page and avg_top < header_thresh:
            header_lines.append(text_line)
        elif is_last_page and avg_top > footer_thresh:
            footer_lines.append(text_line)
        else:
            body_lines.append(text_line)

    return {
        "header": header_lines,
        "body": body_lines,
        "footer": footer_lines
    }

def find_author_and_title(header_lines, footer_lines):
    """
    Uses NLP logic to determine which text is Title vs Author.
    Returns: title, author, header_remainder, footer_remainder
    """
    author = "Unknown"
    title = "Untitled"
    header_remainder = []
    footer_remainder = []

    def is_person_entity(text):
        doc = NLP(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                if len(text.split()) < 6: 
                    return True
        return False

    # STRATEGY 1: Check Footer
    author_idx = -1
    for i in range(len(footer_lines) - 1, -1, -1):
        if is_person_entity(footer_lines[i]):
            author = footer_lines[i]
            author_idx = i
            break
            
    for i, line in enumerate(footer_lines):
        if i != author_idx:
            footer_remainder.append(line)
            
    # STRATEGY 2: Check Header
    title_found = False
    
    for line in header_lines:
        clean_line = line.strip()
        lower_line = clean_line.lower()
        
        if lower_line.startswith("for ") or lower_line.startswith("to "):
            header_remainder.append(clean_line)
            continue
            
        if author == "Unknown":
            if "by " in lower_line:
                author = clean_line.replace("By ", "").replace("by ", "")
                continue
            
            if is_person_entity(clean_line):
                author = clean_line
                continue 
        
        if not title_found:
            title = clean_line
            title_found = True
            continue
            
        header_remainder.append(clean_line)

    return title, author, header_remainder, footer_remainder

# --- CORE LOGIC ---

def process_post(post_id, file_paths):
    logging.info(f"--- Processing Post ID: {post_id} ---")
    
    images = sorted([f for f in file_paths if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    txt_file = next((f for f in file_paths if f.endswith('.txt')), None)
    caption = ""
    if txt_file:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                caption = f.read()
        except Exception:
            logging.warning(f"Could not read text file for {post_id}")

    full_body_parts = []
    potential_header = [] 
    potential_footer = [] 
    
    is_valid_poem = False

    for i, img_path in enumerate(images):
        base_name = os.path.basename(img_path)
        df, height = get_image_layout(img_path)
        
        if df is None: continue

        # Pass filename for logging
        if is_poem_image(df.copy(), filename=base_name): 
            is_valid_poem = True

        is_first = (i == 0)
        is_last = (i == len(images) - 1)

        extracted = extract_content_from_page(df, height, is_first, is_last)
        
        if is_first:
            potential_header.extend(extracted['header'])
        
        full_body_parts.extend(extracted['body'])
        
        if is_last:
            potential_footer.extend(extracted['footer'])

    if not is_valid_poem:
        logging.info(f"Skipping {post_id}: No images classified as text/poems.")
        return None 

    title, author, header_rem, footer_rem = find_author_and_title(potential_header, potential_footer)
    
    logging.info(f"Found Metadata -> Title: '{title}' | Author: '{author}'")
    
    final_body_parts = header_rem + full_body_parts + footer_rem

    return {
        "id": post_id,
        "title": title,
        "author": author,
        "body": "\n".join(final_body_parts),
        "caption_excerpt": caption[:200], 
        "original_files": [os.path.basename(x) for x in images]
    }

def main():
    groups = defaultdict(list)
    
    logging.info(f"Scanning {INPUT_DIR} recursively...")
    
    file_count = 0
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.txt')):
                if '_UTC' in file:
                    base_part = file.split('_UTC')[0]
                    post_id = base_part + '_UTC'
                    full_path = os.path.join(root, file)
                    groups[post_id].append(full_path)
                    file_count += 1

    logging.info(f"Found {file_count} files across {len(groups)} unique posts.")
    logging.info("Starting processing loop...")

    processed_count = 0
    skipped_count = 0
    
    # Iterate with TQDM for visual progress bar, but log details to file
    for post_id, file_paths in tqdm(groups.items()):
        try:
            result = process_post(post_id, file_paths)
            
            if result:
                out_name = f"{post_id}.json"
                out_path = os.path.join(OUTPUT_DIR, out_name)
                
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
                processed_count += 1
            else:
                skipped_count += 1
                
        except Exception as e:
            logging.error(f"CRITICAL ERROR processing {post_id}: {e}", exc_info=True)

    logging.info("-" * 30)
    logging.info("PROCESSING COMPLETE")
    logging.info(f"Poems Extracted: {processed_count}")
    logging.info(f"Images Skipped: {skipped_count}")
    logging.info(f"Log file saved to: poem_processor.log")

if __name__ == "__main__":
    main()