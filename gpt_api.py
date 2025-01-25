# gpt_api.py
from openai import OpenAI
import json
import time
from tqdm import tqdm

# Initialize OpenAI client
client = OpenAI()

def get_completion(prompt, model="gpt-3.5-turbo", max_retries=3, retry_delay=5):
    """Get completion with retry logic"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {str(e)}")
                return None
            print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

def process_prompts(input_file, output_file="results.json"):
    """Process all prompts and save results"""
    # Load prompts
    with open(input_file, 'r') as f:
        prompts_data = json.load(f)
    
    results = []
    
    # Process each prompt with progress bar
    for prompt_data in tqdm(prompts_data, desc="Processing prompts"):
        result = {
            "index": prompt_data["index"],
            "image_name": prompt_data["image_name"],
            "response": get_completion(prompt_data["prompt"])
        }
        results.append(result)
        
        # Save results after each completion (in case of interruption)
        with open(output_file, 'w') as f:
            json.dump(results, f)
        
        # Optional: Add a small delay to avoid rate limits
        time.sleep(0.5)
    
    print(f"Completed processing {len(results)} prompts. Results saved to {output_file}")
    return results

if __name__ == "__main__":
    results = process_prompts(input_file='esnlive_sample_prompts.jsonl')