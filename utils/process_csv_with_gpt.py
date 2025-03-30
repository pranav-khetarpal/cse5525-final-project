import os
import pandas as pd
import csv
import argparse
import openai
import time
import json
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")


# Define Pydantic models for structured output
class TickerExtraction(BaseModel):
    ticker: str


class BatchTickerExtraction(BaseModel):
    results: List[TickerExtraction]


# Function to get ticker symbols using GPT-4o in batch with structured output
def get_tickers_from_gpt_batch(
    texts: List[str], batch_size: int = 10
) -> Dict[str, str]:
    """Process multiple texts in batches to extract ticker symbols."""
    result_dict = {}

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        print(f"Processing batch {i//batch_size + 1} of {len(texts)//batch_size}")
        # Create the JSON schema for structured output
        extraction_schema = {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "The ticker symbol prefixed with '$'. If no ticker can be identified, use '$UNKNOWN'.",
                            }
                        },
                        "required": ["ticker"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["results"],
            "additionalProperties": False,
        }

        try:
            client = OpenAI(api_key=api_key)

            # Create messages for batch processing
            system_message = """Extract the MOST RELEVANT stock ticker symbol from each text. Return ONLY ONE ticker symbol per text, even if multiple tickers are mentioned. If no ticker can be identified, use '$UNKNOWN'.

Your response must be valid JSON in the following format:
{
  "results": [
    {"ticker": "$TICKER1"},
    {"ticker": "$TICKER2"},
    ...
  ]
}

Where $TICKER1, $TICKER2, etc. are the ticker symbols you extracted (with $ prefix)."""
            user_content = (
                "Extract ticker symbols from the following texts, one per text:\n\n"
            )
            for j, text in enumerate(batch):
                user_content += f"{j+1}. {text}\n"

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                max_tokens=1000,
            )

            # Parse the structured response
            extraction_data = json.loads(response.choices[0].message.content)

            # Make sure we have the right number of results
            if len(extraction_data["results"]) != len(batch):
                print(
                    f"Warning: Expected {len(batch)} tickers, but got {len(extraction_data['results'])}"
                )
                # Handle the mismatch (should be rare with structured output)
                if len(extraction_data["results"]) < len(batch):
                    # Add UNKNOWN for missing results
                    for _ in range(len(batch) - len(extraction_data["results"])):
                        extraction_data["results"].append({"ticker": "$UNKNOWN"})
                else:
                    # Truncate extra results
                    extraction_data["results"] = extraction_data["results"][
                        : len(batch)
                    ]

            # Add to result dictionary
            for text, result in zip(batch, extraction_data["results"]):
                ticker = result["ticker"]
                # Ensure the ticker has $ prefix
                if (
                    ticker
                    and ticker not in ["UNKNOWN", "$UNKNOWN"]
                    and not ticker.startswith("$")
                ):
                    ticker = f"${ticker}"
                result_dict[text] = ticker

            # Add a small delay to avoid rate limiting
            time.sleep(1)

        except Exception as e:
            print(f"Error calling GPT API for batch: {e}")
            # If batch fails, mark all as UNKNOWN
            for text in batch:
                result_dict[text] = "$UNKNOWN"

    return result_dict


def process_csv_file(input_file, batch_size=20, mock_mode=False):
    """
    Process a CSV file containing financial text data.

    Args:
        input_file: Path to the input CSV file
        batch_size: Number of texts to process in a single GPT batch
        mock_mode: If True, don't make actual API calls (for testing)
    """
    # Create output directory if it doesn't exist
    os.makedirs("gpt_preprocessor", exist_ok=True)

    # Create output filename
    input_filename = os.path.basename(input_file)
    output_file = os.path.join("gpt_preprocessor", f"gpt_{input_filename}")

    print(f"Processing {input_file}...")

    # Read the CSV file
    df = pd.read_csv(input_file)

    # Make sure we have the required columns
    if "text" not in df.columns or "label" not in df.columns:
        print(
            f"Error: Input file must have 'text' and 'label' columns. Found: {df.columns.tolist()}"
        )
        return

    # Filter out entries with label 2 (neutral)
    print(f"Total entries before filtering label 2: {len(df)}")
    df = df[df["label"] != 2]
    print(f"Total entries after filtering label 2: {len(df)}")

    # Identify texts that need ticker extraction
    texts_for_extraction = []
    indices_for_extraction = []

    for idx, row in df.iterrows():
        text = row["text"]
        first_word = text.split()[0] if text else ""
        if not (first_word and "$" in first_word):
            texts_for_extraction.append(text)
            indices_for_extraction.append(idx)

    print(f"Found {len(texts_for_extraction)} texts that need ticker extraction")

    # Get tickers in batches or use mock mode
    ticker_dict = {}
    if texts_for_extraction and not mock_mode:
        print(f"Extracting tickers in batches of {batch_size}...")
        ticker_dict = get_tickers_from_gpt_batch(texts_for_extraction, batch_size)
    elif mock_mode:
        print("Running in mock mode - all unknown tickers will be set to $UNKNOWN")

    # Process all rows and create new data
    new_data = []

    for idx, row in df.iterrows():
        text = row["text"]
        label = row["label"]

        # Convert label to sentiment text
        if label == 1:
            sentiment = "bullish"
        elif label == 0:
            sentiment = "bearish"
        else:
            sentiment = str(
                label
            )  # This shouldn't happen since we filtered neutral (2), but just in case

        # Get ticker - either from first word or from extraction results
        first_word = text.split()[0] if text else ""
        if first_word and "$" in first_word:
            ticker = first_word
        else:
            ticker = ticker_dict.get(text, "$UNKNOWN")

        # Add to new data
        new_data.append({"ticker": ticker, "senti_label": sentiment, "text": text})

    # Create new DataFrame and save to CSV
    new_df = pd.DataFrame(new_data)
    new_df.to_csv(output_file, index=False)
    print(f"Processed file saved to {output_file}")
    print(
        f"Output contains {len(new_df)} entries with columns: {new_df.columns.tolist()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process CSV files for ticker and sentiment analysis"
    )
    parser.add_argument("input_file", help="Input CSV file to process")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Batch size for GPT API calls (default: 20)",
    )
    parser.add_argument(
        "--mock", action="store_true", help="Run in mock mode without making API calls"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist")
        exit(1)

    # Check for API key if not in mock mode
    if not api_key and not args.mock:
        print("Warning: No OpenAI API key provided. Either:")
        print("  1. Set the OPENAI_API_KEY environment variable")
        print("  2. Use --mock to run without API calls")
        print("Running in mock mode...")
        args.mock = True

    process_csv_file(args.input_file, args.batch_size, args.mock)
