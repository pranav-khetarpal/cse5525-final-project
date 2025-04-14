import pandas as pd
import os


def fix_csv_order(input_file, output_file=None):
    if output_file is None:
        directory = os.path.dirname(input_file)
        filename = os.path.basename(input_file)
        output_file = os.path.join(directory, "fixed_" + filename)

    df = pd.read_csv(input_file, header=None, names=["ticker", "senti_label", "tweet"])


    df = df[["ticker", "tweet", "senti_label"]]

    df.to_csv(output_file, index=False, header=False)
    print(f"Processed {input_file} -> {output_file}")


    print("\nFirst few rows of the fixed file:")
    print(df.head().to_string())


def main():
    val_file = "tweet/adjusting_val_data.csv"
    if os.path.exists(val_file):
        fix_csv_order(val_file)
    else:
        print(f"Warning: {val_file} not found")

    train_file = "tweet/adjusting_train_data.csv"
    if os.path.exists(train_file):
        fix_csv_order(train_file)
    else:
        print(f"Warning: {train_file} not found")




if __name__ == "__main__":
    main()
