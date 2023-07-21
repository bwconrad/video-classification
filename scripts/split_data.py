import random
from argparse import ArgumentParser


def split_data_file(
    input_file,
    train_file: str = "train.txt",
    val_file: str = "val.txt",
    val_split: float = 0.3,
    seed: int | None = None,
):
    # Set the random seed for reproducibility
    random.seed(seed)

    # Read the input file
    with open(input_file, "r") as f:
        lines = f.readlines()

    # Shuffle the lines randomly
    random.shuffle(lines)

    # Calculate the number of lines for validation set
    num_val = int(len(lines) * val_split)

    # Split the lines into train and validation sets
    train_lines = lines[num_val:]
    val_lines = lines[:num_val]

    # Write the train lines to train file
    with open(train_file, "w") as f:
        f.writelines(train_lines)

    # Write the validation lines to validation file
    with open(val_file, "w") as f:
        f.writelines(val_lines)


if __name__ == "__main__":
    # Create an argument parser
    parser = ArgumentParser(
        description="Split a data file into train and validation files"
    )

    # Add the arguments
    parser.add_argument(
        "--input-file", "-i", type=str, help="Path to the input data file"
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="train.txt",
        help="Output path to the train file",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="val.txt",
        help="Output path to the validation file",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.3, help="Validation split ratio"
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducibility"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the split_data_file function
    split_data_file(
        args.input_file, args.train_file, args.val_file, args.val_split, args.seed
    )
