import json


def filter_models(input_file, output_file):
    """
    Reads models from input_file, filters based on 'dataset' or 'arxiv' presence,
    and writes the filtered models to output_file.
    """
    try:
        # Load the models from the input JSON file
        with open(input_file, "r") as file:
            models = json.load(file)

        # Filter models that contain 'dataset:' or 'arxiv:'
        filtered_models = [
            model
            for model in models
            if "dataset:" in json.dumps(model).lower()
            or "arxiv:" in json.dumps(model).lower()
        ]

        # Save the filtered models to the output JSON file
        with open(output_file, "w") as file:
            json.dump(filtered_models, file, indent=4)

        print(f"Filtered models are successfully written to {output_file}.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Specify the input and output file names
input_file = "models.json"
output_file = "models2.json"

# Call the function with the file paths
filter_models(input_file, output_file)
