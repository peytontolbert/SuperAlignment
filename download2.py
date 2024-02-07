import requests
import json

# Define the API endpoint URL
api_url = "https://huggingface.co/api/datasets"  # Note: This is a placeholder. You should replace it with the correct API endpoint.

# If needed, define headers for authentication
headers = {
    # 'Authorization': 'Bearer YOUR_ACCESS_TOKEN',
}


def download_models_list(url, headers=None):
    """Downloads a list of models from the specified API endpoint and saves it to a JSON file."""
    try:
        # Send a GET request to the API
        response = requests.get(url, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse JSON response
            models_list = response.json()

            # Save the data to a JSON file
            with open("datasets.json", "w") as json_file:
                json.dump(models_list, json_file, indent=4)

            print("Models list downloaded successfully.")
        else:
            print(
                f"Failed to download models list. HTTP Status Code: {response.status_code}"
            )
    except Exception as e:
        print(f"An error occurred: {e}")


# Call the function
download_models_list(api_url, headers)
