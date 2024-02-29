import requests
import pandas as pd

# Replace with your API endpoint and credentials if needed
api_endpoint = "https://ci03.simtlix.com/api/stations"
api_username = "monitor"
api_password = "SaltaLaLinda01"

# Define a function to call the API and return the data as a DataFrame
def get_data():
    # Set up authentication if needed
    if api_username and api_password:
        auth = (api_username, api_password)
    else:
        auth = None
    
    # Call the API and get the response
    response = requests.get(api_endpoint, auth=auth)
    
    # Check if the response was successful
    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code}")
    
    # Convert the JSON response into a DataFrame
    data = pd.DataFrame(response.json())
    
    return data

get_data().to_csv('./info.csv')