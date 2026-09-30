## Project Overview
The project aims to see how S.M.E.s in Africa can leverage consistent language patterns observed in positive or neutral customer reviews to enhance their customer experience and align their offerings with customer expectations. By analyzing data from customer reviews across various sectors, the insights can be used to help S.M.E.s and women led businesses identify opportunities for improvement, strengthen their brand perception, and ultimately drive customer satisfaction and loyalty in the African market. 

Dataset Accessible here [Dataset](https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset)

### Project Setup Instructions

- Create a Virtual Environment using either Venv/ Conda
- Activate the Environment and  run ```pip install -r requirements.txt```

1. **PyTorch Library Installation:**
   - If PyTorch Installation fails via `requirements.txt`, please visit the [Official PyTorch Page](https://pytorch.org/) and follow the installation instructions. It is recommended to install PyTorch without CUDA support.

2. **Setting Up Secrets for Local Run:**
   - Create a `secrets.toml` file in the `./streamlit` directory within your project root folder. 
   
   - The structure of the `secrets.toml` file should be as follows:

     ```toml
     [openai]
     api_key = "Your API KEY"
     ```

     Replace `"Your API KEY"` with your actual OpenAI API Key. This file will be used to store sensitive information securely within your project.


3. **System Error Handling:**

   - The system verifies if the uploaded file conforms to the required file structure, notifying businesses about any missing columns.
   - Automatically detecting file encodings, the system ensures that files are read based on these encodings.
   - Date columns are converted to the required date format by the system.
   - To prevent key errors caused by columns specified in different formats, the system standardizes all columns to lowercase.

     ```
     streamlit run sentiment.py
     ```
     
### Project Testing

  - Download Flipkart Customer Product Reviews Data found in the `Dataset` Folder.
  - Upload the Dataset to the sentiment Page to generate the Sentiment Scores and Overall Sentiments for each of the Customer Reviews.
  - Download the Updated Reviews( e.g `Flipkart_updated_reviews_2019_11_01.csv`) and store them Locally.
  - Upload the Updated Reviews to Business Recommendations Page to get actionable recommendations from `gpt3.5-turbo` model and ask follow-up questions.
  - Upload the Updated Reviews to Data Insights to generate Insights such as Product Vs Sentiment, Distribution of sentiments across the year etc.

     
Other Contributors 
John Thuo,
Clare Kanja
