# Disaster Response Pipeline Project

### Project Highlight:

Within this project, data was treated through an ETL pipeline and then introduced into a machine learning model in order to predict categories of messages received from different sources when a disaster occurs. Since our model learns from previous data already classified, it is of a supervised learning type. Once all these data engineering processes were applied, an API (Application Programming Interface) was created in order for the users to have a visual representation of the model. 

### Project Structure:

The project is divided in three different parts: 

**ETL Pipeline -** This part is where the data processing occurs. Data was loaded from two different datasets, merged, cleaned and stored in a SQLite database.

**ML Pipeline -** Within the machine learning pipeline, which includes a text processing part, the model was created and tested with different configurations through GridSearchCV in order to find the most optimal one. Then, it was trained and tested.  

**Flask Web App -** This is the interface of the app which includes the ETL and ML pipeline parts in order to make the app work. 

### Files:

* data/process_data.py: The ETL pipeline where data was extracted, loaded and transformed.

* models/train_classifier.py: The Machine Learning pipeline where the model was built, optimized, trained and tested. It is finally saved as a pickle file. app/templates/*.html: HTML templates for the web app. 

* run.py: Start the Python server for the web app.

### Author:
Jessica Garre
