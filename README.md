## Project Overview <a name="Project-Overview"></a>
Disaster Response Pipeline App project is part of the Udacity Data Science Nano Degree. This project builds a Natural Language Processing (NLP) for calssification of disaster messages into various categories. There is also a provision for emergency classification of a disaster message, wherein one can input a message and direct it to the appropriate organization/team upon its classification. 



## Instructions <a name="How-To-Run-This-Project"></a>
### 1. Download the files or clone this repository
  ```
  git clone https://github.com/Gagankaurgill/Disaster-Response-App.git
  ```
### 2. Execute the scripts
a. Open a terminal <br>
b. Direct to the project's root directory <br>
c. Run the following commands: <br>
- To run ETL pipeline that cleans data and stores in database
  ```
  python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
  ```
- To run ML pipeline that trains classifier and saves
  ```
  python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
  ```

d. Go to the app's directory and run the command
```sh
cd app
python run.py
```
e. The web app is hosted at: http://0.0.0.0:3001/ or http://localhost:3001/ 

f. Input any message in the input box and click on the Classify Message button to see the categories that the message may belong to.