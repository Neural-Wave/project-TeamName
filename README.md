# Neural Wave Hackaton
- **Team Name**: \<teamName\>
- **Project Name**: Swisscom project \<teamName\>
- **Hackathon**: NeuralWave 
- **Date**: October 25-27, 2024
- **Team Members**:
    - Tornike Onoprishvili
    - Riccardo Sacco
    - Carla Lopez Zurita
    - Roberts Kalvitis
    - Michele Smaldone

### Project Description
While Swisscom has a fully functional chatbot implemented on their website, we believe there is still room for improvement. The original chatbot sometimes presents issues with consistency when the user switches languages, which is a common occurrence given Switzerland's multilingual setting. Other considerations include providing helpful references within the Swisscom website, as well as offering truly useful responses to users' questions. Our goal is to create an effective chatbot that assists users with a variety of issues, beating performance of the currently implementing chatbot. The chatbot should:
- Detect language and apply it to conversation correctly;
- Filter out irrelevant user questions;
- Give users useful responses that are based on Swisscom’s publicly available data on their website.    


### Folder structure
TODO this needs changing at the end, when file cleanup is done
```text
    ├── LICENSE.md
    ├── README.md
    ├── assets
    │   ├── chroma
    │   │   └── swisscom_openai
    │   ├── evaldata
    │   │   └── all.json
    │   └── prompts.md
    ├── folder_structure.txt
    ├── requirements.txt
    └── src
        ├── __init__.py
        ├── ai_judge.py
        ├── chat.py
        ├── evaluator.py
        ├── swisscom_rag_chat.py
        ├── make_predictions.py
        ├── swisscom_rag.py
        └── utils.py
```

### Installing Dependencies
To install the necessary packages listed in `requirements.txt`, run the following command:

```bash
pip install -r requirements.txt
```

### Create a `.env` File

1. In the root directory of your project, create a new file named `.env`.
   
2. Open the `.env` file and add your API key as follows:

```plaintext
OPENAI_API_KEY=your_api_key_here
```
## Running json evaluation






## Running project from scrach



### Download the dataset

1. Download the dataset from https://swisscom-my.sharepoint.com/:u:/p/stefano_taillefert/EegWIyF8835PuUXsyuzmGGsBcxu7gFVcJVhyOpLVhZ_g4A?e=nsivZN
2. place it in the `root directory` and run command in terminal.
```bash
unzip dataset.zip
```


### Ingest the document base:
```bash
python src/ingest_documents.py
```

### Run chat:

```bash
python src/swisscom_rag_chat.py
```



