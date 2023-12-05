# ReTeach | AI-Assisted Coding
> Creating tools to help your company accelerate the process of labeling for training set creation

**Due Thursday, December 14th at 11:59pm**

## Motivation
Recall that you're the co-founder of an education startup, and are responsible for demonstrating a proof of concept for a new automated teacher feedback product called ReTeach. ReTeach is an automated platform that listens to teachers in the classroom, and provides feedback.

You may recall that you've already seen this data, which was already **coded**. In a fun twist of jargon, "coding" to your subject matter colleagues is "labelling" in data science jargon. Regardless, coding/labelling is expensive, slow, tedious, and time-consuming. Your subject matter colleagues request an in-house solution to rapidly label this sheer volume of data. They request the following functionality in a user interface:
* **Something that helps them label FAST (no excessive clicking)**
* **A model that can provide some guesses about each statement and add labels for easy review**
* **The ability to ask questions of the data and get an answer**

You decide to do them one better and also provide:
* **An interactive labeling platform that "learns" while the user is labeling and iteratively creates better and better predictions.**

Your team decides to deliver an in-house solution for this, which needs to be maintainable and extensible.

## Data
Your data is a pre-processed set of transcripts into csv files. Each csv file is a recording of a classroom session. The csv currently has two columns - one for the transcribed text, and another for the labels. However, when starting, some csvs may only have one column as there are no labels yet provided. Each row is a statement made by the teacher. The labels are:
* **OTR**: An opportunity to respond (e.g., 'What color is this candy?')
* **PRS**: Praise (e.g., "Great job, Morgan!")
* **REP**: Reprimand (e.g., "You need to keep quiet while someone else is reading.")
* **NEU**: None of the above


# Requirements and Grading (Out of 200 points)
Your project should be submitted as a GitHub repository with well-organized Jupyter notebooks, an `app.py` file (or file representing the user interface), and a `requirements.txt` file (or file with additional package installation requirements). You should also include a `.devcontainer` folder and `ReadMe.md` files. The `.devcontainer` directory should allow the graders to create containers with your specified environment without needing to install any packages on their own. The Readme should provide an overview of the files of the repository and any additional information needed to run the app if you choose not to use gradio or streamlit.

## Jupyter Notebooks and Software Development Practices (150 points)
### Software Development Principles (40 points)
This project very much blends data science requirements with software engineering requirements, and for this reason, code reuse, modularity, loose coupling, and other software engineering principles are pivotal. The majority of this work will be completed in Jupyter notebooks, allowing us to separate true functionality from user interface design. In keeping with this, you will be graded on:

#### Modular design, code reuse, and repo quality (30 points)
For this project, you must use nbdev. Beyond the use of nbdev, your code will be assessed based on:
* Repo design (e.g., titles, scope of Jupyter notebooks, etc)
* Usage of nbdev to create reusable components for your team and a reusable library of components
* Design of functions, classes, modularity of design, ease of reuse, clarity, code documentation
* Utilization of software engineering principles and overall impressions

#### Good usage of GH flow, commit, and branching strategies (10 points)
Since this is a group project for data science, it is expected that best practices will be used when committing code to the repository and building the deliverables. There is an expectation of seeing different branches for different components (e.g., we would expect a different branch for a different notebook, different UI page, etc.), with pull requests, commentary/approval, and merging. It is expected that there should be a minimum of 1-2 commits on each branch to finish the task. It is additionally expected that branches will be merged into the main branch, and at any time, the main branch will represent a final version of the current state of the work.

Please note that egregious violations of good GH flow, commit, etc may result in greater penalties as this is foundational to programming for data science.

### AI-Assisted Labeling (20 points)
Although we have the capacity provided above to manually label every line once we have the transcripts, it would be really nice if we could use an LLM to take a first pass at labeling. Although you have SEVERAL options in terms of implementation, a transformer model from Huggingface, langchain, or openai should be utilized via API. You can use the OpenAI assistants API, chat completion API, zero-shot classification via Huggingface, or through langchain. In this Jupyter notebook, you need to develop and demonstrate the following functionality based on a CSV file using one of these APIs:
* The model should receive the data in some format (probably CSV) or however you choose to approach this (e.g., line by line from the pandas dataframes).
* The model should provide a label for each of the texts and a value representing the confidence in the answer in some format
* Using this returned data, you should create a new table (probably pandas dataframe) with all of the information and the new labels. Since you are generally able to obtain the model's certainty with these labels, use Pandas styler in order to set the 
background color of cells with high confidence answers green (or some color), low confidence answers red (or some different color), and standard white backround (or some other different color) for so-so confidence answers.
* Your notebook should provide the ability to write out a CSV with the AI-assisted labels with the same data as the original, but updated label values and confidence values.
* Provide an example of this labeled dataset saved in a directory of your repository and make sure that you save this programmatically.

### Ask the Dataset (40 points)
Even beyond the capacity to label every line, we may want to know some things about the data and not want to write code to make this happen. Instead, with the beauty of generative AI, we can literally just ask questions of the data once we have uploaded it. Again, although there are MANY ways that we can do this, one way that is tried and true is OpenAI's chat completion API, although could also use Huggingface or the OpenAI assistants API. In this Jupyter notebook, you should develop and demonstrate the following functionality based on a CSV file using one of these APIs:
* The model should receive the data in some format (probably CSV) or however you choose to approach this (e.g., string version of CSV), and you should also be able to input some string so a question can be asked of the model.
* The model should return an answer
* You should demonstrate (and keep this in your Jupyter notebook in an organized way) providing the data to the model as well as a question about the data and comment on how well this functionality works. In other words, provide some examples of the functionality and comment on the behavior.

### In-the-loop AI-Assisted Labeling (50 points)
Even further beyond the capacity for AI-Assisted labeling, we can also ask the model to help us as we try to label the dataset manually. In other words, we can "train" a model as we go along with labeling, and use this feedback for better predictions.
In this notebook, you will develop and demonstrate the following functionality based on a CSV file using one of these APIs:
* Create functionality that returns a variable number of lines from the transcript (probably consecutive) with their labels
* Create functionality that provides additional corrective information about anything mislabeled as well as returning the accuracy of the predictions
* Provide additional training to the model so that it utilizes this additional corrective information to make predictions on a new batch of the data
* Condition the batch sizes on the accuracy of the new predictions

## User Interface (50 points)
The user interface should be a deployable `app.py` file with a `requirements.txt` file or similar with additional required packages. The expectation is that the user interface will be created using streamlit or gradio, and these two platforms will be the only ones supported by your instructional team; however, feel free to try other packages like Flask if desired. Your best bet is probably gradio.

### UI Overview Page or Information (5 points)
You are delivering an internal software solution for your company. Because of this and the nature of project longevity, you should provide some information on the front page about what this project is, who developed it, what it's for, etc. This does not need to be overly long, and consider that if you write the README for the project well, it may be importable and directly reusable. Alternately, you can choose to have a brief explanation at the top of the page (or similar) and utilize a "Credits" or "Learn More" page where you can have more extensive information.

### AI-Assisted Labeling Page (5 points)
This page is relatively simple. Here, we will manifest the majority of the functionality that we created in our Jupyter notebook. For the user interface components and functionality, then, you need to make sure:
* A user is able to upload a CSV file of the current transcript table, whether labeled or unlabeled
* The page should display an explorable pandas chart table with all of the information and new labels with colors indicated.
* The user should be able to download the new CSV file with the suggested transcript labels. The filename should indicate a relation with the original filename that this is AI-assisted labeling.

### Ask the Dataset Page (5 points + 10 extra credit/optional points)
Again, most of your functionality is reusable and has already been implemented in your Jupyter notebooks so you need only add the user interface components and connect the components to the functionality. The main objectives of this page are:

* Allow the user to select one or more CSV files of interest that they want to know about. If the user selects more than one file, you should concatenate these together to form one big file (note that you can go back and implement this in your Jupyter notebook for reusability).
* The user should be able to input some question about the dataset, which should then be submitted to the system using your current functionality.
* All responses should be shown to the user. Although no conversation history is required, the user should be able to ask multiple independent questions.
* (10 points extra credit) Extra credit if conversation history is added (through any mechanism desired) so the model is able to utilize the entire chat history.

### AI-Assisted Labeling Page (30 points)
This page requires the most additional UI functionality to be added on top of the Jupyter notebooks. This page should allow:
* All the user to select one or more CSV files that they want to label.
* Display _n_ (or allow the user to select a starting number) of lines to label. 
* Allow the user to apply labels (i.e., select the correct label if no labels are provided or fix erroneous predictions) and ensure this is updated
* After the first round of example labeling, use this "training data" to "train" the model, and have it make predictions for a next batch of statements. Show these predictions to the user, again allowing them to fix erroneous predictions and repeat the "training" and prediction process. 
* At any time, the user should be able to download an updated version of the CSV file with all of the correct/verified labels added to the dataset. However, if the user decides to continue labeling, they should be able to do so without issue. If they decide to download later on in the training, they should also be able to do this.
* Throughout the process, a log of the accuracy of the model predictions should be kept, updated, and plotted per batch along with the batch size. This information should also be available for download for the user.

### Overall Impressions (5 points)
All work pages should contain some BRIEF, aesthetically pleasing explanation of how to use the page and its purpose.

Keep in mind that you're making this for other humans to interact with, so aesthetically pleasing and well laid-out user interfaces are necessary. Although this section is worth 5 points, egregious violations may incur greater point loss from the offending section.

# Logistical Details
## Teams
You will be working in teams of 4. Each member is fully responsible for all of the work done on the project. The expectation is that each member will be primarily responsible for one page of the user interface and one page of the Jupyter notebook analysis. The remaining tasks should be divided among members.

## Usage of Generative AI
You can use generative AI to the desired extent that you wish. Upload data to OpenAI Advanced Data Analytics, download Jupyter notebook analyses, use Bing Chat to help with new APIs. However, it is your responsibility to review and assess the output of the generative AI platforms that you use; the work is expected to reflect the beliefs, activity, and ideas of your team. You own the work.

## Additional Notes:
- Your code will need to be reproducible. We reserve the right to re-run all the notebooks and follow any reasonable instructions provided in the ReadMe to re-generate any code.
- Code which does not run may be subject to larger penalties than those described by the rubric depending on the magnitude of challenge faced when trying to evaluate the submissions.
- Literate programming!
- Write code (and design UIs) for humans
- Have fun!

Good luck!
