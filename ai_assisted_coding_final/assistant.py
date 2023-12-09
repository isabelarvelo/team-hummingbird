# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_assistants.ipynb.

# %% auto 0
__all__ = ['OpenAIAssistantManager']

# %% ../nbs/00_assistants.ipynb 6
class OpenAIAssistantManager:
    def __init__(self, client):
        self.client = client
        self.current_assistant = None
        self.current_thread = None

    def add_file(self, file_path, purpose='assistants'):
        with open(file_path, "rb") as file_data:
            return self.client.files.create(file=file_data, purpose=purpose)

    def create_assistant(self, name="Classify Teacher Utterances",
                            description="A tool for classifying teacher utterances into categories like OTR, PRS, REP, NEU.",
                            instructions="""You are the co-founder of an ed-tech startup training an automated teacher feedback tool to classify utterances made. I am going to provide several sentences. 
                                            Please classify each sentence as one of the following: OTR (opportunity to respond), PRS (praise), REP (reprimand), or NEU (neutral)
        
                                            user: Can someone give me an example of a pronoun?
                                            assistant: OTR
                                            user: That's right, 'he' is a pronoun because it can take the place of a noun.
                                            assistant: PRS
                                            user: "You need to keep quiet while someone else is reading."
                                            assistant: REP
                                            user: A pronoun is a word that can take the place of a noun.
                                            assistant: NEU

                                            Only answer with the following labels: OTR, PRS, REP, NEU""",
                            model="gpt-4-1106-preview",
                            tools=None, file_id=None):
        assistant_kwargs = {
                "name": name,
                "description": description,
                "instructions": instructions,
                "model": model,
                "tools": tools if tools else []
            }

        if file_id:
            assistant_kwargs["file_ids"] = [file_id] if isinstance(file_id, str) else file_id

        self.current_assistant = self.client.beta.assistants.create(**assistant_kwargs)
        print(self.current_assistant.id)
        return self.current_assistant

    def create_custom_assistant(self, name="Custom Teacher Utterances Classifier",
                                description="A custom tool for classifying teacher utterances using gpt-3.5-turbo.",
                                instructions="""You are the co-founder of an ed-tech startup training an automated teacher feedback tool to classify utterances made. I am going to provide several sentences. 
                                            Please classify each sentence as one of the following: OTR (opportunity to respond), PRS (praise), REP (reprimand), or NEU (neutral)
                                            Only answer with the following labels: OTR, PRS, REP, NEU""",
                                model="gpt-3.5-turbo",
                                tools=[],
                                file_id=None):
        assistant_kwargs = {
            "name": name,
            "description": description,
            "instructions": instructions,
            "model": model,
            "tools": tools
        }

        if file_id:
            assistant_kwargs["file_ids"] = [file_id] if isinstance(file_id, str) else file_id
    
        self.current_assistant = self.client.beta.assistants.create(**assistant_kwargs)
        print(self.current_assistant.id)
        return self.current_assistant
    
    # add retreive assistant function

    def retrieve_assistant(self, assistant_id):
        self.current_assistant = self.client.beta.assistants.retrieve(assistant_id)
        return self.current_assistant

    def create_thread(self, user_message, file_id=None):
        message = {
            "role": "user",
            "content": user_message
        }

        if file_id:
            message["file_ids"] = [file_id]

        self.current_thread = self.client.beta.threads.create(messages=[message])
        return self.current_thread
    
    def delete_thread(self, user_message, file_id=None):
        self.client.beta.threads.delete(thread_id=self.current_thread.id)
        return self.current_thread

    
    def send_message(self, message_content):
        if self.current_thread is None:
            raise Exception("No active thread. Create a thread first.")
        return self.client.beta.threads.messages.create(
            thread_id=self.current_thread.id,
            role = "user",
            content = message_content
        )
    
    def list_messages(self):
        if self.current_thread is None:
            raise Exception("No active thread. Create a thread first.")
        thread_messages = self.client.beta.threads.messages.list(thread_id=self.current_thread.id)
        return thread_messages

    def retrieve_message(self, message_content):
        if self.current_thread is None:
            raise Exception("No active thread. Create a thread first.")
        return self.client.beta.threads.messages.retrieve('message_id', thread_id=self.current_thread.id)

    def get_response(self):
        if self.current_thread is None:
            raise Exception("No active thread. Create a thread first.")
        return self.client.beta.threads.messages.list(thread_id=self.current_thread.id, order="asc")


    def create_thread_and_run(self, user_input):
        # Create a new thread for each input
        self.current_thread = self.client.beta.threads.create()

        # Submit the message and wait for the run to complete
        run = self.submit_message(user_input)
        completed_run = self.wait_on_run(run)

        return self.current_thread, completed_run
    
    def wait_on_run(self, run):
        import time

        while run.status == "queued" or run.status == "in_progress":
            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.current_thread.id,
                run_id=run.id,
            )
            time.sleep(0.5)
        return run
    
    def submit_message(self, user_message):
        if self.current_thread is None or self.current_assistant is None:
            raise Exception("Assistant and Thread must be initialized before submitting a message.")
        
        self.client.beta.threads.messages.create(
            thread_id=self.current_thread.id, 
            role="user", 
            content=user_message
        )
        run = self.client.beta.threads.runs.create(
            thread_id=self.current_thread.id,
            assistant_id=self.current_assistant.id,
        )

        # Wait for the run to complete before returning
        return self.wait_on_run(run)

