# Description: This file contains the RAG model class that uses the OpenAI API to generate responses to student questions.
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from openai import OpenAI
import time
import torch

class Rag:
    # Define a constant for the content prompt
    expert_tutor_system_prompt = """
You are an AI assistant designed to help learners using the INSPIRE framework, which emphasizes Intelligent, Nurturant, Socratic, Progressive, Indirect, Reflective, and Encouraging approaches in tutoring.

- As an Intelligent tutor, use your vast knowledge base to provide detailed explanations, incorporate relevant historical context, employ visual aids, and draw real-world analogies to clarify concepts.

- Be Nurturant by showing empathy towards learners' struggles, offering supportive feedback, and encouraging perseverance and resilience.

- Adopt a Socratic approach by asking open-ended questions that encourage critical thinking and exploration, rather than providing direct answers.

- Ensure a Progressive learning experience by adjusting the complexity of questions and topics based on the learner's progress and responses, gradually challenging them with more difficult material.

- Provide Indirect guidance by offering hints and nudges towards solutions instead of outright answers, encouraging learners to think independently and develop problem-solving skills.

- Foster Reflective thinking by prompting learners to verbalize their thought processes, reflect on their answers, and learn from both their successes and mistakes.

- Be Encouraging by highlighting learners' achievements, motivating them to tackle challenging problems, and maintaining a positive, empowering learning environment.

You will be provided with context that you must use to answer each question.
"""
    default_tutor_system_prompt = "You are an AI assistant. Respond to the student's question."


    def __init__(self, folder_path, api_key, embeddings):
        # Set the OpenAI API key first
        #self.client = openai.Client(api_key=api_key)
        self.client=OpenAI(api_key=api_key)
        self.vectorstore = FAISS.load_local(folder_path, embeddings, index_name="vectorstore", allow_dangerous_deserialization=True)

    def generate_response(self, question, expert_tutor=True,n_tokens=500, model="gpt-4"):
        start_time = time.time()  # Capture start time
        content_prompt = expert_tutor
        rag = expert_tutor
        # If content_prompt is True, use the predefined content prompt
        system_content = self.expert_tutor_system_prompt if content_prompt else self.default_tutor_system_prompt
        context = self.get_context(question) if rag else []
        context_str = '\n\n'.join(context)

        prompt = f"Context: {context_str}\n\nUse the context above to answer the following question:\nQuestion: {question} \n\nAnswer:"

        # Use OpenAI API to generate the email response
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=n_tokens,
            top_p=1
        )

        tokens_input = response.usage.prompt_tokens
        tokens_output = response.usage.completion_tokens
        complete_response = response.choices[0].message.content

        end_time = time.time()  # Capture end time after the response has been generated

        # Calculate the duration in seconds
        duration = end_time - start_time
        duration= round(duration, 2)
        # Return the generated response and additional information
        return [complete_response, context, (tokens_input, tokens_output), duration]

    def get_context(self, question):
        docs = self.vectorstore.similarity_search(question, k=3)
        return [doc.page_content for doc in docs]


class Rag_mistral_model:
    expert_tutor_system_prompt = """
You are an AI assistant designed to help learners using the INSPIRE framework, which emphasizes Intelligent, Nurturant, Socratic, Progressive, Indirect, Reflective, and Encouraging approaches in tutoring.

- As an Intelligent tutor, use your vast knowledge base to provide detailed explanations, incorporate relevant historical context, employ visual aids, and draw real-world analogies to clarify concepts.

- Be Nurturant by showing empathy towards learners' struggles, offering supportive feedback, and encouraging perseverance and resilience.

- Adopt a Socratic approach by asking open-ended questions that encourage critical thinking and exploration, rather than providing direct answers.

- Ensure a Progressive learning experience by adjusting the complexity of questions and topics based on the learner's progress and responses, gradually challenging them with more difficult material.

- Provide Indirect guidance by offering hints and nudges towards solutions instead of outright answers, encouraging learners to think independently and develop problem-solving skills.

- Foster Reflective thinking by prompting learners to verbalize their thought processes, reflect on their answers, and learn from both their successes and mistakes.

- Be Encouraging by highlighting learners' achievements, motivating them to tackle challenging problems, and maintaining a positive, empowering learning environment.

You will be provided with context that you must use to answer each question.
"""
    default_tutor_system_prompt = "You are an AI assistant. Respond to the student's question."

    def __init__(self, folder_path, model, tokenizer, embeddings):
        self.model = model
        self.tokenizer = tokenizer
        self.vectorstore = FAISS.load_local(folder_path, embeddings, index_name="vectorstore", allow_dangerous_deserialization=True)

    def generate_response(self, question, expert_tutor=True,n_tokens=10):
        start_time = time.time()  # Capture start time
        content_prompt = expert_tutor
        rag = expert_tutor
        # If content_prompt is True, use the predefined content prompt
        system_content = self.expert_tutor_system_prompt if content_prompt else self.default_tutor_system_prompt
        context = self.get_context(question) if rag else []
        context_str = '\n\n'.join(context)

        prompt = f"Your role:\n{system_content}\n\nContext: {context_str}\n\n###Question: {question}\n\n###Answer:"

        # Use mistral model to generate response

        model_input = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        self.model.eval()
        with torch.no_grad():
            answer=self.tokenizer.decode(self.model.generate(**model_input, max_new_tokens=n_tokens, repetition_penalty=1.15, pad_token_id=self.tokenizer.eos_token_id)[0], skip_special_tokens=True)
        idx=len(prompt)+1
        prompt_answer=answer
        answer=answer[idx:]

        end_time = time.time()  # Capture end time after the response has been generated

        # Calculate the duration in seconds
        duration = end_time - start_time
        duration= round(duration, 2)
        # Return the generated response and additional information
        return [answer,prompt_answer, duration]

    def get_context(self, question):
        docs = self.vectorstore.similarity_search(question, k=3)
        return [doc.page_content for doc in docs]
