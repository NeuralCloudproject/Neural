import openai

class GPTIntegrator:
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def chat(self, neuron_state: str, user_input: str):
        prompt = f"Neuron State: {neuron_state}\nUser: {user_input}\nAI:"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].text.strip()
