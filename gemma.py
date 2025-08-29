from transformers import pipeline

# Load instruction-tuned Gemma if possible
pipe = pipeline("text-generation", model="google/gemma-2b-it")

print("Chatbot ready! Type 'exit' to quit.\n")

while True:
    prompt = input("You: ")

    if prompt.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Generate response
    result = pipe(
        prompt,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    # Clean result
    response = result[0]["generated_text"]
    if response.startswith(prompt):
        response = response[len(prompt):].strip()

    print(f"AI: {response}\n")
