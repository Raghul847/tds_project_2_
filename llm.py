from openai import OpenAI

# init the client but point it to TGI
client = OpenAI(api_key="AIzaSyB3MeAH1ewDuW_AW7ood6GU_Gcn1fhl020", base_url="http://127.0.0.1:8080/v1")
chat_response = client.chat.completions.create(
    model="-",
    messages=[
      {"role": "user", "content": "What is deep learning?"}
    ]
)

print(chat_response)
