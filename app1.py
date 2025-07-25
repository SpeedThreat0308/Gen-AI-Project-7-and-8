import requests
import json
import gradio as gr


url="http://localhost:11434/api/generate"

headers={
    'Content-Type': 'application/json'
}

history=[]

def generate_response(prompt):
    history.append(prompt)
    final_prompt="\n".join(history)
    data={
        "model": "brocode",
        "prompt": final_prompt,
        "stream":False,
    }

    response=requests.post(url,headers=headers,data=json.dumps(data))

    if response.status_code==200:
        response=response.text
        data=json.loads(response)
        actual_response=data['response']
        return actual_response
    else:
        print("Error:",response.text)

interface=gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=5,label="Enter your prompt", placeholder="Type here..."),
    outputs=gr.Textbox(label="Response"),
    title="brocode Chatbot",
    description="A simple chatbot powered by brocode.",
    theme="default",
    allow_flagging="never"
)

interface.launch()