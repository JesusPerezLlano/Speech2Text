from transformers import pipeline
import gradio as gr
import time

p = pipeline("automatic-speech-recognition",model="jonatasgrosman/wav2vec2-large-xlsr-53-spanish")
pc = pipeline("automatic-speech-recognition",model="softcatala/wav2vec2-large-xlsr-catala")
pe = pipeline("automatic-speech-recognition",model="jonatasgrosman/wav2vec2-large-xlsr-53-english")




def transcribe(language,audio, state=""):#language="Spanish",
    time.sleep(2)
    if language=="Spanish":
        state=""
        text = p(audio)["text"]
    if language=="Catalan":
        state=""
        text = pc(audio)["text"]
    if language=="English":
        state=""
        text = pe(audio)["text"]
    state += text + " "
    #text2="Esto es loq ue te he entendido"
    return state, state

demo=gr.Interface(
    fn=transcribe, 
    
    title="TEDCAS Offline Speech recognition",
    description="1)Select language 2)Click on 'record from microphone' and talk 3)Click on 'stop recording' 4)Click on submit 5)Before starting again, click on 'clear'",

    inputs=[
        gr.Dropdown(["Spanish","Catalan","English"]),
        #gr.Audio(source="microphone", type="filepath", streaming=True), 
        gr.inputs.Audio(source="microphone", type="filepath"), 
        "state"#,"language"
    ],
    outputs=[
        "textbox",
        "state"
    ],
    #live=True).launch()
)
demo.launch()
#gr.Interface(
#    fn=transcribe, 
#    inputs=[
#        gr.inputs.Audio(source="microphone", type="filepath"), 
#        "state"
#    ],
#    outputs=[
#        "textbox",
#        "state"
#    ],
#    live=True).launch()
