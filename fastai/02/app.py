import gradio as gr
from fastai.vision.all import *

learn_inf = load_learner('export.pkl')

def classify(image):
    label, _, probs = learn_inf.predict(image)
    return dict(zip(learn_inf.dls.vocab, map(float, probs)))

demo = gr.Interface(
    fn=classify,
    inputs=["image"],
    outputs=["label"],
    examples=['teddy.jpg', 'black.jpg', 'grizzly.jpg'],
    allow_flagging='never'
)

demo.launch()