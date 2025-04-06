import gradio as gr
from transformers import pipeline
from PIL import Image

# Load the model once
pipe = pipeline('image-segmentation', model='briaai/RMBG-1.4', trust_remote_code=True)

def remove_background(image):
    result = pipe(image)
    return result[0]['segmentation'] if isinstance(result, list) else result

app = gr.Interface(
    fn=remove_background,
    inputs=gr.Image(type='pil'),
    outputs=gr.Image(type='pil'),
    title='Remoção de background de Imagens',
    description='Envie uma imagem e veja o background removido automaticamente.'
)

app.launch(server_name="0.0.0.0", server_port=8000)  # For Railway
