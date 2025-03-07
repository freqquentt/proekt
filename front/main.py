import gradio as gr
import pandas as pd
import requests
import io
import os

BACKEND_URL = 'http://localhost:8000'

MODELS_DIR = '../back/saved_models'


def update_column_choices(file):
    if file is not None:
        df = pd.read_csv(file.name, nrows=5)
        return gr.update(choices=df.columns.tolist(), value=df.columns[0] if len(df.columns) > 0 else None)
    return gr.update(choices=[], value=None)


def list_trained_models():
    models_dir = MODELS_DIR
    if os.path.exists(models_dir):
        return [f for f in os.listdir(models_dir)]
    return []


def refresh_models():
    return gr.update(choices=list_trained_models())


def train_model(model_name, file, target_column):
    if file is None or target_column is None or model_name is None:
        return "Please fill in all fields."

    df = pd.read_csv(file.name)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    response = requests.post(
        f"{BACKEND_URL}/train/",
        json={
            "model_name": model_name,
            "target_column": target_column,
            "csv_data": csv_buffer.getvalue()
        }
    )

    if response.status_code == 200:
        return response.json().get("message", "Training started.")
    else:
        return f"Error: {response.text}"


def test_model(model_file, file, target_column):
    if file is None or model_file is None or target_column is None:
        return None, "Please fill in all fields."

    df = pd.read_csv(file.name)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    response = requests.post(
        f"{BACKEND_URL}/test/",
        json={
            "model_name": model_file,
            "target_column": target_column,
            "csv_data": csv_buffer.getvalue()
        }
    )

    if response.status_code == 200:
        output_filename = "results.zip"
        with open(output_filename, "wb") as f:
            f.write(response.content)
        return output_filename, "Success"
    else:
        return None, f"Error: {response.text}"


with gr.Blocks() as demo:
    gr.Markdown("## Classification Demo App (FastAPI Backend)")

    with gr.Tabs():
        with gr.TabItem("Train"):
            gr.Markdown("### Train the Model")
            train_model_dropdown = gr.Dropdown(
                choices=["TabNet", "TabTransformer", "FT-Transformer", "XGBClassifier", "RandomForestClassifier"],
                label="Choose a Model",
                value="TabNet"
            )
            train_file_input = gr.File(label="Upload CSV File", file_types=[".csv"])
            train_column_dropdown = gr.Dropdown(
                choices=[],
                label="Select Target Column"
            )
            train_button = gr.Button("Start Training")
            train_output = gr.Textbox(label="Output")

            train_file_input.change(fn=update_column_choices, inputs=train_file_input, outputs=train_column_dropdown)
            train_button.click(fn=train_model, inputs=[train_model_dropdown, train_file_input, train_column_dropdown],
                               outputs=train_output)

        with gr.TabItem("Test"):
            gr.Markdown("### Test the Model")
            test_model_dropdown = gr.Dropdown(
                choices=list_trained_models(),
                label="Choose a Trained Model"
            )

            demo.load(refresh_models, inputs=[], outputs=test_model_dropdown)

            test_file_input = gr.File(label="Upload CSV File", file_types=[".csv"])
            test_column_dropdown = gr.Dropdown(
                choices=[],
                label="Select Target Column"
            )

            test_button = gr.Button("Test Model")
            test_output_file = gr.File(label="Download Predictions")
            test_output_text = gr.Textbox(label="Status / Error Message")

            test_file_input.change(fn=update_column_choices, inputs=test_file_input, outputs=test_column_dropdown)
            test_button.click(
                fn=test_model,
                inputs=[test_model_dropdown, test_file_input, test_column_dropdown],
                outputs=[test_output_file, test_output_text]
            )

demo.launch()