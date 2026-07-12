import gradio as gr
from fastapi import FastAPI

from stroke_risk.serving.inference import predict as run_prediction


def build_demo(app: FastAPI) -> gr.Blocks:
    """Build the Gradio UI that scores patients via the given FastAPI app's model."""
    def call_predict(
        patient_id: int,
        age: float,
        hypertension: str,
        heart_disease: str,
        avg_glucose_level: float,
        bmi: float | None,
        gender: str,
        ever_married: str,
        work_type: str,
        residence_type: str,
        smoking_status: str,
    ) -> str:
        """Score patient inputs from the UI and format a result string."""
        payload = {
            "id": int(patient_id),
            "age": age,
            "hypertension": int(hypertension),
            "heart_disease": int(heart_disease),
            "avg_glucose_level": avg_glucose_level,
            "bmi": bmi,
            "gender": gender,
            "ever_married": ever_married,
            "work_type": work_type,
            "residence_type": residence_type,
            "smoking_status": smoking_status,
        }

        try:
            result = run_prediction(payload, app.state.model)
        except Exception as exc:
            return f"Prediction failed: {exc}"

        return f"Patient {result['patient_id']}: {result['prediction']}"

    with gr.Blocks(title="Stroke Risk Predictor") as demo:
        gr.Markdown("# Stroke Risk Predictor")

        with gr.Row():
            with gr.Column():
                patient_id = gr.Number(label="Patient ID", value=1, precision=0, minimum=1)
                age = gr.Slider(label="Age", minimum=0.1, maximum=119.9, value=45, step=0.1)
                avg_glucose_level = gr.Number(label="Avg Glucose Level", value=100.0, minimum=50.01)
                bmi = gr.Number(label="BMI (optional)", value=None)
                gender = gr.Radio(["male", "female", "other"], label="Gender", value="male")
                ever_married = gr.Radio(["yes", "no"], label="Ever Married", value="yes")

            with gr.Column():
                hypertension = gr.Radio(["0", "1"], label="Hypertension", value="0")
                heart_disease = gr.Radio(["0", "1"], label="Heart Disease", value="0")
                work_type = gr.Dropdown(["children","govt_job","never_worked","private","self_employed",],
                    label="Work Type",
                    value="private",
                )
                residence_type = gr.Radio(["urban", "rural"], label="Residence Type", value="urban")
                smoking_status = gr.Dropdown(["formerly_smoked", "never_smoked", "smokes", "unknown"],
                    label="Smoking Status",
                    value="never_smoked",
                )

        predict_btn = gr.Button("Predict", variant="primary")
        output = gr.Textbox(label="Result")

        predict_btn.click(
            fn=call_predict,
            inputs=[
                patient_id,
                age,
                hypertension,
                heart_disease,
                avg_glucose_level,
                bmi,
                gender,
                ever_married,
                work_type,
                residence_type,
                smoking_status,
            ],
            outputs=output,
        )

    return demo