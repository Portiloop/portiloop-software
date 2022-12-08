import gradio as gr

from portiloop.src.demo.offline import run_offline

        
def on_upload_file(file):
    # Check if file extension is .xdf
    if file.name.split(".")[-1] != "xdf":
        raise gr.Error("Please upload a .xdf file.")
    else:
        yield f"File {file.name} successfully uploaded!"


with gr.Blocks(title="Portiloop") as demo:
    gr.Markdown("# Portiloop Demo")
    gr.Markdown("This Demo takes as input an XDF file coming from the Portiloop EEG device and allows you to convert it to CSV and perform the following actions:: \n * Filter the data offline \n * Perform offline spindle detection using Wamsley or Lacourse. \n * Simulate the Portiloop online filtering and spindle detection with different parameters.")
    gr.Markdown("Upload your CSV file and click **Run Inference** to start the processing...")

    with gr.Row():
        xdf_file = gr.UploadButton(label="XDF File", type="file")

        # Offline Filtering (Boolean)
        offline_filtering = gr.Checkbox(label="Offline Filtering (On/Off)", value=True)
        # Online Filtering (Boolean)
        online_filtering = gr.Checkbox(label="Online Filtering (On/Off)", value=True)
        # Lacourse's Method (Boolean)
        lacourse = gr.Checkbox(label="Lacourse Detection (On/Off)", value=True)
        # Wamsley's Method (Boolean)
        wamsley = gr.Checkbox(label="Wamsley Detection (On/Off)", value=True)
        # Online Detection (Boolean)
        online_detection = gr.Checkbox(label="Online Detection (On/Off)", value=True)

        # Threshold value
        threshold = gr.Slider(0, 1, value=0.82, step=0.01, label="Threshold", interactive=True)
        # Detection Channel
        detect_channel = gr.Dropdown(choices=["1", "2", "3", "4", "5", "6", "7", "8"], value="2", label="Detection Channel in XDF recording", interactive=True) 
        # Frequency
        freq = gr.Dropdown(choices=["100", "200", "250", "256", "500", "512", "1000", "1024"], value="250", label="Sampling Frequency (Hz)", interactive=True)

    # Output elements
    update_text = gr.Textbox(value="Upload an XDF File and click 'Run Inference'...", label="Status", interactive=False)
    output_plot = gr.Plot()
    output_array = gr.File(label="Output CSV File")
    xdf_file.upload(fn=on_upload_file, inputs=[xdf_file], outputs=[update_text])

    def clear():
        output_plot.clear()
        output_array.clear()
        update_text.clear()
        xdf_file.clear()

    # Row containing all buttons:
    with gr.Row():
        # Run inference button
        run_inference = gr.Button(
            value="Run Inference")
        # Reset button
        reset = gr.Button(value="Reset", variant="secondary", on_click=clear,)
    run_inference.click(
        fn=run_offline, 
        inputs=[
            xdf_file, 
            offline_filtering, 
            online_filtering, 
            online_detection, 
            lacourse, 
            wamsley, 
            threshold, 
            detect_channel,
            freq], 
        outputs=[output_plot, output_array, update_text])

demo.queue()
demo.launch(share=False)
