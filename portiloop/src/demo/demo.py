import gradio as gr

from portiloop.src.demo.offline import run_offline

        
def on_upload_file(file):
    # Check if file extension is .xdf
    if file.name.split(".")[-1] != "xdf":
        raise gr.Error("Please upload a .xdf file.")
    else:
        return file.name


def main():
    with gr.Blocks(title="Portiloop") as demo:
        gr.Markdown("# Portiloop Demo")
        gr.Markdown("This Demo takes as input an XDF file coming from the Portiloop EEG device and allows you to convert it to CSV and perform the following actions:: \n * Filter the data offline \n * Perform offline spindle detection using Wamsley or Lacourse. \n * Simulate the Portiloop online filtering and spindle detection with different parameters.")
        gr.Markdown("Upload your XDF file and click **Run Inference** to start the processing...")

        with gr.Row():
            xdf_file_button = gr.UploadButton(label="Click to Upload", type="file", file_count="single")
            xdf_file_static = gr.File(label="XDF File", type='file', interactive=False)

            xdf_file_button.upload(on_upload_file, xdf_file_button, xdf_file_static)

            # Make a checkbox group for the options
            detect_filter = gr.CheckboxGroup(['Offline Filtering', 'Lacourse Detection', 'Wamsley Detection', 'Online Filtering', 'Online Detection'], type='index', label="Filtering/Detection options")

            # Threshold value
            threshold = gr.Slider(0, 1, value=0.82, step=0.01, label="Threshold", interactive=True)
            # Detection Channel
            detect_channel = gr.Dropdown(choices=["1", "2", "3", "4", "5", "6", "7", "8"], value="2", label="Detection Channel in XDF recording", interactive=True) 
            # Frequency
            freq = gr.Dropdown(choices=["100", "200", "250", "256", "500", "512", "1000", "1024"], value="250", label="Sampling Frequency (Hz)", interactive=True)

        output_array = gr.File(label="Output CSV File")

        run_inference = gr.Button(value="Run Inference")
        run_inference.click(
            fn=run_offline, 
            inputs=[
                xdf_file_static, 
                detect_filter,
                threshold, 
                detect_channel,
                freq], 
            outputs=[output_array])

    demo.queue()
    demo.launch(share=False)

if __name__ == "__main__":
    main() 
