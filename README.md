To use this code, create a new virtual environment:

    python3 -m venv voice_id
    source voice_id/bin/activate

Install the dependencies using pip:

    pip install -r requirements.txt

librosa depends on libffmpg to generate MFCCs, and must be installed before
running voice_id.py.

Get the data from LibriSpeech:

    python get_data.py

Run the training script:

    python voice_id.py

To test the model:

    python test_model.py
