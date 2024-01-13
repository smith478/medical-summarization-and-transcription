import io
import nltk
import numpy as np
import soundfile as sf
import streamlit as st
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

TARGET_SAMPLE_RATE = 16000
MODEL_PATH = 'openai/whisper-large-v3'
SUMMARIZATION_PATH = "Falconsai/medical_summarization"


@st.cache_resource 
def load_model():
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
    model.config.forced_decoder_ids = None
    return model

@st.cache_resource 
def load_processor():
    processor = WhisperProcessor.from_pretrained(MODEL_PATH)
    return processor

@st.cache_resource 
def load_summarization_model():
    model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_PATH)
    return model

@st.cache_resource 
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_PATH)
    return tokenizer

# nltk.download('punkt')
def split_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return '\n'.join(sentences)

model = load_model()
processor = load_processor()
summarization_model = load_summarization_model()
tokenizer = load_tokenizer()

def main():
    st.title("Speech to summarization")

    st.write(
        """  
    -   Upload a wav file, transcribe it, then summarize it!
        """
    )

    st.text("")

    c1, c2, c3 = st.columns([1, 4, 1])

    with c2:

        with st.form(key="my_form"):

            audio_byte_data = st.file_uploader("", type=[".wav"])

            st.info(f"""👆 Upload a .wav file.""")

            submit_button = st.form_submit_button(label="Transcribe")

    if audio_byte_data is not None:
        st.audio(audio_byte_data, format="wav")

        input_audio_data, loaded_sample_rate = sf.read(io.BytesIO(audio_byte_data.getvalue()), dtype='float32', always_2d=True)
        assert loaded_sample_rate == TARGET_SAMPLE_RATE, f'sample rate must be {TARGET_SAMPLE_RATE}. Uploaded file had sample rate of {loaded_sample_rate}'

        print(f'Input audio: {input_audio_data}')
        print(f'Sample rate: {loaded_sample_rate}')
        
        # Calculate the number of samples per chunk
        samples_per_chunk = int(loaded_sample_rate * 30)

        # Split the audio data into chunks
        chunks = np.array_split(input_audio_data, np.ceil(len(input_audio_data) / samples_per_chunk))

        transcriptions = []

        for chunk in chunks:
            input_features = processor(chunk.T, sampling_rate=loaded_sample_rate, return_tensors="pt").input_features 
            predicted_ids = model.generate(input_features)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            transcriptions.append(transcription[0])

        # Join the transcriptions together
        transcription = ' '.join(transcriptions)

        st.info(transcription)
        st.write(f'Transcription: {transcription}')

        c0, c1 = st.columns([2, 2])

        with c0:
            st.download_button(
                "Download the transcription",
                transcription[0],
                file_name=None,
                mime=None,
                key=None,
                help=None,
                on_click=None,
                args=None,
                kwargs=None,
            )

        # Summarize the extracted text
        summarizer = pipeline("summarization", model=summarization_model, tokenizer=tokenizer)
        summary = summarizer(transcription, max_length=300, min_length=5, do_sample=False)[0]['summary_text']
        # st.write(f'Summary: {split_into_sentences(summary)}')
        st.write(f'Summary: {summary}')


if __name__ == "__main__":
    main()
