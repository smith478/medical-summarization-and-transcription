import io
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
        print('Running processor')
        input_features = processor(input_audio_data.T, sampling_rate=loaded_sample_rate, return_tensors="pt").input_features 
        # generate token ids
        print('Running model')
        predicted_ids = model.generate(input_features)

        print('Decoding transcription')
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        print(transcription)

        st.info(transcription[0])
        st.write(f'Transcription: {transcription[0]}')

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
        summary = summarizer(transcription[0], max_length=300, min_length=5, do_sample=False)[0]['summary_text']
        st.write(f'Summary: {summary}')


if __name__ == "__main__":
    main()
