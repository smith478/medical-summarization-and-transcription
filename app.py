import io
import nltk
import numpy as np
import soundfile as sf
import streamlit as st
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering, pipeline

TARGET_SAMPLE_RATE = 16000
MODEL_PATH = 'openai/whisper-large-v3'
SUMMARIZATION_PATH = "Falconsai/medical_summarization"
QA_MODEL_PATH = "deepset/roberta-base-squad2"

# Set page configuration to wide layout
st.set_page_config(layout='wide')

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

@st.cache_resource
def load_qa_model():
    model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_PATH)
    return model

@st.cache_resource
def load_qa_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_PATH)
    return tokenizer

# nltk.download('punkt')
def split_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return '\n'.join(sentences)

model = load_model()
processor = load_processor()
summarization_model = load_summarization_model()
tokenizer = load_tokenizer()
qa_model = load_qa_model()
qa_tokenizer = load_qa_tokenizer()
qa = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)

def update_inference_required():
    st.session_state['inference_required'] = True

def main():
    st.title("Speech to summarization")

    st.write(
        """  
    -   Upload a wav file, transcribe it, then summarize it!
        """
    )

    st.text("")

    if 'inference_required' not in st.session_state:
        st.session_state['inference_required'] = True

    c1, c2, c3 = st.columns([1, 4, 1])

    with c2:

        audio_byte_data = st.file_uploader("", type=[".wav"], on_change=update_inference_required)

        st.info(f"""👆 Upload a .wav file.""")

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

        if st.session_state['inference_required']:
            transcriptions = []

            for chunk in chunks:
                input_features = processor(chunk.T, sampling_rate=loaded_sample_rate, return_tensors="pt").input_features 
                predicted_ids = model.generate(input_features)
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
                transcriptions.append(transcription[0])

            # Join the transcriptions together
            transcription = ' '.join(transcriptions)

            # Summarize the extracted text
            summarizer = pipeline("summarization", model=summarization_model, tokenizer=tokenizer)
            summary = summarizer(transcription, max_length=300, min_length=5, do_sample=False)[0]['summary_text']

            # Update the session states
            st.session_state['transcriptions'] = transcriptions
            st.session_state['transcription'] = transcription
            st.session_state['summary'] = summary
            st.session_state['inference_required'] = False

        st.info(st.session_state['transcription'])
        # st.write(f'Transcription: {transcription}')

        c0, c1 = st.columns([2, 2])

        with c0:
            st.download_button(
                "Download the transcription",
                st.session_state['transcription'],
                file_name=None,
                mime=None,
                key=None,
                help=None,
                on_click=None,
                args=None,
                kwargs=None,
            )
        
        # Split the summary into sentences and display each sentence on a new line
        sentences = st.session_state['summary'].split('. ')
        for sentence in sentences:
            st.write(f'* {sentence}')

        with st.form('question_form'):
            # Ask questions about the transcription
            question = st.text_input("Ask a question about the transcription:")
            submitted = st.form_submit_button('Submit Question')

            if submitted:
                answer = qa(question=question, context=st.session_state['transcription'])
                st.write(f"Answer: {answer['answer']}") 


if __name__ == "__main__":
    main()
