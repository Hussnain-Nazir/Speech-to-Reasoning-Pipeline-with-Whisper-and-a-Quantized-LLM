# Speech-to-Reasoning Pipeline with Whisper and a Quantized LLM

## Project Summary

This project implements an end-to-end pipeline that converts spoken audio into text using OpenAI
Whisper, cleans the resulting transcription, and passes it to a 4-bit quantized reasoning language
model (DeepSeek-R1-Distill-Llama-8B, loaded via Unsloth) to produce a step-by-step reasoning
answer. The entire pipeline is implemented in a single, self-contained Google Colab notebook that
runs from start to finish with no manual configuration required.

## Objective

Build a Google Colab application that:

1. Accepts an audio file.
2. Transcribes speech to text using OpenAI Whisper.
3. Cleans the transcription.
4. Sends the transcription to a quantized reasoning LLM.
5. Displays the model's final, logically reasoned answer.

## Folder Structure  

```
Speech-to-Reasoning/
├── Speech_to_Reasoning_Pipeline.ipynb    ← Notebook
├── audio/      ← Auto-generated (gTTS) and/or user-uploaded audio clips
├── outputs/           Saved transcription + reasoning results
│   ├── pipeline_results.txt
├── requirements.txt                         
└── README.md   
```

## Environment

- Platform: Google Colab, Free Tier
- Hardware: T4 GPU (Runtime > Change runtime type > T4 GPU)
- Language: Python 3
- Key libraries: openai-whisper, unsloth, unsloth_zoo, transformers, accelerate, bitsandbytes, gTTS

## Models Used

| Component        | Model                                                       | Purpose                         |
|-------------------|--------------------------------------------------------------|----------------------------------|
| Speech recognition | Whisper `base` (OpenAI)                                       | Speech-to-text transcription     |
| Reasoning          | `unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit`       | Step-by-step reasoning / QA      |

Both models were chosen specifically to run reliably within the memory limits of a free-tier T4
GPU (16 GB VRAM).

## How to Run

1. Open `Speech_to_Reasoning_Pipeline.ipynb` in Google Colab (File > Upload notebook, or
   drag-and-drop).
2. Set the runtime to GPU: **Runtime > Change runtime type > T4 GPU**.
3. Run all cells in order: **Runtime > Run all**.
4. The notebook automatically generates three sample audio clips (using gTTS) and runs the full
   pipeline on each one. No file upload is required, though an optional upload cell is provided if
   you want to test your own audio.
5. Results (transcriptions and reasoning responses) are printed inline and saved to
   `Speech_to_Reasoning/outputs/pipeline_results.txt`.

## Key Design Decisions

- **Single self-contained notebook** - no external file dependencies, matching internship
  submission preferences.
- **No emojis** anywhere in code, comments, markdown, or generated output.
- **Sequential model loading** - Whisper is explicitly unloaded (`del` + `gc.collect()` +
  `torch.cuda.empty_cache()`) before the reasoning LLM is loaded, to avoid out-of-memory errors
  from holding two models on the GPU simultaneously on a free-tier T4.
- **Standard chat templating** - the reasoning model's prompt is built with the tokenizer's native
  `apply_chat_template()` method rather than any custom/experimental chat-template import.
- **Self-contained testing** - sample audio is generated on the fly with gTTS so the notebook can
  be graded/run without requiring any external audio files.
