# Lathe: An AI-Powered Development Environment

This project, codenamed "Lathe", is an interactive, AI-powered development environment built with Streamlit. It acts as a sophisticated co-pilot for software engineering tasks, enabling developers to seamlessly integrate large language models (LLMs) into their coding workflow.

## Overview

The application provides a chat-based interface for interacting with AI models like Google's Gemini. It's designed to understand the context of your project by managing source files, code snippets, and conversation history. You can use it to generate new code, refactor existing files, ask architectural questions, and execute complex, multi-step tasks using a prompt flow engine.

The core workflow revolves around providing the AI with relevant context, receiving code suggestions as `revisions` (modifications) and `artifacts` (new files), staging them as a batch, and finally `accepting` or `rejecting` the entire change-set.

## Key Features

- **Interactive Chat Interface**: A Streamlit-based UI for real-time conversation with the AI.
- **Dynamic Context Management**: Automatically builds prompts using relevant files (`Artifacts`), temporary `Snippets`, and conversation history.
- **Multi-Model Support**: Pluggable architecture (`ai_models.py`) to support various models, including Google's Vertex AI (Gemini) and local APIs.
- **Batch Proposal Workflow**: A structured process to stage, review, and then accept or reject an entire set of AI-generated code changes atomically. This ensures dependent files are updated together and allows for the creation of new files.
- **Automated Feedback Loop**: Rejecting a change-set automatically prompts the AI to retry, incorporating your rejection reasons, the original prompt, and the rejected code as corrective feedback.
- **Prompt Flow Engine**: Execute multi-step, chained AI tasks defined in simple JSON files. This allows for creating complex, repeatable workflows (`engine_runner.py`).
- **Instruction Profiles**: Easily switch the AI's persona, system instructions, and behavior by loading different YAML-based profiles.
- **Conversation Summarization**: An optional "Laconic History" feature automatically summarizes conversation turns to manage context limits during long sessions.
- **Session Persistence**: Save and load entire development sessions, including context, chat history, and configuration.
- **Extensible Prompting**: Customize system instructions and templates via `prompts/system_prompts.yaml` to tailor the AI's behavior.

## Core Components

- `main.py`: The main application entry point that orchestrates the UI and backend logic.
- `ui_sidebar.py`: Renders all the interactive widgets in the Streamlit sidebar for managing the session, project files, and model settings.
- `session_manager.py`: The state management engine. It handles the session's lifecycle, including loading/saving files, chat history, and the code proposal/approval workflow.
- `prompt_builder.py`: Dynamically constructs the context-rich prompts sent to the AI model by assembling various pieces of information.
- `ai_models.py`: An abstraction layer for communicating with different LLMs, providing a consistent interface for token counting and generation.
- `engine_runner.py`: Executes prompt flow definitions, allowing for chained, multi-step interactions with the AI model.

## Getting Started

### Prerequisites

- Python 3.9+
- Access to a supported AI model API (e.g., Google Cloud Platform project with Vertex AI enabled).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment:**
    - For Google Cloud, ensure your environment is authenticated:
      ```bash
      gcloud auth application-default login
      ```
    - The application will create `prompt_flows/`, `instruction_profiles/`, and `chat_sessions/` directories on first run. You can add your own custom JSON flows and YAML profiles to the respective folders.

### Running the Application

Launch the Streamlit app from your terminal:

```bash
streamlit run main.py
```

The application will open in your web browser.

## Basic Usage

1.  **Start a Session**: Use the sidebar to start a new session or load a previous one.
2.  **Set Project Root**: Define the root directory of the project you want to work on.
3.  **Add Artifacts**: Add key source files to the context by selecting them in the "Artifact Manager". The AI will use these files as its primary source of truth.
4.  **Chat**: Interact with the AI. Ask it to write new functions, create new files, refactor code, or explain concepts. Reference files using their filenames.
5.  **Stage Proposals**: When the AI suggests changes (new files or modifications), click "Stage All Revisions" to bundle them into a single change-set for review. This applies the changes to your local file system and creates backups of any modified files.
6.  **Accept or Reject**: A review panel will appear.
    - Click **Accept All** to commit the changes. The original files are archived, the source files are updated, and the AI's background context is regenerated.
    - Click **Reject All** and provide a reason. The changes are discarded from your file system by reverting to the backups. The AI is automatically instructed to try again based on your feedback.