### Personalized Multimodal AI Assistant

This project builds a multimodal AI assistant that recognizes who is interacting with it and what they are trying to do. It combines facial recognition for authentication, intent classification for command understanding, and an agent style execution layer designed to route tasks to tools or an LLM.

## Features

**Facial Recognition Authentication**
MobileNetV2 transfer learning model that identifies registered users with high accuracy.

**Intent Classification**
Custom deep learning classifier trained on a subset of the CLINC150 dataset to map user commands into actionable categories.

**Personalized Profiles**
Assistant behavior adapts based on the detected user.

**Agent Workflow**
Identity → Intent → Tool routing pipeline, compatible with LLM-based reasoning.

## Project Structure

```
AAI-590-Capstone/
├── app/
│   ├── api/
│   │   ├── assistant-mcp-server/
│   │   │   └── src/assistant_mcp/
│   │   │       ├── server.py (MCP server for tools retrival and execution)
│   │   │       ├── intent_classifier.py (intent classification code used in the MCP server)
│   │   │       └── services/ (All custom functionality code)
│   │   │           ├── alarm.py, calendar.py, media.py
│   │   │           ├── phone.py, shopping.py, smart_home.py
│   │   ├── face_classifier.py (Used in web app for facial recognition step)
│   │   ├── main.py (server used by web app to access facial recognition)
│   │   └── models.py
│   └── ui/app/
│       └── src/
│           ├── App.tsx
│           └── pages/
│               ├── ChatbotPage.tsx (Main interface UI page)
│               └── LandingPage.tsx (Authentication page)
├── facial recognition/
│   ├── CNN_Classifier.ipynb (training and eval of facial recognition)
│   ├── collect_face_data.py (Custom data collector)
│   ├── detect_and_classify.py (test script for facial recognition)
│   └── face_classifier_transfer_final.keras
├── intent classifiers/
│   ├── IntentClassifier_Alexis.ipynb (custom picked intent classifier)
│   ├── IntentClassifier_Pallav.ipynb (custom picked intent classifier)
│   ├── IntentClassifierEDA.ipynb (EDA for full intent classes)
│   └── best_model_with_oos.pt
├── face_data/
├── assistant.py (assistnant script that handles wake word activation)
├── setup.sh (sets up venv for python)
└── README.md
```

## How It Works

**1. Face Recognition**

-Uses a MobileNetV2 backbone (ImageNet pretrained).
-Custom classification head fine tuned on team members.
-Cropped facial images collected with an automated capture script.
-Class weights + augmentation used to handle limited data.
-Achieved 100% accuracy on held out test set.

**2. Intent Classification**

-Trained using a selected subset of CLINC150 intents.
-Deep learning model: BiLSTM + attention + dense layers.
-Includes OOS detection to handle irrelevant queries.
-Supports music control, smart home style actions, timers, calendars, etc.

**3. Assistant Pipeline**

-Wake word triggers assistant.
-Webcam image is passed to face model → user identified.
-Text input is passed to intent model → intent predicted.
-Assistant loads the user profile, selects tools, and synthesizes a response.
-An LLM (Claude) can optionally be called for natural language output.

## Results
**Facial Recognition**
-Perfect identity prediction across all users.
-Strong generalization due to transfer learning.

**Intent Classifier**
-High accuracy for selected intents.
-Cleaner predictions after reducing intent space.
-OOS detection prevents incorrect routing.

**End-to-End Behavior**
-Identity recognition → intent routing → tool selection functions smoothly.
-Low latency and stable performance.

## Authors

Dimitri Dumont
Pallav Kamojjhala
Alexis Lim

## License

MIT License unless otherwise specified.
