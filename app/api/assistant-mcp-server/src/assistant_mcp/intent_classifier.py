"""Intent classifier for filtering relevant tools."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter


class Vocabulary:
    """Vocabulary for text encoding."""
    def __init__(self, min_freq=1):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_freq = Counter()
        self.min_freq = min_freq

    def build_vocab(self, texts):
        """Build vocabulary from texts."""
        for text in texts:
            words = text.lower().split()
            self.word_freq.update(words)

        idx = 2
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def encode(self, text):
        """Encode text to indices."""
        words = text.lower().split()
        return [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]

    def decode(self, indices):
        """Decode indices to text."""
        return ' '.join([self.idx2word.get(idx, '<UNK>') for idx in indices])


class IntentClassifierWithOOS(nn.Module):
    """
    Deep Learning Intent Classifier with OOS Detection.
    - Embedding Layer
    - Bidirectional LSTM
    - Attention Mechanism
    - Fully Connected Layers with Dropout
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers=2, dropout=0.5):
        super(IntentClassifierWithOOS, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )

        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim // 2)

    def attention_net(self, lstm_output):
        """Attention mechanism to focus on important parts of the sequence."""
        attention_weights = torch.tanh(self.attention(lstm_output))
        attention_weights = torch.softmax(attention_weights, dim=1)
        weighted_output = lstm_output * attention_weights
        return torch.sum(weighted_output, dim=1)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)

        lstm_out, (hidden, cell) = self.lstm(embedded)

        attended = self.attention_net(lstm_out)
        attended = self.layer_norm(attended)

        x = F.relu(self.fc1(attended))
        x = self.batch_norm1(x)
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.batch_norm2(x)
        x = self.dropout(x)

        output = self.fc3(x)

        return output


class IntentClassifier:
    """Wrapper class for intent classification inference."""
    
    # Intent classes that the model can predict
    INTENT_CLASSES = [
    'alarm',                # 0
    'calendar',             # 1
    'calendar_update',      # 2
    'change_speed',         # 3
    'change_volume',        # 4
    'find_phone',           # 5
    'oos',                  # 6
    'play_music',           # 7
    'shopping_list',        # 8
    'shopping_list_update', # 9
    'smart_home',           # 10
    'timer'                 # 11
]
    
    # Mapping from predicted intents to relevant tools
    # Some intents map to multiple tools (e.g., shopping_list queries might need both read and update)
    INTENT_TO_TOOLS = {
        'calendar': ['calendar', 'calendar_update'],
        'calendar_update': ['calendar_update', 'calendar'],
        'alarm': ['alarm'],
        'change_speed': ['change_speed'],
        'change_volume': ['change_volume'],
        'shopping_list': ['shopping_list', 'shopping_list_update'],
        'shopping_list_update': ['shopping_list_update', 'shopping_list'],
        'timer': ['timer'],
        'find_phone': ['find_phone'],
        'play_music': ['play_music', 'change_volume', 'change_speed'],
        'smart_home': ['smart_home'],
        'oos': []  # OOS means no tools are relevant
    }
    
    def __init__(self, model_path, vocab_path=None, device=None):
        """
        Initialize the intent classifier.
        
        Args:
            model_path: Path to the saved model weights (.pt file)
            vocab_path: Path to saved vocabulary (optional, will build from scratch if not provided)
            device: torch device to use (defaults to cuda if available, else cpu)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len = 50
        
        # Initialize vocabulary (you'll need to load or rebuild this)
        self.vocab = None  # Will be set via load_vocab or build_vocab
        
        # Model hyperparameters (must match training)
        self.embedding_dim = 128
        self.hidden_dim = 256
        self.n_layers = 2
        self.dropout = 0.5
        self.num_classes = len(self.INTENT_CLASSES)
        
        # Initialize model
        self.model = None
        self.label_to_intent = {i: intent for i, intent in enumerate(self.INTENT_CLASSES)}
        self.intent_to_label = {intent: i for i, intent in enumerate(self.INTENT_CLASSES)}
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def build_vocab(self, texts, min_freq=2):
        """Build vocabulary from training texts."""
        self.vocab = Vocabulary(min_freq=min_freq)
        self.vocab.build_vocab(texts)
        return self.vocab
    
    def load_vocab(self, vocab_path):
        """Load pre-saved vocabulary."""
        import pickle
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
    
    def save_vocab(self, vocab_path):
        """Save vocabulary to file."""
        import pickle
        with open(vocab_path, 'wb') as f:
            pickle.dump(self.vocab, f)
    
    def load_model(self, model_path):
        """Load trained model weights."""
        if self.vocab is None:
            raise ValueError("Vocabulary must be loaded before model")
        
        self.model = IntentClassifierWithOOS(
            vocab_size=len(self.vocab.word2idx),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.num_classes,
            n_layers=self.n_layers,
            dropout=self.dropout
        )
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
    
    def predict_intent(self, text, oos_threshold=0.5, top_k=3):
        """
        Predict intent for a given text.
        
        Args:
            text: Input text string
            oos_threshold: Confidence threshold for flagging low confidence
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Encode text
        tokens = self.vocab.encode(text)
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + [0] * (self.max_len - len(tokens))
        
        input_tensor = torch.LongTensor([tokens]).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            prediction = np.argmax(probs)
            confidence = probs[prediction]
        
        intent = self.label_to_intent[prediction]
        
        # Get all probabilities
        all_probs = {
            self.label_to_intent[i]: float(prob)
            for i, prob in enumerate(probs)
        }
        
        # Get top k predictions
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Check if OOS
        is_oos = intent == 'oos'
        oos_confidence = all_probs.get('oos', 0.0)
        
        # Determine if we should flag as uncertain
        max_in_scope_prob = max([p for intent_name, p in all_probs.items()
                                  if intent_name != 'oos'])
        
        return {
            'intent': intent,
            'confidence': float(confidence),
            'is_oos': is_oos,
            'oos_confidence': float(oos_confidence),
            'max_in_scope_confidence': float(max_in_scope_prob),
            'all_probabilities': all_probs,
            'top_predictions': sorted_probs,
            'needs_review': confidence < oos_threshold
        }
    
    def get_relevant_tools(self, text, confidence_threshold=0.3, top_n_intents=2):
        """
        Get relevant tools based on predicted intent.
        
        Args:
            text: Input text string
            confidence_threshold: Minimum confidence to consider an intent
            top_n_intents: Consider top N predicted intents for tool selection
            
        Returns:
            List of relevant tool names
        """
        prediction = self.predict_intent(text)
        
        # If OOS or very low confidence, return empty list
        if prediction['is_oos'] or prediction['confidence'] < confidence_threshold:
            return []
        
        # Collect tools from top predicted intents
        relevant_tools = set()
        
        for intent, confidence in prediction['top_predictions'][:top_n_intents]:
            if intent != 'oos' and confidence >= confidence_threshold:
                tools = self.INTENT_TO_TOOLS.get(intent, [])
                relevant_tools.update(tools)
        
        return list(relevant_tools)
