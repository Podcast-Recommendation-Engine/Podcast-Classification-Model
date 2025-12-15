import pickle
import os
from typing import List


class ClassifyPodcast:
  
    
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')
    
    def __init__(self):
        """
        Initializing the model by loading the weights
        Charge  le modèle depuis models/best_model.pkl
        """
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle depuis le fichier pickle"""
        if not os.path.exists(self.MODEL_PATH):
            raise FileNotFoundError(
                f"Modèle introuvable: {self.MODEL_PATH}\n"
                f"Exécutez d'abord le notebook pour générer le modèle."
            )
        
        with open(self.MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)
    
    def classify_podcast(self, keywords: List[str]) -> bool:
        """
        Call the function that will classify the podcast and return True or False
        
        Args:
            keywords: Liste de keywords du podcast
        
        Returns:
            bool: True si kid-friendly, False sinon
        """
        if not keywords:
            raise ValueError("keywords ne peut pas être vide")
        
        keywords_text = ' '.join([str(k).lower() for k in keywords])
        prediction = self.model.predict([keywords_text])[0]
        
        return bool(prediction)
