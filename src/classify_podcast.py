import pickle
from typing import List

class ClassifyPodcast:
   
    
    def __init__(self, model=None, model_path=None):
        """
        Initializing the model by loading the weights
        
        """
        if model is not None:
            self.model = model
        elif model_path is not None:
            import pickle
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            raise ValueError("Fournir soit 'model' soit 'model_path'")
    
    def classify_podcast(self, keywords: list[str]) -> bool:
        """
        Call the function that will classify the podcast and return True or False
        
        Args:
            keywords: podcast keywords list
                      Exemple: ['education', 'kids', 'science']
        
        Returns:
            bool: True if kid-friendly,else  False 
        """
        
        keywords_text = ' '.join([str(k).lower() for k in keywords])
        
        
        prediction = self.model.predict([keywords_text])[0]
        
        
        return bool(prediction)