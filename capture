import cv2
import numpy as np
import pickle
import os
from datetime import datetime
import json

class MultiShapeLearner:
    def __init__(self):
        self.shapes_db = {}
        self.current_shape_name = ""
        self.capture_mode = False
        self.capture_count = 0
        self.max_captures = 10  # Nombre de captures par forme
        
    def detect_blue_color(self, image):
        """Détection spécifique de la couleur bleue du marqueur"""
        # Convertir en HSV pour une meilleure détection de couleur
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Plage de couleurs pour le bleu
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Masque pour la couleur bleue
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Améliorer le masque
        kernel = np.ones((5, 5), np.uint8)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        
        return mask_blue
    
    def extract_shape_from_mask(self, image, mask):
        """Extraire la forme du masque de couleur"""
        # Trouver les contours dans le masque
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Prendre le plus grand contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Créer un masque pour le contour
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)
        
        # Extraire la région d'intérêt
        x, y, w, h = cv2.boundingRect(largest_contour)
        roi = contour_mask[y:y+h, x:x+w]
        
        # Redimensionner à une taille standard pour la comparaison
        roi_resized = cv2.resize(roi, (100, 100))
        
        return roi_resized, largest_contour
    
    def extract_shape_features(self, shape_mask, contour):
        """Extraire les caractéristiques de la forme"""
        # Calculer les moments
        moments = cv2.moments(contour)
        
        # Moments de Hu (7 invariants)
        hu_moments = cv2.HuMoments(moments)
        
        # Log transformation pour normaliser
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        # Caractéristiques géométriques
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Rectangle englobant
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Convex hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Circularité
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Moments centraux
        mu20 = moments['mu20'] / area if area > 0 else 0
        mu02 = moments['mu02'] / area if area > 0 else 0
        mu11 = moments['mu11'] / area if area > 0 else 0
        
        # Orientation
        orientation = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
        
        # Histogramme de la forme (simplifié)
        hist = cv2.calcHist([shape_mask], [0], None, [8], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        features = {
            'hu_moments': hu_moments.flatten(),
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'circularity': circularity,
            'orientation': orientation,
            'histogram': hist,
            'mask': shape_mask  # Masque binaire de la forme
        }
        
        return features
    
    def learn_shape(self, image, shape_name, capture_index):
        """Apprendre une nouvelle capture d'une forme"""
        # Détection de la couleur bleue
        blue_mask = self.detect_blue_color(image)
        
        # Extraire la forme
        result = self.extract_shape_from_mask(image, blue_mask)
        if result is None:
            print("Aucune forme bleue détectée!")
            return False
        
        shape_mask, contour = result
        
        # Extraire les caractéristiques
        features = self.extract_shape_features(shape_mask, contour)
        
        # Initialiser la forme si elle n'existe pas
        if shape_name not in self.shapes_db:
            self.shapes_db[shape_name] = {
                'features_list': [],
                'captures': [],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'average_features': None
            }
        
        # Ajouter la capture
        self.shapes_db[shape_name]['features_list'].append(features)
        self.shapes_db[shape_name]['captures'].append(image)
        
        print(f"Capture {capture_index + 1}/{self.max_captures} pour '{shape_name}' ajoutée!")
        
        return True
    
    def compute_average_features(self):
        """Calculer les caractéristiques moyennes pour chaque forme"""
        for shape_name, data in self.shapes_db.items():
            if data['features_list']:
                # Moyenne des caractéristiques numériques
                avg_features = {}
                
                # Pour chaque caractéristique numérique
                for key in data['features_list'][0].keys():
                    if key != 'mask' and key != 'histogram':
                        values = [f[key] for f in data['features_list']]
                        if isinstance(values[0], np.ndarray):
                            avg_features[key] = np.mean(values, axis=0)
                        else:
                            avg_features[key] = np.mean(values)
                
                # Moyenne des histogrammes
                histograms = [f['histogram'] for f in data['features_list']]
                avg_features['histogram'] = np.mean(histograms, axis=0)
                
                # Masque représentatif (le plus proche de la moyenne)
                data['average_features'] = avg_features
                
                print(f"Forme '{shape_name}': {len(data['features_list'])} captures analysées")
    
    def save_database(self, filename="multi_shapes_database.pkl"):
        """Sauvegarder la base de données"""
        self.compute_average_features()
        
        # Ne sauvegarder que les données essentielles pour réduire la taille
        compact_db = {}
        for shape_name, data in self.shapes_db.items():
            compact_db[shape_name] = {
                'average_features': data['average_features'],
                'timestamp': data['timestamp'],
                'sample_count': len(data['features_list'])
            }
        
        with open(filename, 'wb') as f:
            pickle.dump(compact_db, f)
        
        # Sauvegarder aussi en JSON pour lisibilité
        json_data = {}
        for shape_name, data in compact_db.items():
            json_data[shape_name] = {
                'timestamp': data['timestamp'],
                'sample_count': data['sample_count'],
                'features': {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                           for k, v in data['average_features'].items() 
                           if k != 'mask'}
            }
        
        with open("shapes_info.json", 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Base de données sauvegardée dans {filename}")
        print(f"Information détaillée dans shapes_info.json")
    
    def load_database(self, filename="multi_shapes_database.pkl"):
        """Charger la base de données"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.shapes_db = pickle.load(f)
            print(f"Base de données chargée: {len(self.shapes_db)} formes")
            return True
        return False
    
    def display_shapes_info(self):
        """Afficher les informations sur les formes apprises"""
        print("\n=== FORMES APPRISES ===")
        for shape_name, data in self.shapes_db.items():
            sample_count = len(data['features_list']) if 'features_list' in data else data.get('sample_count', 0)
            print(f"{shape_name}: {sample_count} captures")
        print("======================\n")

def main():
    learner = MultiShapeLearner()
    
    # Initialiser la webcam
    cap = cv2.VideoCapture("http://192.168.4.100:4747/video")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("=== PROGRAMME D'APPRENTISSAGE DE FORMES MULTIPLES ===")
    print("Instructions:")
    print("1. Appuyez sur 'n' pour apprendre une NOUVELLE forme")
    print("2. Appuyez sur 'a' pour ajouter des captures à la forme en cours")
    print("3. Appuyez sur 's' pour sauvegarder la base de données")
    print("4. Appuyez sur 'i' pour voir les informations")
    print("5. Appuyez sur 'q' pour quitter")
    print("6. Appuyez sur 'd' pour supprimer la dernière forme")
    
    current_shape = None
    capture_index = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame = frame.copy()
        
        # Afficher le nom de la forme en cours
        if current_shape:
            cv2.putText(display_frame, f"Forme en cours: {current_shape}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Capture: {capture_index + 1}/{learner.max_captures}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Détecter et afficher la forme bleue
        blue_mask = learner.detect_blue_color(frame)
        result = learner.extract_shape_from_mask(frame, blue_mask)
        
        if result:
            shape_mask, contour = result
            cv2.drawContours(display_frame, [contour], -1, (0, 255, 0), 2)
            
            # Rectangle englobant
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(display_frame, "Forme bleue detectee", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Afficher les commandes
        cv2.putText(display_frame, "n: Nouvelle forme", (10, 400), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display_frame, "a: Ajouter capture", (10, 425), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display_frame, "s: Sauvegarder", (10, 450), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display_frame, "q: Quitter", (10, 475), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Apprentissage de Formes Multiples", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('n'):
            # Nouvelle forme
            shape_name = input("\nEntrez le nom de la nouvelle forme: ")
            current_shape = shape_name
            capture_index = 0
            
            # Première capture
            if learner.learn_shape(frame, shape_name, capture_index):
                capture_index += 1
                print(f"Commencez l'apprentissage de '{shape_name}'")
                print(f"Placez la forme sous différents angles et appuyez sur 'a'")
        
        elif key == ord('a') and current_shape:
            # Ajouter une capture à la forme en cours
            if capture_index < learner.max_captures:
                if learner.learn_shape(frame, current_shape, capture_index):
                    capture_index += 1
                    
                    if capture_index >= learner.max_captures:
                        print(f"\nApprentissage de '{current_shape}' termine!")
                        print(f"{learner.max_captures} captures enregistrees.")
                        current_shape = None
                        capture_index = 0
            else:
                print(f"Nombre maximum de captures atteint pour '{current_shape}'")
        
        elif key == ord('s'):
            learner.save_database()
            print("Base de donnees sauvegardee!")
        
        elif key == ord('i'):
            learner.display_shapes_info()
        
        elif key == ord('d'):
            # Supprimer la dernière forme
            if learner.shapes_db:
                last_shape = list(learner.shapes_db.keys())[-1]
                del learner.shapes_db[last_shape]
                print(f"Forme '{last_shape}' supprimee!")
                current_shape = None
                capture_index = 0
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Sauvegarder avant de quitter
    if learner.shapes_db:
        if input("\nSauvegarder la base de donnees avant de quitter? (o/n): ").lower() == 'o':
            learner.save_database()

if __name__ == "__main__":
    main()
