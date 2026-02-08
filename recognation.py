import cv2
import numpy as np
import pickle
from scipy.spatial.distance import cosine, euclidean

class ShapeRecognizer:
    def __init__(self, database_file="shapes_database.pkl"):
        self.shapes_db = {}
        self.load_database(database_file)
        self.similarity_threshold = 0.70  # Seuil ajusté
    
    def load_database(self, filename):
        """Charger la base de données d'apprentissage"""
        try:
            with open(filename, 'rb') as f:
                self.shapes_db = pickle.load(f)
            print(f"Base de données chargée: {len(self.shapes_db)} formes apprises")
            for name, data in self.shapes_db.items():
                num_samples = data.get('num_samples', 1)
                print(f"  - {name} ({num_samples} échantillons)")
        except FileNotFoundError:
            print(f"Fichier {filename} non trouvé. Lancez d'abord le programme d'apprentissage.")
            self.shapes_db = {}
    
    def preprocess_image(self, image):
        """Même prétraitement que lors de l'apprentissage"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        return cleaned
    
    def extract_features(self, binary_image):
        """Extraire les caractéristiques de la forme détectée - VERSION AMÉLIORÉE"""
        contours, _ = cv2.findContours(binary_image, 
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Filtrer les petits contours (bruit)
        if cv2.contourArea(largest_contour) < 500:
            return None
        
        moments = cv2.moments(largest_contour)
        hu_moments = cv2.HuMoments(moments)
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # ✅ NOUVELLES CARACTÉRISTIQUES
        
        # 1. Nombre de sommets
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        num_vertices = len(approx)
        
        # 2. Compacité
        compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        # 3. Convexité
        convexity = perimeter / cv2.arcLength(hull, True) if cv2.arcLength(hull, True) > 0 else 0
        
        # 4. Excentricité
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis > 0 else 0
        else:
            eccentricity = 0
        
        # 5. Extent
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0
        
        # 6. Équivalent diamètre
        equiv_diameter = np.sqrt(4 * area / np.pi)
        
        features = {
            'hu_moments': hu_moments.flatten(),
            'area': area,
            'perimeter': perimeter,
            'solidity': solidity,
            'aspect_ratio': aspect_ratio,
            'contour': largest_contour,
            # ✅ NOUVELLES FEATURES
            'num_vertices': num_vertices,
            'compactness': compactness,
            'convexity': convexity,
            'eccentricity': eccentricity,
            'extent': extent,
            'equiv_diameter': equiv_diameter
        }
        
        return features
    
    def compare_features(self, features1, features2):
        """Comparer deux ensembles de caractéristiques - VERSION AMÉLIORÉE"""
        
        # 1. Comparaison des moments de Hu (distance cosinus)
        try:
            hu_similarity = 1 - cosine(features1['hu_moments'], features2['hu_moments'])
        except:
            hu_similarity = 0.0
        
        # 2. Comparaison du nombre de sommets (TRÈS IMPORTANT)
        # Triangle=3, Carré=4, Cercle≈8-15
        vertices_diff = abs(features1['num_vertices'] - features2['num_vertices'])
        vertices_similarity = 1 / (1 + vertices_diff)  # 1 si identique, diminue si différent
        
        # 3. Comparaison de la compacité (TRÈS IMPORTANT pour cercle vs autres)
        compactness_diff = abs(features1['compactness'] - features2['compactness'])
        compactness_similarity = 1 - compactness_diff
        
        # 4. Autres caractéristiques
        area_ratio = min(features1['area'], features2['area']) / max(features1['area'], features2['area'])
        solidity_diff = 1 - abs(features1['solidity'] - features2['solidity'])
        aspect_diff = 1 - abs(features1['aspect_ratio'] - features2['aspect_ratio'])
        eccentricity_diff = 1 - abs(features1['eccentricity'] - features2['eccentricity'])
        extent_diff = 1 - abs(features1['extent'] - features2['extent'])
        
        # ✅ SCORE COMBINÉ avec PONDÉRATION OPTIMISÉE
        similarity_score = (
            0.25 * hu_similarity +           # Moments de Hu
            0.25 * vertices_similarity +     # ⭐ Sommets (crucial!)
            0.20 * compactness_similarity +  # ⭐ Compacité (crucial!)
            0.10 * area_ratio +              # Surface
            0.08 * solidity_diff +           # Solidité
            0.05 * aspect_diff +             # Aspect ratio
            0.04 * eccentricity_diff +       # Excentricité
            0.03 * extent_diff               # Extent
        )
        
        return similarity_score
    
    def recognize_shape(self, image):
        """Reconnaître la forme dans l'image - VERSION MULTI-ÉCHANTILLONS"""
        # Prétraitement
        processed = self.preprocess_image(image)
        
        # Extraction des caractéristiques
        features = self.extract_features(processed)
        
        if features is None:
            return None, None, 0.0, {}
        
        best_match = None
        best_score = 0.0
        all_scores = {}
        
        # Comparer avec chaque forme apprise
        for shape_name, shape_data in self.shapes_db.items():
            
            if 'features_samples' in shape_data:
                # Nouveau format avec multi-échantillons
                samples = shape_data['features_samples']
                scores = []
                
                # Comparer avec chaque échantillon
                for sample_features in samples:
                    score = self.compare_features(features, sample_features)
                    scores.append(score)
                
                # Prendre la MOYENNE des 5 meilleurs scores
                scores.sort(reverse=True)
                top_scores = scores[:min(5, len(scores))]
                avg_score = sum(top_scores) / len(top_scores)
                
                all_scores[shape_name] = {
                    'best': max(scores),
                    'average': avg_score,
                    'all': scores
                }
                
                final_score = avg_score
                
            else:
                # Ancien format (rétrocompatibilité)
                learned_features = shape_data['features']
                final_score = self.compare_features(features, learned_features)
                all_scores[shape_name] = {
                    'best': final_score,
                    'average': final_score,
                    'all': [final_score]
                }
            
            # Garder le meilleur match
            if final_score > best_score and final_score > self.similarity_threshold:
                best_score = final_score
                best_match = shape_name
        
        return best_match, features['contour'], best_score, all_scores
    
    def process_frame(self, frame):
        """Traiter une frame pour la reconnaissance"""
        # Détecter la forme
        shape_name, contour, confidence, all_scores = self.recognize_shape(frame)
        
        result_frame = frame.copy()
        
        if contour is not None:
            # Dessiner le contour
            cv2.drawContours(result_frame, [contour], -1, (0, 255, 0), 3)
            
            # Calculer le rectangle englobant
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Afficher le nom et la confiance
            if shape_name:
                label = f"{shape_name}: {confidence:.1%}"
                color = (0, 255, 0)  # Vert si reconnu
                
                # Afficher aussi le score détaillé
                if shape_name in all_scores:
                    detail = f"(meilleur: {all_scores[shape_name]['best']:.2f})"
                    cv2.putText(result_frame, detail, (x, y + h + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                label = "Forme inconnue"
                color = (0, 0, 255)  # Rouge si inconnu
            
            # Label principal
            cv2.putText(result_frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Afficher TOP 3 des scores (pour debug)
            y_offset = 60
            cv2.putText(result_frame, "Top 3 scores:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            
            for idx, (name, scores) in enumerate(sorted(all_scores.items(), 
                                      key=lambda x: x[1]['average'], 
                                      reverse=True)[:3]):
                score_text = f"{idx+1}. {name}: {scores['average']:.2f}"
                color = (0, 255, 0) if idx == 0 else (200, 200, 200)
                cv2.putText(result_frame, score_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 20
        
        return result_frame, shape_name, confidence

def main():
    recognizer = ShapeRecognizer()
    
    if not recognizer.shapes_db:
        print("Aucune forme apprise. Veuillez d'abord lancer le programme d'apprentissage.")
        return
    
    # Initialiser la webcam
    cap = cv2.VideoCapture("http://192.168.4.186:4747/video")
    
    print("\n=== Programme de reconnaissance AMÉLIORÉ ===")
    print("🎯 Nouvelles caractéristiques utilisées:")
    print("   - Nombre de sommets (25% du score)")
    print("   - Compacité (20% du score)")
    print("   - Hu moments (25% du score)")
    print("\nInstructions:")
    print("1. Placez une forme devant la webcam")
    print("2. Le programme comparera avec TOUS les échantillons")
    print("3. Appuyez sur 'q' pour quitter")
    print("4. Appuyez sur 's' pour sauvegarder une capture")
    print(f"5. Seuil de confiance: {recognizer.similarity_threshold:.0%}")
    print("\n" + "="*60 + "\n")
    
    capture_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Traiter la frame
        result_frame, shape_name, confidence = recognizer.process_frame(frame)
        
        # Afficher les informations générales
        total_shapes = len(recognizer.shapes_db)
        total_samples = sum(data.get('num_samples', 1) for data in recognizer.shapes_db.values())
        
        info_text = f"DB: {total_shapes} formes, {total_samples} echantillons"
        cv2.putText(result_frame, info_text, 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Reconnaissance AMELIOREE - 'q' pour quitter", result_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            # Sauvegarder la capture
            filename = f"capture_{capture_count:04d}.png"
            cv2.imwrite(filename, frame)
            print(f"✅ Capture sauvegardée: {filename}")
            if shape_name:
                print(f"   Forme détectée: {shape_name} ({confidence:.1%})")
            capture_count += 1
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
