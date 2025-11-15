"""
Clasificador de Baches según la Norma ASTM D6433-03

Este módulo implementa la clasificación de severidad de baches de acuerdo
con la norma ASTM D6433-03 (Standard Practice for Roads and Parking Lots
Pavement Condition Index Surveys).

Clasificación de Severidad de Baches (Potholes):
- Low (L): Diámetro < 200 mm (profundidad < 25 mm)
- Medium (M): Diámetro 200-450 mm o profundidad 25-50 mm
- High (H): Diámetro > 450 mm o profundidad > 50 mm

Para este proyecto, usaremos el diámetro del bache calculado a partir
de la máscara de segmentación.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class SeverityLevel(Enum):
    """Niveles de severidad según ASTM D6433-03"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    UNKNOWN = "Unknown"


@dataclass
class PotholeInfo:
    """Información detallada de un bache detectado"""
    id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    mask: np.ndarray
    confidence: float
    diameter_mm: float
    diameter_px: float
    area_px: float
    severity: SeverityLevel
    severity_score: float  # 0-100, donde 100 es más severo

    def to_dict(self) -> Dict:
        """Convertir a diccionario"""
        return {
            'id': self.id,
            'bbox': self.bbox,
            'confidence': float(self.confidence),
            'diameter_mm': float(self.diameter_mm),
            'diameter_px': float(self.diameter_px),
            'area_px': float(self.area_px),
            'severity': self.severity.value,
            'severity_score': float(self.severity_score)
        }


class ASTMPotholeClassifier:
    """
    Clasificador de baches según norma ASTM D6433-03

    Parámetros:
        pixels_per_mm: Factor de conversión de píxeles a milímetros
                       Puede calibrarse si se conoce la distancia de la cámara
        low_threshold_mm: Umbral inferior en mm (default: 200)
        high_threshold_mm: Umbral superior en mm (default: 450)
    """

    def __init__(
        self,
        pixels_per_mm: float = 1.0,
        low_threshold_mm: float = 200.0,
        high_threshold_mm: float = 450.0
    ):
        self.pixels_per_mm = pixels_per_mm
        self.low_threshold_mm = low_threshold_mm
        self.high_threshold_mm = high_threshold_mm

        # Umbrales en píxeles
        self.low_threshold_px = low_threshold_mm * pixels_per_mm
        self.high_threshold_px = high_threshold_mm * pixels_per_mm

    def set_calibration(self, pixels_per_mm: float):
        """
        Establecer calibración de píxeles a milímetros

        Args:
            pixels_per_mm: Factor de conversión
        """
        self.pixels_per_mm = pixels_per_mm
        self.low_threshold_px = self.low_threshold_mm * pixels_per_mm
        self.high_threshold_px = self.high_threshold_mm * pixels_per_mm

    @staticmethod
    def calculate_diameter_from_mask(mask: np.ndarray) -> Tuple[float, float]:
        """
        Calcular el diámetro de un bache a partir de su máscara

        Usa dos métodos:
        1. Diámetro del círculo mínimo que encierra la máscara
        2. Diámetro equivalente (área como círculo)

        Args:
            mask: Máscara binaria del bache (numpy array)

        Returns:
            Tuple[float, float]: (diameter_enclosing, diameter_equivalent)
        """
        # Asegurar que la máscara sea binaria
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        # Encontrar contornos
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return 0.0, 0.0

        # Usar el contorno más grande
        largest_contour = max(contours, key=cv2.contourArea)

        # Método 1: Círculo mínimo que encierra el contorno
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        diameter_enclosing = 2 * radius

        # Método 2: Diámetro equivalente basado en área
        area = cv2.contourArea(largest_contour)
        diameter_equivalent = 2 * np.sqrt(area / np.pi)

        # Usar el promedio de ambos métodos para mayor robustez
        diameter_avg = (diameter_enclosing + diameter_equivalent) / 2

        return diameter_avg, area

    def classify_severity(self, diameter_mm: float) -> Tuple[SeverityLevel, float]:
        """
        Clasificar severidad basada en diámetro

        Args:
            diameter_mm: Diámetro en milímetros

        Returns:
            Tuple[SeverityLevel, float]: (nivel de severidad, score 0-100)
        """
        if diameter_mm < 0:
            return SeverityLevel.UNKNOWN, 0.0

        # Clasificar según umbrales ASTM D6433-03
        if diameter_mm < self.low_threshold_mm:
            severity = SeverityLevel.LOW
            # Score de 0-33 para Low
            score = min(33, (diameter_mm / self.low_threshold_mm) * 33)
        elif diameter_mm < self.high_threshold_mm:
            severity = SeverityLevel.MEDIUM
            # Score de 34-66 para Medium
            range_size = self.high_threshold_mm - self.low_threshold_mm
            position_in_range = (diameter_mm - self.low_threshold_mm) / range_size
            score = 33 + (position_in_range * 33)
        else:
            severity = SeverityLevel.HIGH
            # Score de 67-100 para High
            excess = diameter_mm - self.high_threshold_mm
            # Escalar logarítmicamente para valores muy grandes
            score = 67 + min(33, (np.log1p(excess) / np.log1p(self.high_threshold_mm)) * 33)

        return severity, score

    def process_detection(
        self,
        mask: np.ndarray,
        bbox: Tuple[float, float, float, float],
        confidence: float,
        detection_id: int
    ) -> PotholeInfo:
        """
        Procesar una detección individual

        Args:
            mask: Máscara de segmentación
            bbox: Bounding box (x1, y1, x2, y2)
            confidence: Confianza de la detección
            detection_id: ID único de la detección

        Returns:
            PotholeInfo: Información completa del bache
        """
        # Calcular diámetro en píxeles
        diameter_px, area_px = self.calculate_diameter_from_mask(mask)

        # Convertir a milímetros
        diameter_mm = diameter_px / self.pixels_per_mm

        # Clasificar severidad
        severity, severity_score = self.classify_severity(diameter_mm)

        # Crear objeto PotholeInfo
        pothole = PotholeInfo(
            id=detection_id,
            bbox=bbox,
            mask=mask,
            confidence=confidence,
            diameter_mm=diameter_mm,
            diameter_px=diameter_px,
            area_px=area_px,
            severity=severity,
            severity_score=severity_score
        )

        return pothole

    def process_yolo_results(self, results) -> List[PotholeInfo]:
        """
        Procesar resultados de YOLO

        Args:
            results: Resultados de predicción de YOLO

        Returns:
            List[PotholeInfo]: Lista de baches procesados
        """
        potholes = []

        # Iterar sobre cada imagen en los resultados
        for result in results:
            if result.masks is None:
                continue

            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            # Procesar cada detección
            for idx, (mask, bbox, conf) in enumerate(zip(masks, boxes, confidences)):
                pothole = self.process_detection(
                    mask=mask,
                    bbox=tuple(bbox),
                    confidence=float(conf),
                    detection_id=idx
                )
                potholes.append(pothole)

        return potholes

    def get_severity_statistics(self, potholes: List[PotholeInfo]) -> Dict:
        """
        Obtener estadísticas de severidad

        Args:
            potholes: Lista de baches procesados

        Returns:
            Dict: Estadísticas de severidad
        """
        if not potholes:
            return {
                'total': 0,
                'by_severity': {
                    'Low': 0,
                    'Medium': 0,
                    'High': 0
                },
                'percentages': {
                    'Low': 0.0,
                    'Medium': 0.0,
                    'High': 0.0
                },
                'average_diameter_mm': 0.0,
                'average_severity_score': 0.0
            }

        total = len(potholes)
        severity_counts = {
            'Low': sum(1 for p in potholes if p.severity == SeverityLevel.LOW),
            'Medium': sum(1 for p in potholes if p.severity == SeverityLevel.MEDIUM),
            'High': sum(1 for p in potholes if p.severity == SeverityLevel.HIGH)
        }

        percentages = {
            k: (v / total * 100) if total > 0 else 0.0
            for k, v in severity_counts.items()
        }

        avg_diameter = np.mean([p.diameter_mm for p in potholes])
        avg_score = np.mean([p.severity_score for p in potholes])

        return {
            'total': total,
            'by_severity': severity_counts,
            'percentages': percentages,
            'average_diameter_mm': float(avg_diameter),
            'average_severity_score': float(avg_score),
            'min_diameter_mm': float(min(p.diameter_mm for p in potholes)),
            'max_diameter_mm': float(max(p.diameter_mm for p in potholes))
        }


def estimate_pixels_per_mm(
    image_height_px: int,
    camera_height_m: float = 3.0,
    camera_fov_degrees: float = 60.0
) -> float:
    """
    Estimar la conversión de píxeles a milímetros

    Basado en geometría de cámara. Asume que los baches están en el suelo.

    Args:
        image_height_px: Altura de la imagen en píxeles
        camera_height_m: Altura de la cámara sobre el suelo en metros
        camera_fov_degrees: Campo de visión vertical de la cámara en grados

    Returns:
        float: Píxeles por milímetro
    """
    # Convertir FOV a radianes
    fov_rad = np.deg2rad(camera_fov_degrees)

    # Calcular el ancho real del campo de visión en el suelo
    ground_width_m = 2 * camera_height_m * np.tan(fov_rad / 2)
    ground_width_mm = ground_width_m * 1000

    # Píxeles por milímetro
    pixels_per_mm = image_height_px / ground_width_mm

    return pixels_per_mm


if __name__ == "__main__":
    # Ejemplo de uso
    print("ASTM D6433-03 Pothole Classifier")
    print("=" * 80)

    # Crear clasificador
    classifier = ASTMPotholeClassifier(pixels_per_mm=2.0)

    print(f"\nUmbrales configurados:")
    print(f"  Low severity: < {classifier.low_threshold_mm} mm")
    print(f"  Medium severity: {classifier.low_threshold_mm}-{classifier.high_threshold_mm} mm")
    print(f"  High severity: > {classifier.high_threshold_mm} mm")

    # Ejemplos de clasificación
    print(f"\nEjemplos de clasificación:")
    test_diameters = [100, 200, 300, 450, 600, 1000]
    for diameter in test_diameters:
        severity, score = classifier.classify_severity(diameter)
        print(f"  Diámetro {diameter} mm -> {severity.value} (Score: {score:.1f})")
