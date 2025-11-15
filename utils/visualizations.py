"""
Utilidades de Visualización para Detección de Baches

Este módulo proporciona funciones para crear visualizaciones de los resultados
de detección, incluyendo mapas de calor, gráficas de distribución, y reportes.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from typing import List, Tuple, Optional
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from scipy.ndimage import gaussian_filter

from .astm_classifier import PotholeInfo, SeverityLevel


# Colores para cada nivel de severidad (BGR para OpenCV)
SEVERITY_COLORS_BGR = {
    SeverityLevel.LOW: (0, 255, 0),      # Verde
    SeverityLevel.MEDIUM: (0, 165, 255),  # Naranja
    SeverityLevel.HIGH: (0, 0, 255),      # Rojo
    SeverityLevel.UNKNOWN: (128, 128, 128) # Gris
}

# Colores para matplotlib (RGB)
SEVERITY_COLORS_RGB = {
    SeverityLevel.LOW: (0, 1, 0),         # Verde
    SeverityLevel.MEDIUM: (1, 0.647, 0),  # Naranja
    SeverityLevel.HIGH: (1, 0, 0),        # Rojo
    SeverityLevel.UNKNOWN: (0.5, 0.5, 0.5) # Gris
}


def draw_detections_on_frame(
    frame: np.ndarray,
    potholes: List[PotholeInfo],
    show_masks: bool = True,
    show_boxes: bool = True,
    show_labels: bool = True,
    alpha: float = 0.4
) -> np.ndarray:
    """
    Dibujar detecciones en un frame

    Args:
        frame: Imagen original
        potholes: Lista de baches detectados
        show_masks: Mostrar máscaras de segmentación
        show_boxes: Mostrar bounding boxes
        show_labels: Mostrar etiquetas con información
        alpha: Transparencia de las máscaras

    Returns:
        np.ndarray: Frame con detecciones dibujadas
    """
    result = frame.copy()
    overlay = frame.copy()

    for pothole in potholes:
        color = SEVERITY_COLORS_BGR[pothole.severity]

        # Dibujar máscara
        if show_masks and pothole.mask is not None:
            # Redimensionar máscara al tamaño del frame si es necesario
            if pothole.mask.shape != frame.shape[:2]:
                mask_resized = cv2.resize(
                    pothole.mask.astype(np.uint8),
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                mask_resized = pothole.mask.astype(np.uint8)

            # Crear overlay de color
            colored_mask = np.zeros_like(frame)
            colored_mask[mask_resized > 0] = color

            # Aplicar con transparencia
            overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)

            # Dibujar contorno de la máscara
            contours, _ = cv2.findContours(
                mask_resized,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, color, 2)

        # Dibujar bounding box
        if show_boxes:
            x1, y1, x2, y2 = map(int, pothole.bbox)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Dibujar etiqueta
        if show_labels:
            x1, y1, x2, y2 = map(int, pothole.bbox)

            # Preparar texto
            label_lines = [
                f"ID: {pothole.id}",
                f"{pothole.severity.value}",
                f"{pothole.diameter_mm:.0f}mm",
                f"{pothole.confidence:.2f}"
            ]

            # Configuración de texto
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            padding = 5

            # Calcular tamaño del fondo
            text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in label_lines]
            max_width = max(w for w, h in text_sizes)
            total_height = sum(h for w, h in text_sizes) + padding * (len(label_lines) + 1)

            # Dibujar fondo semi-transparente
            bg_x1, bg_y1 = x1, y1 - total_height - padding
            bg_x2, bg_y2 = x1 + max_width + 2 * padding, y1

            # Asegurar que el fondo esté dentro de la imagen
            bg_y1 = max(0, bg_y1)

            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 1)

            # Dibujar texto
            y_offset = bg_y1 + padding
            for line in label_lines:
                y_offset += text_sizes[0][1] + padding
                cv2.putText(
                    overlay,
                    line,
                    (bg_x1 + padding, y_offset),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA
                )

    # Combinar con el frame original
    result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)

    return result


def create_heatmap(
    image_shape: Tuple[int, int],
    potholes: List[PotholeInfo],
    sigma: float = 50.0,
    use_severity: bool = True
) -> np.ndarray:
    """
    Crear mapa de calor de densidad de baches

    Args:
        image_shape: (height, width) de la imagen
        potholes: Lista de baches detectados
        sigma: Desviación estándar del filtro gaussiano
        use_severity: Usar severity score para ponderar el mapa de calor

    Returns:
        np.ndarray: Mapa de calor normalizado (0-255)
    """
    height, width = image_shape
    heatmap = np.zeros((height, width), dtype=np.float32)

    for pothole in potholes:
        if pothole.mask is not None:
            # Redimensionar máscara si es necesario
            if pothole.mask.shape != (height, width):
                mask = cv2.resize(
                    pothole.mask.astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_LINEAR
                )
            else:
                mask = pothole.mask.astype(np.float32)

            # Ponderar por severity score si se solicita
            weight = pothole.severity_score / 100.0 if use_severity else 1.0

            # Agregar al mapa de calor
            heatmap += mask * weight

    # Aplicar filtro gaussiano para suavizar
    if sigma > 0:
        heatmap = gaussian_filter(heatmap, sigma=sigma)

    # Normalizar a rango 0-255
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
    else:
        heatmap = heatmap.astype(np.uint8)

    return heatmap


def apply_heatmap_colormap(
    heatmap: np.ndarray,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Aplicar colormap a un mapa de calor

    Args:
        heatmap: Mapa de calor (0-255, uint8)
        colormap: OpenCV colormap

    Returns:
        np.ndarray: Mapa de calor con colores aplicados
    """
    return cv2.applyColorMap(heatmap, colormap)


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.6
) -> np.ndarray:
    """
    Superponer mapa de calor sobre imagen

    Args:
        image: Imagen original
        heatmap: Mapa de calor (debe tener el mismo tamaño que la imagen)
        alpha: Transparencia del mapa de calor

    Returns:
        np.ndarray: Imagen con mapa de calor superpuesto
    """
    # Asegurar que tengan el mismo tamaño
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Superponer
    result = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    return result


def plot_severity_distribution(
    potholes: List[PotholeInfo],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Crear gráfica de distribución de severidad

    Args:
        potholes: Lista de baches
        save_path: Ruta para guardar la figura

    Returns:
        plt.Figure: Figura de matplotlib
    """
    if not potholes:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No detections', ha='center', va='center')
        return fig

    # Contar por severidad
    severity_counts = {
        'Low': sum(1 for p in potholes if p.severity == SeverityLevel.LOW),
        'Medium': sum(1 for p in potholes if p.severity == SeverityLevel.MEDIUM),
        'High': sum(1 for p in potholes if p.severity == SeverityLevel.HIGH)
    }

    # Crear figura
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfica de barras
    colors = [SEVERITY_COLORS_RGB[SeverityLevel.LOW],
              SEVERITY_COLORS_RGB[SeverityLevel.MEDIUM],
              SEVERITY_COLORS_RGB[SeverityLevel.HIGH]]

    axes[0].bar(severity_counts.keys(), severity_counts.values(), color=colors)
    axes[0].set_xlabel('Severity Level', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Pothole Count by Severity', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # Agregar valores en las barras
    for i, (k, v) in enumerate(severity_counts.items()):
        axes[0].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

    # Gráfica de pastel
    axes[1].pie(
        severity_counts.values(),
        labels=severity_counts.keys(),
        colors=colors,
        autopct='%1.1f%%',
        startangle=90
    )
    axes[1].set_title('Severity Distribution', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_diameter_histogram(
    potholes: List[PotholeInfo],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Crear histograma de diámetros

    Args:
        potholes: Lista de baches
        save_path: Ruta para guardar la figura

    Returns:
        plt.Figure: Figura de matplotlib
    """
    if not potholes:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No detections', ha='center', va='center')
        return fig

    diameters = [p.diameter_mm for p in potholes]
    severities = [p.severity for p in potholes]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Crear bins
    bins = np.linspace(0, max(diameters) * 1.1, 30)

    # Separar por severidad
    for severity in [SeverityLevel.LOW, SeverityLevel.MEDIUM, SeverityLevel.HIGH]:
        severity_diameters = [d for d, s in zip(diameters, severities) if s == severity]
        if severity_diameters:
            ax.hist(
                severity_diameters,
                bins=bins,
                alpha=0.7,
                label=severity.value,
                color=SEVERITY_COLORS_RGB[severity],
                edgecolor='black'
            )

    # Líneas de umbral
    ax.axvline(200, color='black', linestyle='--', linewidth=2, label='Low/Medium threshold')
    ax.axvline(450, color='black', linestyle='--', linewidth=2, label='Medium/High threshold')

    ax.set_xlabel('Diameter (mm)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Pothole Diameters', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_interactive_heatmap(
    image: np.ndarray,
    potholes: List[PotholeInfo],
    save_path: Optional[Path] = None
) -> go.Figure:
    """
    Crear mapa de calor interactivo con Plotly

    Args:
        image: Imagen de fondo
        potholes: Lista de baches
        save_path: Ruta para guardar como HTML

    Returns:
        go.Figure: Figura de Plotly
    """
    height, width = image.shape[:2]

    # Crear matriz de densidad
    density = np.zeros((height, width))

    for pothole in potholes:
        if pothole.mask is not None:
            if pothole.mask.shape != (height, width):
                mask = cv2.resize(
                    pothole.mask.astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_LINEAR
                )
            else:
                mask = pothole.mask.astype(np.float32)

            density += mask * (pothole.severity_score / 100.0)

    # Suavizar
    density = gaussian_filter(density, sigma=30)

    # Crear figura
    fig = go.Figure()

    # Agregar imagen de fondo
    fig.add_trace(go.Image(z=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

    # Agregar mapa de calor
    fig.add_trace(go.Heatmap(
        z=density,
        colorscale='Hot',
        opacity=0.6,
        showscale=True,
        colorbar=dict(title="Severity Density")
    ))

    fig.update_layout(
        title='Interactive Pothole Heatmap',
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        height=800,
        width=1200
    )

    if save_path:
        fig.write_html(str(save_path))

    return fig


def generate_summary_statistics(potholes: List[PotholeInfo]) -> dict:
    """
    Generar estadísticas resumidas

    Args:
        potholes: Lista de baches

    Returns:
        dict: Estadísticas resumidas
    """
    if not potholes:
        return {
            'total_potholes': 0,
            'severity_distribution': {'Low': 0, 'Medium': 0, 'High': 0},
            'average_diameter_mm': 0,
            'average_confidence': 0,
            'average_severity_score': 0
        }

    return {
        'total_potholes': len(potholes),
        'severity_distribution': {
            'Low': sum(1 for p in potholes if p.severity == SeverityLevel.LOW),
            'Medium': sum(1 for p in potholes if p.severity == SeverityLevel.MEDIUM),
            'High': sum(1 for p in potholes if p.severity == SeverityLevel.HIGH)
        },
        'average_diameter_mm': np.mean([p.diameter_mm for p in potholes]),
        'min_diameter_mm': min(p.diameter_mm for p in potholes),
        'max_diameter_mm': max(p.diameter_mm for p in potholes),
        'average_confidence': np.mean([p.confidence for p in potholes]),
        'average_severity_score': np.mean([p.severity_score for p in potholes])
    }
