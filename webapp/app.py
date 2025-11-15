"""
Aplicación Web para Detección y Clasificación de Baches

Esta aplicación permite:
1. Cargar y procesar videos
2. Detectar baches usando YOLOv11
3. Clasificar según norma ASTM D6433-03
4. Generar reportes y visualizaciones
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import json
from pathlib import Path
import sys
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

# Agregar el directorio raíz al path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ultralytics import YOLO
from utils import (
    ASTMPotholeClassifier,
    PotholeInfo,
    draw_detections_on_frame,
    create_heatmap,
    apply_heatmap_colormap,
    overlay_heatmap_on_image,
    plot_severity_distribution,
    plot_diameter_histogram,
    generate_summary_statistics
)

# Configuración de la página
st.set_page_config(
    page_title="Detección de Baches - ASTM D6433-03",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .reportview-container .markdown-text-container {
        font-family: monospace;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path: str):
    """Cargar modelo YOLO (cached)"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        return None


def process_video(
    video_path: str,
    model: YOLO,
    classifier: ASTMPotholeClassifier,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    progress_bar=None
) -> dict:
    """
    Procesar video completo

    Returns:
        dict con resultados y frames procesados
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    all_potholes = []
    processed_frames = []
    frame_count = 0

    # Crear acumulador de mapa de calor
    heatmap_accumulator = np.zeros((height, width), dtype=np.float32)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Ejecutar detección
        results = model.predict(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        # Procesar detecciones
        frame_potholes = classifier.process_yolo_results(results)
        all_potholes.extend(frame_potholes)

        # Dibujar detecciones
        annotated_frame = draw_detections_on_frame(
            frame,
            frame_potholes,
            show_masks=True,
            show_boxes=True,
            show_labels=True
        )

        # Guardar frame procesado (cada 10 frames para ahorrar memoria)
        if frame_count % 10 == 0:
            processed_frames.append({
                'frame_number': frame_count,
                'timestamp': frame_count / fps,
                'frame': annotated_frame,
                'potholes': frame_potholes
            })

        # Actualizar mapa de calor
        for pothole in frame_potholes:
            if pothole.mask is not None:
                if pothole.mask.shape != (height, width):
                    mask = cv2.resize(
                        pothole.mask.astype(np.float32),
                        (width, height),
                        interpolation=cv2.INTER_LINEAR
                    )
                else:
                    mask = pothole.mask.astype(np.float32)

                heatmap_accumulator += mask * (pothole.severity_score / 100.0)

        frame_count += 1

        # Actualizar barra de progreso
        if progress_bar:
            progress_bar.progress(frame_count / total_frames)

    cap.release()

    # Normalizar mapa de calor
    if heatmap_accumulator.max() > 0:
        heatmap_normalized = (heatmap_accumulator / heatmap_accumulator.max() * 255).astype(np.uint8)
    else:
        heatmap_normalized = heatmap_accumulator.astype(np.uint8)

    return {
        'total_frames': total_frames,
        'fps': fps,
        'width': width,
        'height': height,
        'all_potholes': all_potholes,
        'processed_frames': processed_frames,
        'heatmap': heatmap_normalized
    }


def create_video_from_frames(frames_data: list, fps: int, output_path: str):
    """Crear video a partir de frames procesados"""
    if not frames_data:
        return None

    height, width = frames_data[0]['frame'].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_data in frames_data:
        out.write(frame_data['frame'])

    out.release()
    return output_path


def generate_pdf_report(results: dict, stats: dict, output_path: str):
    """Generar reporte PDF"""
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    doc = SimpleDocTemplate(output_path, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()

    # Título
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    story.append(Paragraph("Reporte de Detección de Baches", title_style))
    story.append(Paragraph(f"Según Norma ASTM D6433-03", styles['Heading2']))
    story.append(Spacer(1, 0.5*inch))

    # Información general
    story.append(Paragraph("Información General", styles['Heading2']))
    info_data = [
        ['Fecha de análisis:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Total de frames:', str(results['total_frames'])],
        ['FPS:', str(results['fps'])],
        ['Resolución:', f"{results['width']}x{results['height']}"],
        ['Total de baches detectados:', str(stats['total_potholes'])],
    ]
    info_table = Table(info_data, colWidths=[3*inch, 3*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.grey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.5*inch))

    # Distribución de severidad
    story.append(Paragraph("Distribución por Severidad (ASTM D6433-03)", styles['Heading2']))
    severity_data = [
        ['Nivel de Severidad', 'Cantidad', 'Porcentaje', 'Criterio (Diámetro)'],
        ['Low (Bajo)',
         str(stats['severity_distribution']['Low']),
         f"{stats['severity_distribution']['Low']/stats['total_potholes']*100:.1f}%" if stats['total_potholes'] > 0 else '0%',
         '< 200 mm'],
        ['Medium (Medio)',
         str(stats['severity_distribution']['Medium']),
         f"{stats['severity_distribution']['Medium']/stats['total_potholes']*100:.1f}%" if stats['total_potholes'] > 0 else '0%',
         '200-450 mm'],
        ['High (Alto)',
         str(stats['severity_distribution']['High']),
         f"{stats['severity_distribution']['High']/stats['total_potholes']*100:.1f}%" if stats['total_potholes'] > 0 else '0%',
         '> 450 mm'],
    ]
    severity_table = Table(severity_data, colWidths=[1.5*inch, 1.2*inch, 1.3*inch, 2*inch])
    severity_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(severity_table)
    story.append(Spacer(1, 0.5*inch))

    # Estadísticas de diámetros
    if stats['total_potholes'] > 0:
        story.append(Paragraph("Estadísticas de Diámetros", styles['Heading2']))
        diameter_data = [
            ['Métrica', 'Valor (mm)'],
            ['Diámetro promedio', f"{stats['average_diameter_mm']:.1f}"],
            ['Diámetro mínimo', f"{stats['min_diameter_mm']:.1f}"],
            ['Diámetro máximo', f"{stats['max_diameter_mm']:.1f}"],
            ['Confianza promedio', f"{stats['average_confidence']:.2%}"],
        ]
        diameter_table = Table(diameter_data, colWidths=[3*inch, 3*inch])
        diameter_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.grey),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(diameter_table)

    # Construir PDF
    doc.build(story)
    return output_path


def main():
    """Función principal de la aplicación"""

    # Sidebar
    st.sidebar.title("🛣️ Detección de Baches")
    st.sidebar.markdown("### ASTM D6433-03")

    # Configuración
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Configuración")

    model_path = st.sidebar.text_input(
        "Ruta del modelo",
        value="models/exports/yolo11x_pothole_best.pt",
        help="Ruta al modelo YOLOv11 entrenado"
    )

    conf_threshold = st.sidebar.slider(
        "Confianza mínima",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05,
        help="Umbral de confianza para detecciones"
    )

    iou_threshold = st.sidebar.slider(
        "IoU threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.7,
        step=0.05,
        help="Umbral de IoU para NMS"
    )

    pixels_per_mm = st.sidebar.number_input(
        "Píxeles por mm",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Factor de calibración para conversión píxeles a mm"
    )

    # Título principal
    st.title("🛣️ Sistema de Detección y Clasificación de Baches")
    st.markdown("### Basado en YOLOv11 y Norma ASTM D6433-03")

    # Tabs principales
    tab1, tab2, tab3 = st.tabs(["📹 Procesamiento", "📊 Análisis", "📄 Reportes"])

    # Tab 1: Procesamiento de Video
    with tab1:
        st.header("Procesamiento de Video")

        uploaded_file = st.file_uploader(
            "Cargar video",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Formatos soportados: MP4, AVI, MOV, MKV"
        )

        if uploaded_file is not None:
            # Guardar video temporalmente
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path = tfile.name

            # Mostrar video original
            st.subheader("Video Original")
            st.video(uploaded_file)

            # Botón de procesamiento
            if st.button("🚀 Procesar Video", type="primary"):
                # Cargar modelo
                with st.spinner("Cargando modelo..."):
                    model = load_model(model_path)

                if model is None:
                    st.error("No se pudo cargar el modelo. Verifica la ruta.")
                    return

                # Crear clasificador
                classifier = ASTMPotholeClassifier(pixels_per_mm=pixels_per_mm)

                # Procesar video
                st.info("Procesando video... Esto puede tomar unos minutos.")
                progress_bar = st.progress(0)

                results = process_video(
                    video_path,
                    model,
                    classifier,
                    conf_threshold,
                    iou_threshold,
                    progress_bar
                )

                # Guardar resultados en session state
                st.session_state['results'] = results
                st.session_state['stats'] = generate_summary_statistics(results['all_potholes'])

                st.success("✅ Video procesado exitosamente!")

                # Mostrar video procesado (primeros frames)
                st.subheader("Video Procesado (muestra)")
                if results['processed_frames']:
                    # Crear video temporal con frames procesados
                    output_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    create_video_from_frames(
                        results['processed_frames'],
                        results['fps'],
                        output_video.name
                    )
                    st.video(output_video.name)

    # Tab 2: Análisis
    with tab2:
        st.header("Análisis de Resultados")

        if 'results' in st.session_state and 'stats' in st.session_state:
            results = st.session_state['results']
            stats = st.session_state['stats']

            # Métricas principales
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total de Baches", stats['total_potholes'])

            with col2:
                st.metric("Severidad Baja", stats['severity_distribution']['Low'])

            with col3:
                st.metric("Severidad Media", stats['severity_distribution']['Medium'])

            with col4:
                st.metric("Severidad Alta", stats['severity_distribution']['High'])

            # Mapa de calor
            st.subheader("🔥 Mapa de Calor de Densidad")
            heatmap_colored = apply_heatmap_colormap(results['heatmap'], cv2.COLORMAP_JET)
            st.image(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB), use_container_width=True)

            # Gráficas
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Distribución por Severidad")
                if stats['total_potholes'] > 0:
                    fig = px.pie(
                        values=list(stats['severity_distribution'].values()),
                        names=list(stats['severity_distribution'].keys()),
                        color=list(stats['severity_distribution'].keys()),
                        color_discrete_map={
                            'Low': 'green',
                            'Medium': 'orange',
                            'High': 'red'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("📏 Distribución de Diámetros")
                if results['all_potholes']:
                    diameters = [p.diameter_mm for p in results['all_potholes']]
                    severities = [p.severity.value for p in results['all_potholes']]

                    df = pd.DataFrame({
                        'Diámetro (mm)': diameters,
                        'Severidad': severities
                    })

                    fig = px.histogram(
                        df,
                        x='Diámetro (mm)',
                        color='Severidad',
                        color_discrete_map={
                            'Low': 'green',
                            'Medium': 'orange',
                            'High': 'red'
                        },
                        nbins=30
                    )
                    fig.add_vline(x=200, line_dash="dash", line_color="black", annotation_text="200mm")
                    fig.add_vline(x=450, line_dash="dash", line_color="black", annotation_text="450mm")
                    st.plotly_chart(fig, use_container_width=True)

            # Tabla de detecciones
            st.subheader("📋 Tabla de Detecciones")
            if results['all_potholes']:
                detections_data = []
                for i, pothole in enumerate(results['all_potholes'][:100]):  # Limitar a 100
                    detections_data.append({
                        'ID': i,
                        'Diámetro (mm)': f"{pothole.diameter_mm:.1f}",
                        'Severidad': pothole.severity.value,
                        'Score': f"{pothole.severity_score:.1f}",
                        'Confianza': f"{pothole.confidence:.2%}"
                    })

                df = pd.DataFrame(detections_data)
                st.dataframe(df, use_container_width=True)

        else:
            st.info("👆 Procesa un video primero en la pestaña 'Procesamiento'")

    # Tab 3: Reportes
    with tab3:
        st.header("Generación de Reportes")

        if 'results' in st.session_state and 'stats' in st.session_state:
            results = st.session_state['results']
            stats = st.session_state['stats']

            col1, col2 = st.columns(2)

            with col1:
                # Reporte JSON
                st.subheader("📄 Exportar JSON")
                if st.button("Generar JSON"):
                    report_data = {
                        'metadata': {
                            'timestamp': datetime.now().isoformat(),
                            'total_frames': results['total_frames'],
                            'fps': results['fps'],
                            'resolution': f"{results['width']}x{results['height']}"
                        },
                        'statistics': stats,
                        'detections': [p.to_dict() for p in results['all_potholes'][:1000]]
                    }

                    json_str = json.dumps(report_data, indent=2)
                    st.download_button(
                        label="⬇️ Descargar JSON",
                        data=json_str,
                        file_name=f"reporte_baches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

            with col2:
                # Reporte PDF
                st.subheader("📄 Exportar PDF")
                if st.button("Generar PDF"):
                    with st.spinner("Generando PDF..."):
                        pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf').name
                        generate_pdf_report(results, stats, pdf_path)

                        with open(pdf_path, 'rb') as f:
                            st.download_button(
                                label="⬇️ Descargar PDF",
                                data=f.read(),
                                file_name=f"reporte_baches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf"
                            )

            # Exportar mapa de calor
            st.subheader("🔥 Exportar Mapa de Calor")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Descargar Mapa de Calor"):
                    heatmap_colored = apply_heatmap_colormap(results['heatmap'], cv2.COLORMAP_JET)
                    is_success, buffer = cv2.imencode(".png", heatmap_colored)
                    if is_success:
                        st.download_button(
                            label="⬇️ Descargar PNG",
                            data=buffer.tobytes(),
                            file_name=f"heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png"
                        )

        else:
            st.info("👆 Procesa un video primero en la pestaña 'Procesamiento'")


if __name__ == "__main__":
    main()
