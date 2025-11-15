"""
Script para procesamiento de videos por lotes

Este script permite procesar uno o múltiples videos desde la línea de comandos
y generar reportes automáticamente.
"""

import argparse
import sys
from pathlib import Path
import cv2
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Agregar directorio raíz al path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ultralytics import YOLO
from utils import (
    ASTMPotholeClassifier,
    draw_detections_on_frame,
    create_heatmap,
    apply_heatmap_colormap,
    overlay_heatmap_on_image,
    generate_summary_statistics
)


def process_single_video(
    video_path: Path,
    model: YOLO,
    classifier: ASTMPotholeClassifier,
    output_dir: Path,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    save_video: bool = True,
    save_frames: bool = False,
    frame_interval: int = 30
) -> dict:
    """
    Procesar un solo video

    Args:
        video_path: Ruta al video de entrada
        model: Modelo YOLO cargado
        classifier: Clasificador ASTM
        output_dir: Directorio de salida
        conf_threshold: Umbral de confianza
        iou_threshold: Umbral de IoU
        save_video: Guardar video procesado
        save_frames: Guardar frames individuales
        frame_interval: Intervalo de frames a guardar

    Returns:
        dict con estadísticas del procesamiento
    """
    print(f"\n{'='*80}")
    print(f"Procesando: {video_path.name}")
    print(f"{'='*80}\n")

    # Abrir video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return None

    # Obtener propiedades del video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Propiedades del video:")
    print(f"  Resolución: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total de frames: {total_frames}")
    print(f"  Duración: {total_frames/fps:.2f} segundos\n")

    # Crear directorio de salida para este video
    video_output_dir = output_dir / video_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)

    if save_frames:
        frames_dir = video_output_dir / 'frames'
        frames_dir.mkdir(exist_ok=True)

    # Preparar video de salida
    if save_video:
        output_video_path = video_output_dir / f"{video_path.stem}_processed.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    # Inicializar acumuladores
    all_potholes = []
    heatmap_accumulator = np.zeros((height, width), dtype=np.float32)
    frame_count = 0

    # Procesar video
    pbar = tqdm(total=total_frames, desc="Procesando frames")

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

        # Clasificar detecciones
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

        # Guardar video procesado
        if save_video:
            out.write(annotated_frame)

        # Guardar frames individuales
        if save_frames and frame_count % frame_interval == 0:
            frame_path = frames_dir / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_path), annotated_frame)

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
        pbar.update(1)

    pbar.close()
    cap.release()

    if save_video:
        out.release()
        print(f"\n✓ Video procesado guardado: {output_video_path}")

    # Normalizar y guardar mapa de calor
    if heatmap_accumulator.max() > 0:
        heatmap_normalized = (heatmap_accumulator / heatmap_accumulator.max() * 255).astype(np.uint8)
    else:
        heatmap_normalized = heatmap_accumulator.astype(np.uint8)

    heatmap_colored = apply_heatmap_colormap(heatmap_normalized, cv2.COLORMAP_JET)
    heatmap_path = video_output_dir / 'heatmap.jpg'
    cv2.imwrite(str(heatmap_path), heatmap_colored)
    print(f"✓ Mapa de calor guardado: {heatmap_path}")

    # Generar estadísticas
    stats = generate_summary_statistics(all_potholes)

    # Guardar reporte JSON
    report = {
        'video_info': {
            'filename': video_path.name,
            'resolution': f"{width}x{height}",
            'fps': fps,
            'total_frames': total_frames,
            'duration_seconds': total_frames / fps
        },
        'processing_info': {
            'timestamp': datetime.now().isoformat(),
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'pixels_per_mm': classifier.pixels_per_mm
        },
        'statistics': stats,
        'detections': [p.to_dict() for p in all_potholes]
    }

    report_path = video_output_dir / 'report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✓ Reporte JSON guardado: {report_path}")

    # Mostrar resumen
    print(f"\n{'='*80}")
    print("RESUMEN")
    print(f"{'='*80}")
    print(f"Total de baches detectados: {stats['total_potholes']}")
    print(f"\nDistribución por severidad:")
    print(f"  Low:    {stats['severity_distribution']['Low']:>4} ({stats['severity_distribution']['Low']/max(stats['total_potholes'],1)*100:.1f}%)")
    print(f"  Medium: {stats['severity_distribution']['Medium']:>4} ({stats['severity_distribution']['Medium']/max(stats['total_potholes'],1)*100:.1f}%)")
    print(f"  High:   {stats['severity_distribution']['High']:>4} ({stats['severity_distribution']['High']/max(stats['total_potholes'],1)*100:.1f}%)")

    if stats['total_potholes'] > 0:
        print(f"\nDiámetro promedio: {stats['average_diameter_mm']:.1f} mm")
        print(f"Diámetro mínimo: {stats['min_diameter_mm']:.1f} mm")
        print(f"Diámetro máximo: {stats['max_diameter_mm']:.1f} mm")

    print(f"{'='*80}\n")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Procesar videos para detección y clasificación de baches"
    )

    parser.add_argument(
        'input',
        type=str,
        help='Ruta al video o directorio con videos'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='models/exports/yolo11x_pothole_best.pt',
        help='Ruta al modelo YOLO (default: models/exports/yolo11x_pothole_best.pt)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Directorio de salida (default: output)'
    )

    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Umbral de confianza (default: 0.25)'
    )

    parser.add_argument(
        '--iou',
        type=float,
        default=0.7,
        help='Umbral de IoU (default: 0.7)'
    )

    parser.add_argument(
        '--pixels-per-mm',
        type=float,
        default=1.0,
        help='Factor de conversión píxeles a mm (default: 1.0)'
    )

    parser.add_argument(
        '--no-video',
        action='store_true',
        help='No guardar video procesado'
    )

    parser.add_argument(
        '--save-frames',
        action='store_true',
        help='Guardar frames individuales'
    )

    parser.add_argument(
        '--frame-interval',
        type=int,
        default=30,
        help='Intervalo de frames a guardar (default: 30)'
    )

    args = parser.parse_args()

    # Validar input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} no existe")
        return

    # Crear directorio de salida
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar modelo
    print(f"Cargando modelo: {args.model}")
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Modelo no encontrado en {model_path}")
        return

    model = YOLO(str(model_path))
    print("✓ Modelo cargado\n")

    # Crear clasificador
    classifier = ASTMPotholeClassifier(pixels_per_mm=args.pixels_per_mm)

    # Obtener lista de videos
    if input_path.is_file():
        video_files = [input_path]
    else:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(input_path.glob(f'*{ext}'))
            video_files.extend(input_path.glob(f'*{ext.upper()}'))

    if not video_files:
        print(f"No se encontraron videos en {input_path}")
        return

    print(f"Encontrados {len(video_files)} video(s) para procesar\n")

    # Procesar cada video
    all_stats = []
    for video_file in video_files:
        stats = process_single_video(
            video_file,
            model,
            classifier,
            output_dir,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            save_video=not args.no_video,
            save_frames=args.save_frames,
            frame_interval=args.frame_interval
        )

        if stats:
            all_stats.append({
                'filename': video_file.name,
                'stats': stats
            })

    # Resumen global
    if len(all_stats) > 1:
        print(f"\n{'='*80}")
        print("RESUMEN GLOBAL")
        print(f"{'='*80}")
        print(f"Videos procesados: {len(all_stats)}")

        total_potholes = sum(s['stats']['total_potholes'] for s in all_stats)
        print(f"Total de baches detectados: {total_potholes}")

        total_low = sum(s['stats']['severity_distribution']['Low'] for s in all_stats)
        total_medium = sum(s['stats']['severity_distribution']['Medium'] for s in all_stats)
        total_high = sum(s['stats']['severity_distribution']['High'] for s in all_stats)

        print(f"\nDistribución global:")
        print(f"  Low:    {total_low:>4} ({total_low/max(total_potholes,1)*100:.1f}%)")
        print(f"  Medium: {total_medium:>4} ({total_medium/max(total_potholes,1)*100:.1f}%)")
        print(f"  High:   {total_high:>4} ({total_high/max(total_potholes,1)*100:.1f}%)")
        print(f"{'='*80}\n")

    print(f"Resultados guardados en: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
