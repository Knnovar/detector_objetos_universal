import argparse
import os
import cv2
import pandas as pd
from ultralytics import YOLO
from datetime import datetime

# Funções save_detections_to_file, process_image, process_video
# (O conteúdo interno dessas funções permanece o mesmo do script anterior,
# apenas a forma como são chamadas na função 'main' será alterada)
def save_detections_to_file(detections_list, output_filename_base):
    if not detections_list:
        print("Nenhum objeto foi detectado para salvar.")
        return
    df = pd.DataFrame(detections_list)
    csv_filename = f"{output_filename_base}.csv"
    excel_filename = f"{output_filename_base}.xlsx"
    path = "./Extracao_Dados/"
    df.to_csv(path+csv_filename, index=False, encoding='utf-8')
    print(f"Dados de detecção salvos em: {csv_filename}")
    df.to_excel(path+excel_filename, index=False)
    print(f"Dados de detecção salvos em: {excel_filename}")

def process_image(model, image_path, output_filename_base):
    # (código idêntico ao anterior)
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Erro: Não foi possível ler a imagem em '{image_path}'")
        return
    detections_data = []
    results = model(frame)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = round(float(box.conf[0]), 2)
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            detections_data.append({'source_file': os.path.basename(image_path),'frame': 'N/A','class_name': class_name,'confidence': confidence,'x1': x1,'y1': y1,'x2': x2,'y2': y2})
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            label = f'{class_name} {confidence}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.imshow("Detecção de Objetos em Imagem", frame)
    output_image_filename = f"{output_filename_base}_detectado.jpg"
    cv2.imwrite(output_image_filename, frame)
    print(f"Imagem resultante salva como: {output_image_filename}")
    save_detections_to_file(detections_data, output_filename_base)
    print("Pressione qualquer tecla na janela da imagem para fechar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(model, video_path, output_filename_base):
    # (código idêntico ao anterior)
    cap_source = 0 if video_path == '0' else video_path
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir a fonte de vídeo: '{video_path}'")
        return
    all_detections = []
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        results = model(frame, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = round(float(box.conf[0]), 2)
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                all_detections.append({'source_file': os.path.basename(video_path) if video_path != '0' else 'webcam','frame': frame_count,'class_name': class_name,'confidence': confidence,'x1': x1,'y1': y1,'x2': x2,'y2': y2})
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                label = f'{class_name} {confidence}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
        cv2.imshow("Detecção de Objetos (Pressione 'q' para sair)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    save_detections_to_file(all_detections, output_filename_base)


def main(model_path, source_path):
    """
    Função principal que carrega um modelo específico e processa uma fonte de mídia.
    """
    # Carrega o modelo especificado
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Erro ao carregar o modelo de '{model_path}': {e}")
        print("Verifique se o caminho para o arquivo .pt está correto.")
        return

    # Define o nome base para os arquivos de saída
    if source_path == '0':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename_base = f"webcam_detections_{timestamp}"
    else:
        output_filename_base = os.path.splitext(os.path.basename(source_path))[0] + "_detections"

    # Define extensões válidas
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    file_extension = os.path.splitext(source_path)[1].lower()

    # Decide qual função chamar
    if source_path == '0':
        process_video(model, '0', output_filename_base)
    elif file_extension in image_extensions:
        process_image(model, source_path, output_filename_base)
    elif file_extension in video_extensions:
        process_video(model, source_path, output_filename_base)
    else:
        print(f"Erro: Formato de arquivo não suportado ou caminho inválido: '{source_path}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script de Detecção Universal com YOLOv8")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help="Caminho para o arquivo do modelo treinado (.pt)."
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help="Caminho para a imagem, vídeo ou '0' para a webcam."
    )
    args = parser.parse_args()
    main(args.model, args.source)