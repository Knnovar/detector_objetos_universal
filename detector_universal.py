import argparse
import os
import cv2
import pandas as pd
from ultralytics import YOLO
from datetime import datetime, timedelta
import numpy as np
import mss
from collections import Counter, defaultdict # <<< NOVO: Importa defaultdict >>>

def save_detections_to_file(detections_list, output_filename_base):
    """
    Salva a lista de detecções em arquivos CSV e Excel.
    Agora inclui a contagem persistente.
    """
    if not detections_list:
        print("Nenhum objeto foi detectado para salvar.")
        return
    
    # <<< NOVO: Define a ordem das colunas com a nova contagem >>>
    columns_order = [
        'source_file', 'frame', 'timestamp', 'track_id', 'class_name', 
        'class_count_in_frame', 'persistent_total_count', 
        'confidence', 'x1', 'y1', 'x2', 'y2'
    ]
    
    df = pd.DataFrame(detections_list)
    
    existing_cols = [col for col in columns_order if col in df.columns]
    other_cols = [col for col in df.columns if col not in existing_cols]
    df = df[existing_cols + other_cols]

    path = './Extracao_Dados/'
    csv_filename = f"{output_filename_base}.csv"
    excel_filename = f"{output_filename_base}.xlsx"
    df.to_csv(path+csv_filename, index=False, encoding='utf-8')
    print(f"Dados de detecção salvos em: {csv_filename}")
    df.to_excel(path+excel_filename, index=False)
    print(f"Dados de detecção salvos em: {excel_filename}")


def draw_counters(frame, counts):
    """
    Desenha o painel de contagem. Esta função não muda, 
    ela apenas desenha qualquer Counter que receber.
    """
    y_offset = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (255, 255, 255) # Branco
    bg_color = (0, 0, 0) # Preto
    thickness = 2
    
    # <<< ALTERADO: O título agora reflete a contagem persistente >>>
    title_text = "Contagem Persistente Total"
    cv2.putText(frame, title_text, (10, y_offset), font, font_scale, font_color, thickness, cv2.LINE_AA)
    y_offset += 40
    
    for class_name, count in sorted(counts.items()):
        text = f"{class_name}: {count}"
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(frame, (5, y_offset - text_height - 5), (15 + text_width, y_offset + 5), bg_color, -1)
        cv2.putText(frame, text, (10, y_offset), font, font_scale, font_color, thickness, cv2.LINE_AA)
        y_offset += 30


def process_image(model, image_path, output_filename_base):
    """
    Processa uma imagem estática. O rastreamento não se aplica,
    então a contagem persistente será igual à contagem do frame.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Erro: Não foi possível ler a imagem em '{image_path}'")
        return

    detections_data = []
    class_names_for_counter = []
    
    # <<< ALTERADO: Usa model.track() para consistência >>>
    # Em imagens, model.track() age como model()
    results = model.track(frame, persist=False, verbose=False) 

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = round(float(box.conf[0]), 2)
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            
            # Não há track_id para imagens estáticas
            track_id = 'N/A'
            
            detections_data.append({
                'source_file': os.path.basename(image_path), 'frame': 'N/A', 'timestamp': 'N/A',
                'track_id': track_id, # <<< NOVO >>>
                'class_name': class_name, 'confidence': confidence,
                'x1': x1,'y1': y1,'x2': x2,'y2': y2
            })
            
            class_names_for_counter.append(class_name)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            label = f'{class_name} {confidence}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    counts = Counter(class_names_for_counter)
    for data_dict in detections_data:
        class_name = data_dict['class_name']
        count = counts[class_name]
        data_dict['class_count_in_frame'] = count
        data_dict['persistent_total_count'] = count # Em imagens, é o mesmo

    draw_counters(frame, counts)

    cv2.imshow("Detecção de Objetos em Imagem", frame)
    output_image_filename = f"{output_filename_base}_detectado.jpg"
    cv2.imwrite(output_image_filename, frame)
    print(f"Imagem resultante salva como: {output_image_filename}")
    
    save_detections_to_file(detections_data, output_filename_base)

    print("Pressione qualquer tecla na janela da imagem para fechar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_continuous_stream(model, frame_iterator, output_filename_base, args):
    """
    Processa streams (vídeo, webcam, tela) usando RASTREAMENTO.
    """
    all_detections = [] 
    frame_count = 0
    previous_frame_counts = Counter() # Para o log-on-change
    
    # <<< NOVO: Dicionário para rastrear IDs únicos por classe >>>
    persistent_track_ids = defaultdict(set)

    for frame, timestamp_sec, source_name in frame_iterator:
        frame_count += 1
        
        # <<< ALTERAÇÃO PRINCIPAL: Usa model.track() >>>
        # persist=True diz ao tracker para lembrar dos IDs entre os frames
        results = model.track(frame, persist=True, verbose=False)
        
        current_frame_detections = [] 
        current_frame_class_names = []
        
        td = timedelta(seconds=timestamp_sec)
        formatted_timestamp = f"{td.seconds // 3600:02}:{(td.seconds // 60) % 60:02}:{td.seconds % 60:02}.{td.microseconds // 1000:03}"

        # --- Passada 1: Coletar detecções do frame ---
        for r in results:
            boxes = r.boxes
            if boxes is None: # Se o tracker não retornar nada
                continue
                
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence < args.min_confidence:
                    continue
                
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                current_frame_class_names.append(class_name)
                
                # <<< NOVO: Obtém o ID de Rastreamento >>>
                track_id = 'N/A'
                if box.id is not None:
                    track_id = int(box.id[0])
                    # Adiciona o ID ao set daquela classe
                    persistent_track_ids[class_name].add(track_id)
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                current_frame_detections.append({
                    'source_file': source_name, 'frame': frame_count, 'timestamp': formatted_timestamp,
                    'track_id': track_id, # <<< NOVO >>>
                    'class_name': class_name, 'confidence': round(confidence, 2),
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                })

                # Desenha no frame (agora com o ID)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'ID: {track_id} - {class_name} {round(confidence, 2)}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # --- Passada 2: Calcular contagens e adicionar ao log ---
        
        # Contagem por frame (ex: 2 carros neste frame)
        current_frame_counts = Counter(current_frame_class_names)
        
        # <<< NOVO: Contagem persistente (ex: 30 carros no total) >>>
        persistent_total_counts = Counter(
            {cls_name: len(ids) for cls_name, ids in persistent_track_ids.items()}
        )
        
        for data_dict in current_frame_detections:
            class_name = data_dict['class_name']
            data_dict['class_count_in_frame'] = current_frame_counts[class_name]
            data_dict['persistent_total_count'] = persistent_total_counts.get(class_name, 0)

        # --- Lógica de decisão para salvar ---
        should_log = False
        if args.log_on_change:
            # A lógica de "mudança" agora se baseia na contagem *do frame*
            if current_frame_counts != previous_frame_counts:
                should_log = True
                previous_frame_counts = current_frame_counts
        elif args.log_interval > 0:
            if frame_count % args.log_interval == 0:
                should_log = True
        else:
            should_log = True

        if should_log and current_frame_detections:
            all_detections.extend(current_frame_detections) 
            
        # <<< ALTERADO: Desenha a contagem PERSISTENTE >>>
        draw_counters(frame, persistent_total_counts)
            
        cv2.imshow("Detecção com Rastreamento (Pressione 'q' para sair)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            
    cv2.destroyAllWindows()
    save_detections_to_file(all_detections, output_filename_base)

# ... (Funções video_frame_iterator, screen_frame_iterator, main, e o parser
#      permanecem exatamente os mesmos da versão anterior) ...
def video_frame_iterator(video_path):
    cap_source = 0 if video_path == '0' else video_path
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise IOError(f"Não foi possível abrir a fonte de vídeo: '{video_path}'")
    is_webcam = (video_path == '0')
    source_name = "webcam" if is_webcam else os.path.basename(video_path)
    start_time = datetime.now() if is_webcam else None
    while True:
        success, frame = cap.read()
        if not success:
            break
        if is_webcam:
            timestamp_sec = (datetime.now() - start_time).total_seconds()
        else:
            timestamp_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp_sec = timestamp_msec / 1000.0
        yield frame, timestamp_sec, source_name
    cap.release()

def screen_frame_iterator():
    start_time = datetime.now()
    source_name = "screen"
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        while True:
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            timestamp_sec = (datetime.now() - start_time).total_seconds()
            yield frame, timestamp_sec, source_name

def main(args):
    if args.log_interval > 0 and args.log_on_change:
        raise ValueError("As opções --log-interval e --log-on-change são mutuamente exclusivas.")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Erro ao carregar o modelo de '{args.model}': {e}")
        return
    source_path = args.source
    source_lower = source_path.lower()
    if source_lower in ['screen', '0']:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_name = "screen" if source_lower == 'screen' else "webcam"
        output_filename_base = f"{source_name}_detections_{timestamp}"
    else:
        output_filename_base = os.path.splitext(os.path.basename(source_path))[0] + "_detections"
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    file_extension = os.path.splitext(source_lower)[1]
    try:
        if file_extension in image_extensions:
            process_image(model, source_path, output_filename_base)
        elif file_extension in video_extensions or source_lower == '0':
            iterator = video_frame_iterator(source_path)
            process_continuous_stream(model, iterator, output_filename_base, args)
        elif source_lower == 'screen':
            iterator = screen_frame_iterator()
            process_continuous_stream(model, iterator, output_filename_base, args)
        else:
            print(f"Erro: Fonte de mídia não suportada: '{source_path}'")
    except (IOError, Exception) as e:
        print(f"Ocorreu um erro ao processar a fonte: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script de Detecção Universal com YOLOv8 e Otimização de Log")
    parser.add_argument('--model', type=str, required=True, help="Caminho para o arquivo do modelo treinado (.pt).")
    parser.add_argument('--source', type=str, required=True, help="Caminho para imagem, vídeo, '0' para webcam, ou 'screen' para tela.")
    parser.add_argument('--min-confidence', type=float, default=0.25, help="Confiança mínima para registrar uma detecção (valor entre 0.0 e 1.0). Default: 0.25")
    parser.add_argument('--log-interval', type=int, default=0, help="Intervalo de frames para salvar os dados. Ex: 15 para salvar a cada 15 frames. Default: 0 (salva todos).")
    parser.add_argument('--log-on-change', action='store_true', help="Se especificado, salva os dados apenas quando o número de objetos detectados por classe mudar.")
    
    args = parser.parse_args()
    main(args)