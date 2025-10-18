import argparse
import os
from ultralytics import YOLO
import torch

def main(project_name):
    """
    Função principal para treinar um modelo YOLOv8 para um projeto específico.
    O nome do projeto é usado para encontrar o arquivo de configuração e
    para nomear os resultados do treinamento.
    """
    # Verifica a disponibilidade de GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando o dispositivo: {device}")

    # Constrói o caminho para o arquivo data.yaml do projeto especificado
    project_path = os.path.join('Projetos', project_name)
    data_config_path = os.path.join(project_path, 'data.yaml')

    if not os.path.exists(data_config_path):
        print(f"Erro: Arquivo de configuração não encontrado em '{data_config_path}'")
        print(f"Certifique-se de que o projeto '{project_name}' existe e contém um arquivo data.yaml.")
        return

    # Carrega um modelo pré-treinado como base
    model = YOLO('yolov8n.pt')
    model.to(device)

    # Inicia o treinamento
    try:
        print(f"Iniciando o treinamento para o projeto: '{project_name}'...")
        results = model.train(
            data=data_config_path,
            epochs=50,
            imgsz=640,
            batch=-1,
            project=project_path,  # Salva os resultados dentro da pasta do projeto
            name='train'            # Nome da subpasta de resultados (ex: Projetos/cenario_policial/train)
        )
        print("Treinamento concluído com sucesso!")
        # O caminho do melhor modelo será algo como: Projetos/cenario_policial/train/weights/best.pt
        print(f"O melhor modelo está em: {results.save_dir}/weights/best.pt")

    except Exception as e:
        print(f"Ocorreu um erro durante o treinamento: {e}")

if __name__ == '__main__':
    # Configura o parser de argumentos da linha de comando
    parser = argparse.ArgumentParser(description="Script de Treinamento Genérico para YOLOv8")
    parser.add_argument(
        '--project',
        type=str,
        required=True,
        help="Nome da pasta do projeto a ser treinado (ex: 'cenario_policial')."
    )
    args = parser.parse_args()

    # Chama a função principal com o nome do projeto fornecido
    main(args.project)