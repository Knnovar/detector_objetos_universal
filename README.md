## **Under Construction**
# detector_objetos_universal
Projeto pessoal cujo objetivo é, desenvolver um script genérico para treinamento de modelos de detecção de objetos com propósitos específicos, sejam eles objetos de fotos, vídeos, jogos ou em tempo real.

# Para Utilizar este projeto

Este projeto contém dois arquivos .py principais, sendo eles o detector universal.py e o treinar_modelo.py, é interessante salientar que, para o uso deste, é necessario entendimento pleno da estrutura de pastas, que são estas: 
> Extracao_Dados
> Imagens
> Projetos

> Videos 



# Para treinar um modelo 
Use este comando no terminal para treinar o modelo: python treinar_modelo.py --project 'nome_do_projeto'

# Exemplo 1: Usar o modelo policial para analisar um vídeo
python detector_universal.py --model projetos/cenario_policial/train/weights/best.pt --source videos_policiais/caso_01.mp4

# Exemplo 2: Usar o modelo de jogos para analisar uma captura de tela
python detector_universal.py --model projects/cenario_gaming/train/weights/best.pt --source screenshots/partida_102.jpg

# Exemplo 3: Usar o modelo de tráfego com a webcam em tempo real
python detector_universal.py --model projects/cenario_trafego/train/weights/best.pt --source 0
