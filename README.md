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

# Para executar o projeto de detecção

python detector_universal.py --model Projetos/'nome_do_seu_projeto'/.../'nome_do_modelo'.pt --source 'diretorio_do_video'

# Exemplo 1: Usar o modelo policial para analisar um vídeo
python detector_universal.py --model Projetos/cenario_policial/train/weights/best.pt --source Videos/caso_01.mp4

# Exemplo 2: Usar o modelo de jogos para analisar uma captura de tela
python detector_universal.py --model Projetos/cenario_gaming/train/weights/best.pt --source Imagens/partida_102.jpg
