# DeepSwap AI

Sistema de **Face Swap e Rastreamento Corporal em Tempo Real** inspirado no [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam).

## Funcionalidades
- Deteccao de 468 pontos faciais (MediaPipe FaceMesh)
- Rastreamento corporal com 33 pontos (MediaPipe Pose)
- Analise de expressoes: Feliz, Triste, Surpreso, Irritado, Neutro
- 6 avatares gerados por IA (personagens e jogos)
- Upload de avatar personalizado
- Snapshot para salvar fotos

## Como Usar
1. Clone o repositorio
2. Abra `index.html` em um servidor local (ex: `python -m http.server 8080`)
3. Acesse `http://localhost:8080`
4. Clique em **"Iniciar Camera"** e selecione um avatar

## Tecnologias
- MediaPipe FaceMesh + Pose
- Canvas 2D API (Affine Warp)
- HTML5 / CSS3 (Glassmorphism)
- JavaScript ES6+

## Estrutura
```
deepswap-ai/
|-- index.html    <- Interface principal
|-- style.css     <- Design dark glassmorphism
|-- app.js        <- Engine de face swap
-- assets/       <- Avatares gerados por IA
```

---
Criado com Antigravity AI