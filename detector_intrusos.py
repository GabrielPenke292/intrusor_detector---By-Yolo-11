from ultralytics import YOLO
import cv2
import winsound  # lib para emitir som de alerta 

# Carregar o modelo YOLO pré-treinado
model = YOLO('yolo11n.pt')  # Modelo Nano
model.classes = [0]  # Detectar apenas pessoas (classe 0 no COCO Dataset)
model.conf = 0.5  # Configurar confiança mínima para detecção

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)  # Webcam 

# Verificar se a câmera foi inicializada corretamente
if not cap.isOpened():
    print("Erro ao acessar a câmera!")
    exit()

print("Sistema de detecção de intrusos iniciado. Pressione 'q' para encerrar.")

# Loop principal
while True:
    ret, frame = cap.read()  # Ler o frame da câmera
    if not ret:
        print("Erro ao capturar o frame.")
        break
    
    # Realizar a detecção com o YOLO11
    results = model.predict(frame, stream=True)  # Stream melhora o desempenho para múltiplos frames

    # Processar os resultados
    for box in results:
        # Extrair coordenadas do bounding box
        x1, y1, x2, y2 = map(int, box.boxes[0].xyxy[0])  # Coordenadas do retângulo
        confidence = box.boxes[0].conf[0]  # Confiança da detecção
        label = f"Pessoa {confidence:.2f}"  # Texto com a confiança
        
        # Desenhar o bounding box no frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Alerta sonoro (emite som ao detectar)
        winsound.Beep(2000, 500)  # Frequência: 2000 Hz, Duração: 500 ms

    # Mostrar o frame processado
    cv2.imshow("Detecção de Intrusos", frame)
    
    # Encerrar com a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
